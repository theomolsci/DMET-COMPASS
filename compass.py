"""COMPASS: COMmutativity Pre-screened Automated Selection of Scatterers.

Dynamic ansatz construction for one embedded subsystem (paper Sec. 2.3.2 and
Methodology step (d)):

  i.   One-parameter micro-cycles (eqn 11). Every two-body excitation
       ``T_I`` is optimised alone on top of |phi_ref>. Doubles with
       ``|E_ref - E_I| > double_threshold`` are kept and ordered by
       decreasing energy gain. Each one starts its own operator block.
  ii.  Commutativity pre-screening. For each kept double, candidate
       scatterers from the S_h / S_p pools that share a CSO (HOMO/LUMO) with
       it, and so do not commute with it, are listed.
  iii. Two-parameter micro-cycles (eqn 13). Each (double, scatterer) pair is
       optimised. Scatterers with ``|E_I - E_In| > scatterer_threshold`` join
       the block of that double.
  iv.  Global VQE (eqns 14-15). All blocks, followed by every single
       excitation, are optimised together, starting from the micro-cycle
       parameters (0 for the singles).

Parallelism: all micro-cycles of step i are independent, and so are all
micro-cycles of step iii (each starts from the step-i optimum of its double).
Both steps run in parallel over processes. The finite-difference gradient of
step iv is also evaluated in parallel through SciPy's ``workers`` hook. Results
are gathered in the original order, so the ansatz, the printed log and every
number are those of the serial algorithm.
"""

import inspect
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .circuits import EnergyFunction, excitation_list, ucc_circuit
from .operators import (couples_hole_scatterer, couples_particle_scatterer,
                        hamiltonian_to_sparse_pauli_op, scatterer_pools)
from .parallel import CountingMap, pmap, resolve_n_jobs

# SciPy >= 1.16 can evaluate SLSQP's finite-difference gradient through a
# user-supplied map (``workers``); older versions fall back to serial.
try:
    from scipy.optimize._slsqp_py import _minimize_slsqp
    _SLSQP_HAS_WORKERS = "workers" in inspect.signature(_minimize_slsqp).parameters
except (ImportError, ValueError, TypeError):
    _SLSQP_HAS_WORKERS = False


@dataclass
class VQEResult:
    energy: float
    params: np.ndarray
    circuit: object
    excitations: list
    n_evaluations: int


@dataclass
class _Problem:
    """What a worker process needs to build and evaluate a fragment circuit."""
    observable: object
    n_mo: int
    n_particles: tuple
    n_qubits: int
    n_electrons: int

    def circuit(self, excitations):
        return ucc_circuit(excitations, self.n_mo, self.n_particles,
                           self.n_qubits, self.n_electrons)


def _micro_cycle(problem, excitations, x0, start_params):
    """Optimise a small (1- or 2-parameter) circuit with SLSQP.

    Runs in a worker process. Returns (E_start, E_opt, theta_opt, n_evals).
    """
    energy = EnergyFunction(problem.circuit(excitations), problem.observable)
    e_start = energy.evaluate(start_params)
    res = minimize(energy, x0=x0, method="slsqp")
    return e_start, res.fun, res.x, energy.nfev


def _screen_double(args):
    problem, double = args
    return _micro_cycle(problem, [double], np.zeros((1)), [0])


def _screen_scatterer(args):
    problem, double, scatterer, theta_double = args
    x0 = [theta_double] + [0.0]
    return _micro_cycle(problem, [double, scatterer], x0, x0)


@dataclass
class CompassVQE:
    """High-level solver of an embedded subsystem with the COMPASS ansatz.

    Args:
        double_threshold: Energy threshold for keeping a double (eqn 11).
            The paper quotes 1e-5 Ha; the reference implementation (and this
            default) uses 1e-6 Ha.
        scatterer_threshold: Energy threshold for keeping a scatterer
            (eqn 13), 1e-7 Ha.
        n_jobs: Number of worker processes (-1 = all cores, 1 = serial).
        verbose: Print the screening log.
    """
    double_threshold: float = 1e-6
    scatterer_threshold: float = 1e-7
    n_jobs: int = -1
    verbose: bool = True
    history: list = field(default_factory=list)

    def _print(self, *args):
        if self.verbose:
            print(*args)

    # ------------------------------------------------------------------ #
    def solve(self, hamiltonian, n_mo, n_electrons, n_qubits, label=1):
        """Build and optimise the ansatz for one embedded Hamiltonian.

        Args:
            hamiltonian: Embedded Hamiltonian (OpenFermion ``FermionOperator``).
            n_mo, n_electrons, n_qubits: Active-space dimensions.
            label: Fragment number used in the printed log.
        """
        observable = hamiltonian_to_sparse_pauli_op(hamiltonian, n_qubits)
        problem = _Problem(observable, n_mo, (n_electrons // 2, n_electrons // 2),
                           n_qubits, n_electrons)
        self._print("Total Spin orbitals", n_qubits)
        self._print("Total Electrons", n_electrons)
        if n_qubits == 4:
            result = self._solve_minimal(problem, label)
        else:
            result = self._solve_compass(problem, label)
        self.history.append(result)
        return result

    # ------------------------------------------------------------------ #
    def _solve_minimal(self, problem, label):
        """(2e, 2o) case: fixed 3-parameter ansatz (one double, two singles),
        optimised with L-BFGS-B. As in the reference code, the returned
        parameters are those of the last energy evaluation."""
        excitations = [((0, 2), (1, 3)), ((0,), (1,)), ((2,), (3,))]
        circuit = problem.circuit(excitations)
        energy = EnergyFunction(circuit, problem.observable)
        res = minimize(energy, x0=np.zeros((3)), method="l-bfgs-b")
        self._print(f"Fragment {label} # of iterations = {energy.nfev}")
        self._print(f"Fragment {label} Final Energy = {res.fun}")
        return VQEResult(res.fun, energy.last_params, circuit, excitations, energy.nfev)

    # ------------------------------------------------------------------ #
    def _solve_compass(self, problem, label):
        n_jobs = resolve_n_jobs(self.n_jobs)
        n_evaluations = 0

        # ---- step i: one-parameter micro-cycles over all doubles ----------
        doubles = excitation_list(problem.n_mo, problem.n_particles, "d")
        screened = pmap(_screen_double, [(problem, d) for d in doubles], n_jobs)

        selected = []      # (|dE|, double, E_I, theta_I) in pool order
        for double, (e_start, e_opt, x, nfev) in zip(doubles, screened):
            n_evaluations += nfev
            diff = e_opt - e_start
            self._print(f"Excitation = {double} E_ini = {e_start} E_fin = {e_opt} "
                        f"Difference = {diff} Optimal Parameter = {x[0]}")
            if np.abs(diff) > self.double_threshold:
                selected.append((abs(diff), double, e_opt, x[0]))

        self._print("Differences Energy", [s[0] for s in selected])
        # Operator-block ordering: decreasing energy gain (stable for ties).
        selected.sort(key=lambda s: s[0], reverse=True)
        self._print("Sorted Differences Energy", [s[0] for s in selected])
        self._print("Sorted Doubles List", [s[1] for s in selected])

        # ---- step ii: commutativity pre-screening of scatterers -----------
        s_h, s_p = scatterer_pools(problem.n_mo, problem.n_particles[0])
        candidates = [[sc for sc in s_h if couples_hole_scatterer(double, sc)]
                      + [sc for sc in s_p if couples_particle_scatterer(double, sc)]
                      for _, double, _, _ in selected]

        # ---- step iii: two-parameter micro-cycles (all pairs in parallel) --
        tasks = [(problem, double, sc, theta)
                 for (_, double, _, theta), cands in zip(selected, candidates)
                 for sc in cands]
        pair_results = iter(pmap(_screen_scatterer, tasks, n_jobs))

        excitations, params = [], []
        for (_, double, _, theta), cands in zip(selected, candidates):
            excitations.append(double)
            params.append(theta)
            block_scatterers = []
            for sc in cands:
                e_start, e_opt, x, nfev = next(pair_results)
                n_evaluations += nfev
                self._print(sc)
                self._print("E-diff", e_opt - e_start)
                if np.abs(e_opt - e_start) > self.scatterer_threshold:
                    excitations.append(sc)
                    params.append(x[-1])
                    block_scatterers.append(sc)
            self._print("Doubles", double, "Corresponding S", block_scatterers)

        # ---- step iv: global VQE over blocks + all singles ----------------
        singles = excitation_list(problem.n_mo, problem.n_particles, "s")
        excitations = excitations + singles
        params = params + [0.0] * len(singles)

        circuit = problem.circuit(excitations)
        energy = EnergyFunction(circuit, problem.observable)
        options, workers = {}, None
        if n_jobs > 1 and _SLSQP_HAS_WORKERS:
            workers = CountingMap(n_jobs)
            options["workers"] = workers
        res = minimize(energy, x0=params, method="slsqp", options=options)
        n_evaluations += energy.nfev + (workers.count if workers else 0)

        self._print(f"Fragment {label} # of iterations = {n_evaluations}")
        self._print(f"Fragment {label} Final Energy = {res.fun}")
        self._print("No of parameters", len(excitations))
        self._print("---------------------------------------------------------------------")
        return VQEResult(res.fun, res.x, circuit, excitations, n_evaluations)
