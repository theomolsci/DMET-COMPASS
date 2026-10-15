"""DMET driver: fragments, bath, embedded Hamiltonians and the mu-loop.

Methodology (paper Sec. 3):

  (a) RHF of the full system                         -> :class:`Molecule`
  (b) atom-centred fragments, meta-Lowdin orbitals   -> ``dmet_orbitals``
  (c) bath orbitals from the environment block of the mean-field 1-RDM
      (eigenvalues in (1e-13, 2) are kept)           -> ``dmet_fragment_bath``
  (d) embedded Hamiltonian H_emb(mu) (eqn 5): the chemical potential shifts
      the fragment diagonal of the embedded Fock matrix; an embedded RHF
      provides |phi_ref>; COMPASS-VQE gives |psi(theta_opt)>
  (e) fragment energy E^x (eqn 6) and electron number from the RDMs
  (f) DMET self-consistency: find mu such that  N_act - sum_x N^x(mu) = 0
      with the Newton-secant method (tolerance 1e-5).

The fragment/bath machinery comes from Tangelo, as in the paper.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import newton
from tangelo.problem_decomposition.dmet._helpers.dmet_bath import dmet_fragment_bath
from tangelo.problem_decomposition.dmet._helpers.dmet_fragment import dmet_fragment_constructor
from tangelo.problem_decomposition.dmet._helpers.dmet_onerdm import (dmet_fragment_rdm,
                                                                     dmet_low_rdm_rhf)
from tangelo.problem_decomposition.dmet._helpers.dmet_orbitals import dmet_orbitals
from tangelo.problem_decomposition.dmet._helpers.dmet_scf import dmet_fragment_scf_rhf
from tangelo.problem_decomposition.dmet._helpers.dmet_scf_guess import dmet_fragment_guess_rhf
from tangelo.problem_decomposition.dmet.fragment import SecondQuantizedDMETFragment
from tangelo.toolboxes.molecular_computation.rdms import pad_rdms_with_frozen_orbitals_restricted

from .compass import CompassVQE
from .rdm import measure_rdms


def select_frozen_orbitals(mo_occ, n_occupied=3, n_virtual=3):
    """Freeze everything except the ``n_occupied`` highest occupied and the
    ``n_virtual`` lowest virtual orbitals of the embedded mean field.
    Returns ``None`` (nothing frozen) for embedded problems with <= 2 MOs."""
    if len(mo_occ) <= 2:
        return None
    homo = np.where(mo_occ > 0)[0][-n_occupied:]
    lumo = np.where(mo_occ == 0)[0][:n_virtual]
    active = list(homo) + list(lumo)
    return [i for i in range(len(mo_occ)) if i not in active]


def fragment_energy(fragment, onerdm, twordm):
    """Fragment energy, eqn (6), from RDMs in the embedded MO basis.

    The RDMs are rotated to the embedding (fragment + bath) basis. Only
    elements with at least one index on the fragment are counted, each with
    the democratic-partitioning weight.

    Returns:
        (energy, onerdm in the embedding basis)
    """
    fock = fragment.fock
    oneint = fragment.one_ele
    twoint = fragment.two_ele
    c = fragment.mean_field.mo_coeff
    nf = fragment.t_list[0]

    onerdm = c @ onerdm @ c.T
    twordm = np.einsum("pi,ijkl->pjkl", c, twordm)
    twordm = np.einsum("qj,pjkl->pqkl", c, twordm)
    twordm = np.einsum("rk,pqkl->pqrl", c, twordm)
    twordm = np.einsum("sl,pqrl->pqrs", c, twordm)

    e1 = np.einsum("ij,ij->", onerdm[:nf, :], fock[:nf, :] + oneint[:nf, :]) \
        + np.einsum("ij,ij->", onerdm[:, :nf], fock[:, :nf] + oneint[:, :nf])
    e1 *= 0.25

    e2 = np.einsum("ijkl,ijkl->", twordm[:nf, :, :, :], twoint[:nf, :, :, :]) \
        + np.einsum("ijkl,ijkl->", twordm[:, :nf, :, :], twoint[:, :nf, :, :]) \
        + np.einsum("ijkl,ijkl->", twordm[:, :, :nf, :], twoint[:, :, :nf, :]) \
        + np.einsum("ijkl,ijkl->", twordm[:, :, :, :nf], twoint[:, :, :, :nf])
    e2 *= 0.125

    return e1 + e2, onerdm


@dataclass
class DMETResult:
    chemical_potential: float
    energy: complex                 # DMET energy of the last macro-cycle
    history: list = field(default_factory=list)   # [delta_N, E] per macro-cycle


class DMETCompass:
    """DMET with COMPASS-VQE as the high-level solver.

    Args:
        molecule: A :class:`~dmet_compass.molecule.Molecule`.
        fragment_atoms: Number of atoms in each fragment, in geometry order
            (e.g. ``[1] * 10`` for one carbon per fragment in C10).
        solver: High-level solver (default: :class:`CompassVQE`).
        equivalent_fragments: Optional list giving, for every fragment, the
            index of a symmetry-equivalent fragment whose result is reused.
            Only the distinct representatives are solved. ``[0] * 10``
            reproduces the original C10 script, which solves fragment 0 and
            uses its energy and electron count for all ten carbons.
            ``None`` (default) solves every fragment.
        n_active_occupied, n_active_virtual: Active space of each embedded
            problem (3, 3 gives the (6e, 6o) / 12-qubit space of the paper).
        chemical_potential: Starting guess for mu.
        virtual_orbital_threshold: Bath eigenvalue threshold (1e-13).
        newton_tol, newton_maxiter: Settings of the Newton-secant mu-solver.
        verbose: Print progress.
    """

    def __init__(self, molecule, fragment_atoms, solver=None, equivalent_fragments=None,
                 n_active_occupied=3, n_active_virtual=3, chemical_potential=0.0,
                 virtual_orbital_threshold=1e-13, newton_tol=1e-5, newton_maxiter=100,
                 verbose=True):
        self.molecule = molecule
        self.solver = solver if solver is not None else CompassVQE(verbose=verbose)
        self.n_active_occupied = n_active_occupied
        self.n_active_virtual = n_active_virtual
        self.chemical_potential = chemical_potential
        self.virtual_orbital_threshold = virtual_orbital_threshold
        self.newton_tol = newton_tol
        self.newton_maxiter = newton_maxiter
        self.verbose = verbose
        self.cycle = 0
        self.history = []

        # (b) fragments and localized orbitals, (c) mean-field 1-RDM.
        mol, mf = molecule.mol, molecule.mf
        self.fragment_orbitals, self.fragment_ao_ranges, _ = \
            dmet_fragment_constructor(mol, fragment_atoms, 0)
        self.orbitals = dmet_orbitals(mol, mf, range(mol.nao_nr()), molecule.localization, False)
        self.onerdm_low = dmet_low_rdm_rhf(self.orbitals.active_fock,
                                           self.orbitals.number_active_electrons)

        n_frag = len(self.fragment_orbitals)
        self.equivalent_fragments = (list(range(n_frag)) if equivalent_fragments is None
                                     else list(equivalent_fragments))
        if len(self.equivalent_fragments) != n_frag:
            raise ValueError("equivalent_fragments needs one entry per fragment.")

    def _print(self, *args):
        if self.verbose:
            print(*args)

    # ------------------------------------------------------------------ #
    def build_fragment(self, i, chemical_potential):
        """Bath, embedded Hamiltonian and embedded RHF for fragment ``i``
        (steps (c)-(d))."""
        t_list = [self.fragment_orbitals[i]]
        bath_orb, e_occupied = dmet_fragment_bath(
            self.orbitals.mol_full, t_list, self.fragment_ao_ranges[i],
            self.onerdm_low, self.virtual_orbital_threshold, False)
        norb_high, nelec_high, onerdm_high = dmet_fragment_rdm(
            t_list, bath_orb, e_occupied, self.orbitals.number_active_electrons)
        one_ele, fock, two_ele = self.orbitals.dmet_fragment_hamiltonian(
            bath_orb, norb_high, onerdm_high)
        guess = dmet_fragment_guess_rhf(t_list, bath_orb, chemical_potential, norb_high,
                                        nelec_high, self.orbitals.active_fock)
        mf_frag, fock_frag_copy, mol_frag = dmet_fragment_scf_rhf(
            t_list, two_ele, fock, nelec_high, norb_high, guess, chemical_potential)

        frozen = select_frozen_orbitals(mf_frag.mo_occ, self.n_active_occupied,
                                        self.n_active_virtual)
        return SecondQuantizedDMETFragment(mol_frag, mf_frag, fock, fock_frag_copy, t_list,
                                           one_ele, two_ele, False, frozen_orbitals=frozen)

    def solve_fragment(self, i, chemical_potential):
        """High-level solution of fragment ``i``: (energy E^x, electrons N^x)."""
        self._print(f"Fragment Number {i + 1}")
        fragment = self.build_fragment(i, chemical_potential)
        hamiltonian = fragment.fermionic_hamiltonian   # property: build it once

        result = self.solver.solve(hamiltonian, fragment.n_active_mos,
                                   fragment.n_active_electrons, fragment.n_active_sos,
                                   label=i + 1)

        start = time.time()
        onerdm, twordm = measure_rdms(hamiltonian, fragment.n_active_sos,
                                      fragment.n_active_mos, result.circuit, result.params)
        self._print(f"RDMs measured in {time.time() - start:.2f} seconds")

        onerdm, twordm = pad_rdms_with_frozen_orbitals_restricted(fragment, onerdm, twordm)
        energy, onerdm = fragment_energy(fragment, onerdm, twordm)
        n_electrons = np.trace(onerdm[:fragment.t_list[0], :fragment.t_list[0]])
        self._print(f"fragment no - {i + 1} = {energy}")
        self._print(f"No of electrons in fragment = {n_electrons}")
        return energy, n_electrons

    # ------------------------------------------------------------------ #
    def macro_cycle(self, chemical_potential):
        """One DMET cycle at fixed mu. Returns delta_N = N_act - sum_x N^x(mu)."""
        start = time.time()
        self.cycle += 1
        self._print(f"DMET Computation Cycle No {self.cycle}")
        self._print("Chemical Potential", chemical_potential)
        self._print("------------------------------------------------------------")

        solved = {r: self.solve_fragment(r, chemical_potential)
                  for r in sorted(set(self.equivalent_fragments))}
        results = np.array([solved[r] for r in self.equivalent_fragments])

        energy = np.sum(results[:, 0]) + self.orbitals.core_constant_energy
        n_electrons = np.sum(results[:, 1])
        delta_n = self.orbitals.number_active_electrons - n_electrons
        self.history.append([delta_n, energy])
        self._print(f"Electron number error = {delta_n}  DMET energy = {energy}")
        self._print(f"Time Taken = {time.time() - start} seconds")
        return delta_n

    def run(self):
        """Converge the global chemical potential (Newton-secant)."""
        start = time.time()
        mu = newton(self.macro_cycle, self.chemical_potential,
                    tol=self.newton_tol, maxiter=self.newton_maxiter)
        self.chemical_potential = mu.real
        self._print(f"Chemical Potential = {mu.real}")
        self._print(f"Total Time Taken is {time.time() - start} seconds")
        return DMETResult(mu.real, self.history[-1][1], self.history)
