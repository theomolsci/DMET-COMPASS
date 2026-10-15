"""Reduced density matrices of the optimised high-level wavefunction.

For every term ``a_p^ a_q`` / ``a_p^ a_q^ a_r a_s`` of the embedded Hamiltonian,
the expectation value is measured as the sum of its Jordan-Wigner Pauli
strings. Methodology step (e).

The original code ran one estimator job per distinct Pauli string, thousands
of jobs on the same state. Here all distinct strings are collected first and
evaluated in a *single* estimator job: Aer prepares the state once and saves
every Pauli expectation value from it. Each value is computed exactly as
before, so the RDMs are unchanged.
"""

import itertools

import numpy as np
from openfermion import FermionOperator, jordan_wigner
from qiskit.quantum_info import SparsePauliOp

from .circuits import get_estimator
from .operators import pauli_label


def _spin_sum(rdm_spin, n_mo):
    """Spin-orbital (interleaved) -> spatial-orbital RDM.

    Accumulates the spin blocks in the same order as an explicit loop over
    all spin-orbital indices, so the floating-point result is identical.
    """
    ndim = rdm_spin.ndim
    out = np.zeros((n_mo,) * ndim, dtype=np.complex128)
    for spins in itertools.product(range(2), repeat=ndim):
        out += rdm_spin[tuple(slice(s, None, 2) for s in spins)]
    return out


def measure_rdms(hamiltonian, n_qubits, n_mo, circuit, params):
    """Spin-summed 1- and 2-RDMs in the active space.

    Only elements that appear in the Hamiltonian are measured, because these
    are the only ones that enter the energy expression. The 2-RDM element of
    ``a_i^ a_j^ a_k a_l`` is stored at ``[i, l, j, k]`` (chemists' order).

    Returns:
        (np.ndarray, np.ndarray): ``(n_mo, n_mo)`` and ``(n_mo,)*4`` complex arrays.
    """
    # 1) Jordan-Wigner map every operator and collect the distinct Pauli strings.
    label_index = {}
    terms = []
    for key in hamiltonian.terms:
        if not key:
            continue
        qubit_op = jordan_wigner(FermionOperator(key))
        qubit_op.compress()
        contributions = []
        for pauli_term, coeff in qubit_op.terms.items():
            if pauli_term:
                label = pauli_label(pauli_term, n_qubits)
                contributions.append((label_index.setdefault(label, len(label_index)), coeff))
            else:
                contributions.append((None, coeff))
        terms.append((key, contributions))

    # 2) One estimator job for all Pauli strings.
    observables = [SparsePauliOp(label) for label in label_index]
    evs = get_estimator().run([(circuit, observables, params)]).result()[0].data.evs

    # 3) Assemble the spin-orbital RDMs.
    rdm1_spin = np.zeros((n_qubits,) * 2, dtype=complex)
    rdm2_spin = np.zeros((n_qubits,) * 4, dtype=complex)
    for key, contributions in terms:
        value = 0.0
        for index, coeff in contributions:
            value += coeff if index is None else np.real(coeff * evs[index])
        idx = [int(op[0]) for op in key]
        if len(key) == 2:
            rdm1_spin[idx[0], idx[1]] += value
        elif len(key) == 4:
            i, j, k, l = idx
            rdm2_spin[i, l, j, k] += value

    return _spin_sum(rdm1_spin, n_mo), _spin_sum(rdm2_spin, n_mo)
