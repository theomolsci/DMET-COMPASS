"""Fermion/qubit operator utilities and the COMPASS operator pools.

Index conventions
-----------------
* Excitations are written in the Qiskit-Nature *blocked* spin-orbital
  convention: for ``n_mo`` spatial orbitals, alpha spin-orbitals are
  ``0 .. n_mo-1`` and beta spin-orbitals are ``n_mo .. 2*n_mo-1``. A tuple
  ``((i, j), (a, b))`` annihilates ``i, j`` and creates ``a, b``.
* Circuits use ``InterleavedQubitMapper(JordanWignerMapper())``.
* Hamiltonians come from OpenFermion (interleaved spin-orbitals, qubit ``q``)
  and are converted to Qiskit Pauli labels by writing qubit ``q`` at string
  position ``q``, i.e. OpenFermion qubit ``q`` becomes Qiskit qubit
  ``n_qubits - 1 - q``. This reversal is why the reference state sets the
  *last* ``n_electrons`` qubits (see :func:`reference_qubits`).

These conventions are exactly those of the original implementation, so all
energies and parameters are reproduced.
"""

import numpy as np
from openfermion import jordan_wigner
from qiskit.quantum_info import SparsePauliOp


def pauli_label(term, n_qubits):
    """OpenFermion Pauli term ``((q, 'X'), ...)`` -> Qiskit label string."""
    chars = ["I"] * n_qubits
    for qubit, pauli in term:
        chars[qubit] = pauli
    return "".join(chars)


def qubit_operator_to_sparse_pauli_op(qubit_op, n_qubits):
    """Convert an OpenFermion ``QubitOperator`` to a Qiskit ``SparsePauliOp``
    (real parts of the coefficients, term order preserved)."""
    labels, coeffs = [], []
    for term, coeff in qubit_op.terms.items():
        labels.append(pauli_label(term, n_qubits))
        coeffs.append(np.real(coeff))
    return SparsePauliOp(labels, coeffs)


def hamiltonian_to_sparse_pauli_op(fermion_hamiltonian, n_qubits):
    """Jordan-Wigner map a fermionic Hamiltonian to a ``SparsePauliOp``."""
    return qubit_operator_to_sparse_pauli_op(jordan_wigner(fermion_hamiltonian), n_qubits)


def reference_qubits(n_qubits, n_electrons):
    """Qubits flipped to |1> to prepare the mean-field reference |phi_ref>."""
    return list(range(n_qubits - n_electrons, n_qubits))


# --------------------------------------------------------------------------- #
#  Scatterer pools (paper Sec. 2.3.1 and 4.1, "partial-pairing" pool)
# --------------------------------------------------------------------------- #

def scatterer_pools(n_mo, n_occ):
    """Build the partial-pairing S_h and S_p scatterer pools.

    The contractible set of orbitals (CSOs) is restricted to HOMO and LUMO.

    * ``S_h`` (hole scatterer): both annihilated electrons come from the same
      spatial orbital ``p`` (``p != HOMO``); one goes to a virtual ``a`` and the
      other is scattered occupied->occupied into the HOMO of the opposite spin::

          ((p_s, p_s'), (a_s, HOMO_s'))

    * ``S_p`` (particle scatterer): both created electrons go into the same
      spatial virtual ``a`` (``a != LUMO``); one comes from an occupied ``p``,
      the other is scattered virtual->virtual out of the LUMO of the opposite
      spin::

          ((p_s, LUMO_s'), (a_s, a_s'))

    For a (6e, 6o) active space this reproduces, element by element and in the
    same order, the ``Sh_list`` / ``Sp_list`` hard-coded in the original script.

    Args:
        n_mo: Number of active spatial orbitals.
        n_occ: Number of doubly occupied active spatial orbitals.

    Returns:
        (list, list): ``S_h`` pool and ``S_p`` pool.
    """
    occ = range(n_occ)
    virt = range(n_occ, n_mo)
    homo, lumo = n_occ - 1, n_occ
    spin_offsets = ((0, n_mo), (n_mo, 0))  # (spin s, opposite spin s')

    s_h = [((p + s, p + s_), (a + s, homo + s_))
           for s, s_ in spin_offsets for p in occ if p != homo for a in virt]
    s_p = [((p + s, lumo + s_), (a + s, a + s_))
           for s, s_ in spin_offsets for p in occ for a in virt if a != lumo]
    return s_h, s_p


def couples_hole_scatterer(double, scatterer):
    """Non-commutativity test between a double ``T_ij^ab`` and an ``S_h``.

    The scatterer must create into an orbital that the double annihilates
    (common CSO), while sharing no other index with it.
    """
    (i, j), (a, b) = double
    (p, q), (c, d) = scatterer
    return ((i == d or j == d)
            and p not in (i, j) and q not in (i, j) and c not in (a, b))


def couples_particle_scatterer(double, scatterer):
    """Non-commutativity test between a double ``T_ij^ab`` and an ``S_p``.

    The scatterer must annihilate an orbital that the double creates into
    (common CSO), while sharing no other index with it.
    """
    (i, j), (a, b) = double
    (p, q), (c, d) = scatterer
    return ((a == q or b == q)
            and a not in (c, d) and b not in (c, d) and p not in (i, j))
