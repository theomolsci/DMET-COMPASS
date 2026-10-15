"""Low-level (mean-field) description of the full system.

Methodology step (a) of the paper: an RHF calculation on the whole molecule
provides the MO coefficients and the electron count that DMET partitions.
The localisation function used for step (b) (meta-Lowdin) is attached here
as well, because Tangelo's ``dmet_orbitals`` expects it as a callable
``f(mol, mf) -> C_loc``.
"""

from pyscf import gto, scf
from pyscf.lo import orth


def meta_lowdin(mol, mf):
    """Meta-Lowdin orthogonalised AOs, used as the localized fragment basis."""
    return orth.orth_ao(mol, "meta_lowdin")


class Molecule:
    """A PySCF molecule together with its converged RHF mean field.

    Args:
        atom: Geometry as an xyz string or a ``[[symbol, (x, y, z)], ...]`` list.
        basis: Gaussian basis set name.
        charge: Total charge.
        spin: 2S (number of unpaired electrons). Only closed shells (0) are
            supported by the DMET driver.
        verbose: PySCF verbosity of the SCF run.
    """

    def __init__(self, atom, basis="sto3g", charge=0, spin=0, verbose=0):
        if not isinstance(atom, (str, list)):
            raise ValueError("Give the geometry as an xyz string or as "
                             "[['atom_name', (x, y, z)], ...].")
        self.mol = gto.Mole(atom=atom, spin=spin, charge=charge, basis=basis)
        self.mol.build()

        self.mf = scf.RHF(self.mol)
        self.mf.verbose = verbose
        self.energy = self.mf.kernel()

        self.localization = meta_lowdin

    @property
    def n_electrons(self):
        return self.mol.nelectron

    @property
    def nao(self):
        return self.mol.nao_nr()
