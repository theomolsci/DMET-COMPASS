"""DMET-COMPASS for cyclo[10]carbon (C10), STO-3G, one carbon per fragment.

Full system: 100 qubits. Each embedded problem: (6e, 6o) active space,
12 qubits. The geometry is the polyynic (D5h) ring of radius 2.102 A.

All ten carbons are (to within the precision of the coordinates)
symmetry-equivalent, so, like the original script, only fragment 0 is
solved and its result is used for every fragment
(``equivalent_fragments=[0] * 10``). Pass ``equivalent_fragments=None`` to
solve all ten fragments explicitly.

Usage:
    python examples/c10_dmet_compass.py [--n-jobs N]
"""

import argparse
import os
import sys

# One BLAS/OpenMP thread per process: parallelism comes from worker processes,
# and single-threaded linear algebra makes every number bit-reproducible
# (multi-threaded BLAS changes the rounding of the PySCF steps at ~1e-14).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dmet_compass import CompassVQE, DMETCompass, Molecule  # noqa: E402

C10 = """
C   2.10200   0.00000   0.00
C   1.72186   1.20566   0.00
C   0.64955   1.99912   0.00
C   -0.61457   2.01015   0.00
C   -1.70055   1.23552   0.00
C   -2.10168   0.03668   0.00
C   -1.70055   -1.23552   0.00
C   -0.68434   -1.98748   0.00
C   0.64955   -1.99912   0.00
C   1.67873   -1.26502   0.00
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="worker processes (-1: all cores, 1: serial)")
    parser.add_argument("--all-fragments", action="store_true",
                        help="solve all 10 fragments instead of using the symmetry shortcut")
    args = parser.parse_args()

    molecule = Molecule(C10, basis="sto3g", charge=0, spin=0)
    solver = CompassVQE(double_threshold=1e-6, scatterer_threshold=1e-7, n_jobs=args.n_jobs)
    dmet = DMETCompass(molecule, fragment_atoms=[1] * 10, solver=solver,
                       equivalent_fragments=None if args.all_fragments else [0] * 10,
                       n_active_occupied=3, n_active_virtual=3)
    result = dmet.run()

    print("History [delta_N, E_DMET] per macro-cycle:")
    print(result.history)
    print(f"Converged chemical potential = {result.chemical_potential}")
    print(f"DMET-COMPASS energy (last cycle) = {result.energy.real} Ha")


if __name__ == "__main__":
    main()
