"""DMET-COMPASS: density matrix embedding with a commutativity-based,
dynamically constructed UCC ansatz for each embedded subsystem.

Reference: D. Mondal, A. K. Patra, R. Maitra, "Advancing practical quantum
embedding simulations via operator commutativity-based state preparation for
complex chemical systems", Chem. Sci. (2026), DOI: 10.1039/d6sc03132c.
"""

__author__ = "Dibyendu Mondal"

from .compass import CompassVQE, VQEResult
from .dmet import DMETCompass, DMETResult
from .molecule import Molecule
from .operators import scatterer_pools

__all__ = ["CompassVQE", "VQEResult", "DMETCompass", "DMETResult", "Molecule",
           "scatterer_pools"]
