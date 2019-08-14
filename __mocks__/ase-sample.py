import ase.visualize
from ase.build import molecule

# Create the methanol molecule
methanol = molecule('CH3OH')

ase.visualize.view(methanol, viewer="x3d")

