# Install nb_conda_kernels from the jupycon/label/dev channel in Conda.
# This package allows using Conda environments as Jupyter notebook kernels.
conda install -c jupycon/label/dev nb_conda_kernels

# Install the RDKit library using pip.
!pip install rdkit-pypi

# Import necessary modules from the RDKit library.
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

# List of SMILES strings representing chemical compounds.
smiles_list = ['Oc1cccc2ccccc12',
               'Clc1ccc(Cl)c(Cl)c1',
               'Oc1ccc(Cl)c(Cl)c1Cl',
               'CNC.OC(=O)COc1ccc(Cl)cc1Cl']

# Create an empty list to store RDKit molecule objects.
mol_list = []

# Iterate over each SMILES string and convert it to an RDKit molecule.
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    mol_list.append(mol)

# Generate a grid image from the list of molecule objects.
# 'molsPerRow' specifies the number of molecules to display in each row of the grid.
img = Draw.MolsToGridImage(mol_list, molsPerRow=4)

# Display the generated image.
img
