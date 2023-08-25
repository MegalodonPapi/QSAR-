conda install -c jupycon/label/dev nb_conda_kernels
# Install RDKit.
!pip install rdkit-pypi
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

smiles_list = ['Oc1cccc2ccccc12',
'Clc1ccc(Cl)c(Cl)c1',
'Oc1ccc(Cl)c(Cl)c1Cl',
'CNC.OC(=O)COc1ccc(Cl)cc1Cl']
mol_list = []
for smiles in smiles_list:
  mol=Chem.MolFromSmiles(smiles)
  mol_list.append(mol)

img = Draw.MolsToGridImage(mol_list, molsPerRow = 4)  
img
