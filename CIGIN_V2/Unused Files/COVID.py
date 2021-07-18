import random
import pandas as pd
import pickle
import os
import sys
import warnings
from collections import OrderedDict
from copy import deepcopy
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from tqdm import tqdm
 
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger
import dgl
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
#from utils import *
 
#device = "cuda" if torch.cuda.is_available() else "cpu"
 
sys.path.insert(0, 'C:/Users/Sampr/Downloads/Cigin-master/cigin-master/CIGIN_V2')
#from molecular_graph import ConstructMolecularGraph
from model import CIGINModel
from molecular_graph import get_graph_from_smile
 
 
def get_graph(smile):
 
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    smile = Chem.MolToSmiles(mol)
    smile_graph = get_graph_from_smile(smile)#, cuda)
 
    return smile_graph
 
model=CIGINModel()#.to(device)
model.load_state_dict(torch.load('C:/Users/Sampr/Downloads/Cigin-master/cigin-master/CIGIN_V2/runs/run-cigin/models/best_model.tar'))
model.eval()
 
 
#input solvent list
input_df=pd.read_csv('C:/Users/Sampr/Downloads/Cigin-master/cigin-master/CIGIN_V2/COVID_files/ouput_pubchem.csv')

solute = 'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)OC(C)(C)C)O)O)OC(=O)C6=CC=CC=C6)(CO4)OC(=O)C)OC)C)OC'
output_list=[]
solute_graph=get_graph(solute)
input_data=[]
#input_data_cuda=[]
#model=model.cuda()

for i,smile in enumerate(input_df['SMILES'].to_list()):
    
    solvent=smile
    solvent_graph=get_graph(solvent)
    print("Solvent length is: ", len(solvent))
    print("Solute length is: ", len(solute))
    
    #print(i, smile)
    
    x = torch.Tensor(len(solute))
    y = torch.Tensor(len(solvent))
    input_data=[solute_graph,solvent_graph,x,y]
    
    print("Solute tensor dimensions: ", x.ndimension())
    print("Solvent tensor dimensions: ", y.ndimension())
    
    print(x)
    print(y)
    
    #create a list [solute_graph,solvent_graph,len(solute),len(solvent)]
 
    #input_data_cuda= [solute_graph.to(device), solvent_graph.to(device), torch.tensor(len(solute)).to(device),
    #             torch.tensor(len(solvent)).to(device)]
    
    delta_g, interaction_map =  model(input_data)
    
    print("Predicted free energy of solvation: ",str(delta_g))
 
    output_list.append(str(delta_g.item()))



 
#%%
#input_df['predicted_delta_g']=output_list
#input_df.to_csv('solvation_energy_cabazitaxel_CIGINv2_03112020.csv')