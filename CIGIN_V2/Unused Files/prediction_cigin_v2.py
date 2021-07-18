#%%
# $python prediction_ciginv2.py -i ouput_pubchem_modified.csv -o ouput_pubchem_modified_test_3.csv -m runs/run-cigin_merged_11112020/models/best_model.tar

#python prediction_cigin_v2.py -i ouput_pubchem.csv -o ouput_pubchem_modified_test.csv -m runs/run-cigin/models/best_model.tar

import pandas as pd

import sys

from tqdm import tqdm

from rdkit import Chem

import dgl
from torch.utils.data import DataLoader, Dataset
import torch

from utils import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i','--input_file', required=True, help='Input CSV column name = Solvent,SolventSMILES,Solute,SoluteSMILES,DeltaGsolv')
parser.add_argument('-o','--output_file', required=True, help='output file name')
parser.add_argument('-m','--model_path', required=True, help='output file name with combinations')

arg=parser.parse_args()

input_file=arg.input_file
output_file=arg.output_file
model_path=arg.model_path


#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, './CIGIN_V2/')
#from molecular_graph import ConstructMolecularGraph
from model import CIGINModel
from molecular_graph import get_graph_from_smile

#%%
def get_graph(smile):

    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    smile = Chem.MolToSmiles(mol)
    smile_graph = get_graph_from_smile(smile,device)

    return smile_graph
#%%
model=CIGINModel().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

#%%

# solute = 'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)OC(C)(C)C)O)O)OC(=O)C6=CC=CC=C6)(CO4)OC(=O)C)OC)C)OC'
# output_list=[]
# solute_graph=get_graph(solute)
# input_data=[]
# input_data_cuda=[]
# #model=model.cuda()
# for i,smile in enumerate(input_df['SMILES'].to_list()):
    
#     solvent=smile
#     solvent_graph=get_graph(solvent)
#     input_data=[solute_graph,solvent_graph,len(solute),len(solvent)]
#     #create a list [solute_graph,solvent_graph,len(solute),len(solvent)]
#     input_data_cuda= [solute_graph.to(device), solvent_graph.to(device), torch.tensor(len(solute)).to(device),
#                  torch.tensor(len(solvent)).to(device)]
    
    
#     delta_g, interaction_map =  model(input_data_cuda)
    
#     print("Predicted free energy of solvation: ",str(delta_g.item()))

#     output_list.append(str(delta_g.item()))



#%%
# input_df['predicted_delta_g']=output_list
# input_df.to_csv('solvation_energy_cabazitaxel_CIGINv2_03112020.csv')

#%%
# #! Using exact main method() as in training

def collate(samples):
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    solute_len_matrix = get_len_matrix(solute_graphs.batch_num_nodes())
    solvent_len_matrix = get_len_matrix(solvent_graphs.batch_num_nodes())
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, labels


class Dataclass(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        solute = self.dataset.loc[idx]['SMILES']
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute)#,device)

        solvent = self.dataset.loc[idx]['SMILES']
        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)
        solvent_graph = get_graph_from_smile(solvent)#,device)
        
        delta_g = self.dataset.loc[idx]['DeltaGsolv']
        return [solute_graph, solvent_graph, [delta_g]]


#%%

# Make sure to use column name as Solvent,SolventSMILES,Solute,SoluteSMILES,DeltaGsolv
# Fill DeltaGsol with dummy values


valid_df = pd.read_csv(input_file)

#adding dummy column
valid_df['DeltaGsolv']=[0 for x in range(len(valid_df))]


valid_dataset = Dataclass(valid_df)
valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=4)

tq_loader = tqdm(valid_loader)
output_list=[]
output_list_temp = []
final_out=torch.empty(0,1).to(device)
for samples in tq_loader:
    outputs, interaction_map = model(
        [samples[0].to(device), samples[1].to(device), torch.tensor(samples[2]).to(device),
            torch.tensor(samples[3]).to(device)])

    # output_list_temp=(outputs.tolist())
    # output_list.append(output_list_temp)
    final_out = torch.cat((final_out,outputs))

print(len(final_out))

#remove brackets
out_list_cleaned=[]
out_list=final_out.tolist()
for i in out_list:
    in_str=str(i)
    in_str=in_str.replace('[','')
    in_str=in_str.replace(']','')
    out_list_cleaned.append(float(in_str))



valid_df['delta_g_cigin_v2_predicted(kcal/mol)']=out_list_cleaned
valid_df.drop(columns=['DeltaGsolv'])
valid_df.to_csv(output_file,index=False)


# %%
