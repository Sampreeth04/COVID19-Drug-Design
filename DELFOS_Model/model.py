import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import rdkit

import mol2vec
from mol2vec import features
from mol2vec import helpers
from mol2vec.features import mol2alt_sentence
from gensim.models import Word2Vec

from rdkit import Chem
embedding_model = Word2Vec.load('model_300dim.pkl')

def data_embedder(df):

    df['SoluteMolecule'] = ''
    df['SoluteSentence'] = ''
    df['SoluteVector'] = ''

    df['SolventMolecule'] = ''
    df['SolventSentence'] = ''
    df['SolventVector'] = ''

    for i in range(len(df['SoluteSMILES'])):
        mol = Chem.MolFromSmiles(df['SoluteSMILES'].iloc[i])
        df['SoluteMolecule'].iloc[i] = mol
        df['SoluteSentence'].iloc[i] = mol2vec.features.mol2sentence(mol, 1)
        df['SoluteVector'].iloc[i] = mol2vec.features.sentences2vec(df['SoluteSentence'].iloc[i][0], embedding_model, unseen=None)

    for i in range(len(df['SolventSMILES'])):
        try:
            mol = Chem.MolFromSmiles(df['SolventSMILES'].iloc[i])
            df['SolventMolecule'].iloc[i] = mol
            df['SolventSentence'].iloc[i] = mol2vec.features.mol2sentence(mol, 1)
            df['SolventVector'].iloc[i] = mol2vec.features.sentences2vec(df['SolventSentence'].iloc[i][0], embedding_model, unseen=None)
        except:
            break
    
    #Concatenate Substructure Vectors
    for i in range(len(df['SoluteVector'])):
        df['SoluteVector'][i] = np.append(df['SoluteVector'][i][0], df['SoluteVector'][i][1])
        while len(df['SoluteVector'][i]) != 600:
            df['SoluteVector'][i] = np.append(df['SoluteVector'][i], np.zeros(1))

    for i in range(len(df['SolventVector'])):
        try:
            df['SolventVector'][i] = np.append(df['SolventVector'][i][0], df['SolventVector'][i][1])
            while len(df['SolventVector'][i]) != 600:
                df['SolventVector'][i] = np.append(df['SolventVector'][i], np.zeros(1))
        except:
            break

    input_solute = []
    for i in range(len(df['SoluteVector'])):
        for j in range(len(df['SoluteVector'][i])):
            input_solute.append(df['SoluteVector'][i][j])

    input_solvent = []
    for i in range(len(df['SolventVector'])):
        for j in range(len(df['SolventVector'][i])):
            input_solvent.append(df['SolventVector'][i][j])

    input_solvent = np.reshape(input_solvent, [-1, 1, 600])
    input_solute = np.reshape(input_solute, [-1, 1, 600]) 
    input_solvent = input_solvent.astype(np.float32)
    input_solute = input_solute.astype(np.float32)

    try:
        labels = df['DeltaGsolv']
        return input_solvent, input_solute, labels
        
    except:
        return input_solvent, input_solute
    
train_df = pd.read_csv('data/mnsol_train.csv', sep=';')
valid_df = pd.read_csv('data/mnsol_val.csv', sep=';')
test_df = pd.read_csv('data/mnsol_test.csv', sep=';')

input_solvent_train, input_solute_train, y_train = data_embedder(train_df)
input_solvent_val, input_solute_val, y_val = data_embedder(valid_df)
input_solvent_test, input_solute_test, y_test = data_embedder(test_df)

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

input_dim = 600
hidden_dim = 150
layer_dim = 1
output_dim = 1
batch_size = 1
learning_rate = 0.0002

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim, batch_size):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.batch_size = batch_size

        self.lstm_solvent = nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=True, bidirectional = True)
        self.lstm_solute = nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=True, bidirectional = True)

        self.fc1 = nn.Linear(hidden_dim*4, 2000)
        self.fc2 = nn.Linear(2000, output_dim)
        
    def forward(self, inputs_solv, inputs_solu):
        solvent_hidden_state = torch.zeros(self.num_layer*(2), self.batch_size, self.hidden_dim)
        solute_hidden_state = torch.zeros(self.num_layer*(2), self.batch_size, self.hidden_dim)
        solvent_cell_state = torch.zeros(self.num_layer*(2), self.batch_size, self.hidden_dim)
        solute_cell_state = torch.zeros(self.num_layer*(2), self.batch_size, self.hidden_dim)
        
        H, _ = self.lstm_solvent(inputs_solv, (solvent_hidden_state, solvent_cell_state))
        G, _ = self.lstm_solute(inputs_solu, (solute_hidden_state, solute_cell_state))

        H_size = H.size(1)
        G_size = G.size(1)
        a_score = torch.zeros(H_size, G_size)       
        for i in range(H_size):
            for j in range(G_size):
                a_score[i][j] = torch.matmul(H[0][i],torch.t(G[0][j]))
                
        
        a = F.softmax(a_score,1)
        
        G = torch.squeeze(G, axis=0)
        H = torch.squeeze(H, axis=0)
        
        P = torch.matmul(a, G)
        Q = torch.matmul(torch.t(a), H)
  
        u = torch.max(H, P)
        v = torch.max(G, Q)
        
        input_u = torch.sum(u, dim=0)
        input_v = torch.sum(v, dim=0)
        
        inp = torch.cat((input_u, input_v), 0)
        
        # mlp
        x = F.relu(self.fc1(inp)) 
        deltaGsolv = self.fc2(x)
      
        return deltaGsolv
    
model = Model(input_dim, hidden_dim, layer_dim, output_dim, batch_size)

nest_momentum = 0.9
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=nest_momentum, nesterov=True)
num_epochs = 75
best_model = model
best_loss = 10
losses = []

#Training and Validation

for epoch in range(num_epochs):       
        for i in range(input_solute_train.shape[0]):
            solute_train = input_solute_train[i]
            solvent_train = input_solvent_train[i]
            energy = y_train[i]
            energy = np.array(y_train[i])
            energy = energy.astype(np.float32)
            energy = torch.from_numpy(energy)

            output = model((torch.FloatTensor([solvent_train])), (torch.FloatTensor([solute_train])))
            loss = mse_criterion(output, energy)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        outputs_val=[]
        for i in range(input_solute_val.shape[0]):
            solute_val = input_solute_val[i]
            solvent_val = input_solvent_val[i]
            energy = y_train[i]
            energy = np.array(y_train[i])
            energy = energy.astype(np.float32)
            energy = torch.from_numpy(energy)

            output_val = model((torch.FloatTensor([solvent_val])), (torch.FloatTensor([solute_val])))
            outputs_val.append(output_val)
        
        val_loss = mse_criterion(torch.squeeze(torch.FloatTensor(y_val), axis=0), torch.squeeze(torch.FloatTensor([outputs_val]), axis=0))
        print('Validation MSE: {}'.format(val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            
#Testing

outputs = []

for i in range(input_solute_test.shape[0]):
            solute = input_solute_test[i]#train['SoluteVector'][i]
            solvent = input_solvent_test[i]#train['SolventVector'][i]

            # Forward pass
            output = best_model((torch.FloatTensor([solvent])), (torch.FloatTensor([solute])))
            print(i)
            print(output)
            print(y_test[i])
            outputs.append(output)

y_test = np.array(y_test)
y_test = y_test.astype(np.float32)
y_test = torch.from_numpy(y_test)

mape = np.abs((torch.squeeze(torch.FloatTensor(y_test), axis=0) - torch.squeeze(torch.FloatTensor([outputs]), axis=0)) / torch.squeeze(torch.FloatTensor(y_test), axis=0)).mean(axis=0) * 100
rmse = np.sqrt(mse_criterion(torch.squeeze(torch.FloatTensor(y_test), axis=0), torch.squeeze(torch.FloatTensor([outputs]), axis=0)))
mae = mae_criterion(torch.squeeze(torch.FloatTensor(y_test), axis=0), torch.squeeze(torch.FloatTensor([outputs]), axis=0))
print('MAPE: {}, RMSE: {}, MAE: {}'.format(mape, rmse, mae))


#Outputting COVID Solvent-Solute Matrix
solutes_df = pd.read_csv('COVID_files/solutes_df.csv')
solvents_df = pd.read_csv('COVID_files/solvents_df.csv')

solutes_df = solutes_df.rename(columns={'CANONICAL SMILE' : 'SoluteSMILES'})
solvents_df = solvents_df.rename(columns={'SMILES' : 'SolventSMILES'})

covid_df = pd.concat([solutes_df, solvents_df], axis=1)

solvent_vectors, solute_vectors = data_embedder(covid_df)

final_df = pd.DataFrame(index=solvents_df['NAME'], columns=solutes_df['Drug Name'])

for i in range(len(final_df.columns)):
    output_list = []
    for j in range(len(final_df)):
        output = model((torch.FloatTensor([solvent_vectors[j]])), (torch.FloatTensor([solute_vectors[i]])))
        output = output.detach().numpy()
        output_list.append(output[0])
    final_df[solutes_df['Drug Name'][i]] = output_list

final_df.to_csv('DeltaGSolv_matrix_DELFOS_final.csv')