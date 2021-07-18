from tqdm import tqdm
import torch
import numpy as np


mae_loss_fn = torch.nn.L1Loss()
mse_loss_fn = torch.nn.MSELoss()



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_metrics(model, data_loader):
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []
    valid_mape_loss = []
    for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in tqdm(data_loader):
        outputs, i_map = model(
            [solute_graphs.to(device), solvent_graphs.to(device), torch.tensor(solute_lens).to(device),
             torch.tensor(solvent_lens).to(device)])
        loss = mse_loss_fn(outputs, torch.tensor(labels).to(device).float())
        #print('single loss: ', loss)
        mae_loss = mae_loss_fn(outputs, torch.tensor(labels).to(device).float())
        mape_loss = np.mean(np.abs((np.array(labels) - np.array(outputs.cpu().detach())) / np.array(labels)))
        #print('single mape: ', mape_loss)
        valid_outputs += outputs.cpu().detach().numpy().tolist()
        valid_loss.append(loss.cpu().detach().numpy())
        valid_mae_loss.append(mae_loss.cpu().detach().numpy())
        valid_mape_loss.append(mape_loss)
        valid_labels += labels
        #print(labels)
        #print(outputs)
    #print('loss: ', np.array(valid_loss).flatten())
    #print('MAPE: ', np.array(valid_mape_loss).flatten())
    #print('loss: ', np.mean(np.array(valid_loss).flatten()))
    loss = np.mean(np.array(valid_loss).flatten())
    mape_loss = np.mean(np.array(valid_mape_loss).flatten())*100
    '''
    mape_loss = np.abs(((torch.FloatTensor([valid_labels])).flatten() - (torch.FloatTensor([valid_outputs])).flatten()) / (torch.FloatTensor([valid_labels])).flatten())
    mape_loss = np.array((mape_loss.mean(axis=0)) * 100)
    '''
    rmse_loss = np.array(np.sqrt(mse_loss_fn(torch.tensor([valid_labels]).to(device).float(), torch.squeeze(torch.FloatTensor([valid_outputs]), axis=0))))
    mae_loss = np.mean(np.array(valid_mae_loss).flatten())
    
    return loss, mae_loss, rmse_loss, mape_loss


def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, test_loader, project_name):
    best_val_loss = 100
    for epoch in range(max_epochs):
        model.train()
        running_loss = []
        tq_loader = tqdm(train_loader)
        o = {}
        for samples in tq_loader:
            optimizer.zero_grad()
            outputs, interaction_map = model(
                [samples[0].to(device), samples[1].to(device), torch.tensor(samples[2]).to(device),
                 torch.tensor(samples[3]).to(device)])
            l1_norm = torch.norm(interaction_map, p=2) * 1e-4
            loss = mse_loss_fn(outputs, torch.tensor(samples[4]).to(device).float()) + l1_norm
            loss.backward()
            optimizer.step()
            loss = loss - l1_norm
            running_loss.append(loss.cpu().detach())
            tq_loader.set_description(
                "Epoch: " + str(epoch + 1) + "  Training loss: " + str(np.mean(np.array(running_loss))))
        model.eval()
        val_loss, mae_loss, rmse_loss, mape_loss = get_metrics(model, valid_loader)
        scheduler.step(val_loss)
        print("Epoch: " + str(epoch + 1) + "  train_loss " + str(np.mean(np.array(running_loss))) + " MAE val_loss " + str(mae_loss) + " RMSE val_loss " + str(rmse_loss) + " MAPE val_loss " + str(mape_loss))
    
    if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./runs/run-" + str(project_name) + "/models/best_model.tar")
    
    test_loss, mae_loss, rmse_loss, mape_loss = get_metrics(model, test_loader)
    print("Test: ", "MAE " + str(mae_loss) + " RMSE " + str(rmse_loss) + " MAPE " + str(mape_loss))
            