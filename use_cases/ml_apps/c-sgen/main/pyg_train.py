import torch

from torch_geometric.data import Data, InMemoryDataset

import pickle
import numpy as np
import timeit
import os
import wandb

from torch_geometric.data import DataLoader
from sklearn.metrics import mean_squared_error
from torch_geometric.transforms import AddSelfLoops
from utils import GAT, AGNN, SGC, ARMA

project = "pyg-train"

wandb.init(project=project)
wandb.log({"run_dir": wandb.run.dir})

def rms_score(y_true, y_pred):
    """Computes RMS error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

class TestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(TestDataset, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass

def load_dataset(dataset):

    with open('./dataset/' + dataset +'/full_feature','rb') as node_features:
        x_train = pickle.load(node_features)
    with open('./dataset/' + dataset +'/edge','rb') as f:
        edge_index_train = pickle.load(f)
    y_train = load_tensor('./dataset/' + dataset +'/Interactions', torch.FloatTensor)

    d = []
    for i in range(len(y_train)):
        data = Data(x=x_train[i], edge_index=edge_index_train[i], y=y_train[i])
        data = AddSelfLoops()(data)
        data.atom_num = x_train[i].shape[0]
        d.append(data)
    set = TestDataset(d)
    return set

# FreeSolv
std = 3.8448222046029543
mean = -3.8030062305295975

class Trainer(object):

    def __init__(self, model, std, mean):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        self.std = std
        self.mean = mean

    def train(self, train_loader):

        loss_total = 0
        num = 0
        for data in train_loader:
            num += 1
            data = data.to(device)
            self.optimizer.zero_grad()
            loss = self.model(data, std=self.std, mean=self.mean, train=True)
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()

        loss_mean = loss_total / num
        return loss_mean

class tester(object):

    def __init__(self, model, std, mean):
        self.model = model
        self.std = std
        self.mean = mean

    def test(self, test_loader):

        loss_total = 0
        all_p = []
        all_t = []
        num = 0
        for data in test_loader:
            num += 1
            data = data.to(device)
            loss, predicted, true = self.model(data, std=self.std, mean=self.mean, train=False)

            for i in predicted:
                all_p.append(float(i))
            for i in true:
                all_t.append(float(i))
            loss_total += loss.to('cpu').data.numpy()

        RMSE = rms_score(all_t,all_p)
        loss_mean = loss_total / num
        return loss_mean, RMSE

def metric(RMSE_k_test):
    RMSE_mean_test = np.mean(np.array(RMSE_k_test))
    RMSE_std_test = np.std(np.array(RMSE_k_test))

    return RMSE_mean_test, RMSE_std_test

device = torch.device('cuda')

batch = 8
iteration = 50
lr = 0.01
decay_interval = 10
lr_decay = 0.5

print('decay_interval:', decay_interval)
print('lr:', lr)

train_dataset = load_dataset('train')
valid_dataset = load_dataset('valid')
test_dataset = load_dataset('test')

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

seed_list = [256, 512, 1024]

RMSE_k_train = []
R2_k_train = []

RMSE_k_valid = []
R2_k_valid = []

RMSE_k_test = []
R2_k_test = []

for seed in seed_list:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ARMA().to(device)

    wandb.watch(model)

    trainer = Trainer(model.train(), std, mean)
    Tester = tester(model.eval(), std, mean)

    for epoch in range(1, (iteration + 1)):
        if epoch  % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        start = timeit.default_timer()
        train_loss = trainer.train(train_loader)
        valid_loss, RMSE_valid = Tester.test(valid_loader)
        test_loss, RMSE_test = Tester.test(test_loader)
        end = timeit.default_timer()
        time = end - start

        print(
            'ARMA-epoch:%d,---train loss: %.3f,valid loss: %.3f,test loss: %.3f, valid rmse: %.3f, test rmse: %.3f, time: %.3f' %
            (epoch, train_loss, valid_loss, test_loss, RMSE_valid, RMSE_test, time))
        
        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "RMSE_valid": RMSE_valid,
            "RMSE_test": RMSE_test})

        if epoch == iteration:
            RMSE_k_valid.append(RMSE_valid)
            RMSE_k_test.append(RMSE_test)

            # checkpoint = 'model_'+project+'_'+str(epoch)+'.pt'
            # torch.save(model, os.path.join(wandb.run.dir, checkpoint))  

    print('RMSE_k_valid', RMSE_k_valid)
    print('RMSE_k_test', RMSE_k_test)

RMSE_mean_valid, RMSE_std_valid = metric(RMSE_k_valid)
RMSE_mean_test, RMSE_std_test = metric(RMSE_k_test)

print('result:, valid_RMSE:%.3f, valid_RMSE_std:%.3f' % (RMSE_mean_valid, RMSE_std_valid))
print('result:, test_RMSE:%.3f, test_RMSE_std:%.3f' % (RMSE_mean_test, RMSE_std_test))

