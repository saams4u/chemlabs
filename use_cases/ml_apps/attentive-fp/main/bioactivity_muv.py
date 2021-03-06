import time, gc, copy, os, pickle, sys

sys.setrecursionlimit(50000)

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

# from tensorboardX import SummaryWriter

torch.manual_seed(8)
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, matthews_corrcoef, recall_score, accuracy_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, precision_recall_curve
from sklearn.metrics import auc, f1_score

from rdkit import Chem
from rdkit.Chem import AllChem, QED, rdMolDescriptors, MolSurf, rdDepictor
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

from numpy.polynomial.polynomial import polyfit

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns; sns.set()
import sascorer

import network.AttentiveFP.AttentiveLayers, network.AttentiveFP.AttentiveLayers_viz
import network.AttentiveFP.Featurizer, network.AttentiveFP.Featurizer_aromaticity_rm 
import network.AttentiveFP.getFeatures, network.AttentiveFP.getFeatures_aromaticity_rm

from network.AttentiveFP.getFeatures import save_smiles_dicts, get_smiles_array
from network.AttentiveFP.AttentiveLayers import Fingerprint

import config, wandb

wandb.init(project="bioactivity-muv")
wandb.log({"run_dir": wandb.run.dir})

task_name = 'muv'
tasks = [
    "MUV-466","MUV-548","MUV-600","MUV-644","MUV-652","MUV-689","MUV-692","MUV-712","MUV-713","MUV-733","MUV-737","MUV-810","MUV-832","MUV-846","MUV-852","MUV-858","MUV-859"
]
raw_filename = "dataset/muv.csv"

feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')

smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values
print("number of all smiles: ",len(smilesList))

atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []

for smiles in smilesList:
    try:        
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        print("not successfully processed smiles: ", smiles)
        pass

print("number of successfully processed smiles: ", len(remained_smiles))
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]

print(smiles_tasks_df)
smiles_tasks_df['cano_smiles'] =canonical_smiles_list
assert canonical_smiles_list[8]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]), isomericSmiles=True)

plt.figure(figsize=(5, 3))
sns.set(font_scale=1.5)
ax = sns.distplot(atom_num_dist, bins=28, kde=False)

plt.tight_layout()
plt.savefig("atom_num_dist_"+prefix_filename+".png",dpi=200)

print(len([i for i in atom_num_dist if i<51]),len([i for i in atom_num_dist if i>50]))

random_seed = 68
start_time = str(time.ctime()).replace(':','-').replace(' ','_')
start = time.time()

batch_size = 100
epochs = 800
p_dropout = 0.2
fingerprint_dim = 250

radius = 3
T = 2
weight_decay = 3.5 # also known as l2_regularization_lambda
learning_rate = 3.7

per_task_output_units_num = 2 # for classification model with 2 classes
output_units_num = len(tasks) * per_task_output_units_num

smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())<151]
# uncovered = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())>150]

# smiles_tasks_df = smiles_tasks_df[~smiles_tasks_df["cano_smiles"].isin(uncovered)]

if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)
# feature_dicts = get_smiles_dicts(smilesList)

remained_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)

weights = []

for i,task in enumerate(tasks):    
    negative_df = remained_df[remained_df[task] == 0][["smiles",task]]
    positive_df = remained_df[remained_df[task] == 1][["smiles",task]]

    negative_test = negative_df.sample(frac=1/10,random_state=random_seed)
    negative_valid = negative_df.drop(negative_test.index).sample(frac=1/9,random_state=random_seed)
    negative_train = negative_df.drop(negative_test.index).drop(negative_valid.index)
    
    positive_test = positive_df.sample(frac=1/10,random_state=random_seed)
    positive_valid = positive_df.drop(positive_test.index).sample(frac=1/9,random_state=random_seed)
    positive_train = positive_df.drop(positive_test.index).drop(positive_valid.index)
    
    weights.append([(positive_test.shape[0]+negative_test.shape[0])/negative_test.shape[0],\
                    (positive_test.shape[0]+negative_test.shape[0])/positive_test.shape[0]])
    
    train_df_new = pd.concat([negative_train,positive_train])
    valid_df_new = pd.concat([negative_valid,positive_valid])
    test_df_new = pd.concat([negative_test,positive_test])

    if i==0:
        train_df = train_df_new
        test_df = test_df_new
        valid_df = valid_df_new
    else:
        train_df = pd.merge(train_df, train_df_new, on='smiles', how='outer') 
        test_df = pd.merge(test_df, test_df_new, on='smiles', how='outer')
        valid_df = pd.merge(valid_df, valid_df_new, on='smiles', how='outer')

x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[0]],feature_dicts)

num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]

loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight),reduction='mean') for weight in weights]
model = Fingerprint(radius, T, num_atom_features,num_bond_features,
            fingerprint_dim, output_units_num, p_dropout)
model.cuda()

# tensorboard = SummaryWriter(log_dir="runs/"+start_time+"_"+prefix_filename+"_"+str(fingerprint_dim)+"_"+str(p_dropout))
# optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

wandb.watch(model)
# config.logger.info(
#         "Model:\n"
#         f"  {model.named_parameters}")

optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())

params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)

def train(model, dataset, optimizer, loss_function):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])

    #shuffle them
    np.random.shuffle(valList)
    batch_list = []

    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   

    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df.smiles.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
#         print(torch.Tensor(x_atom).size(),torch.Tensor(x_bonds).size(),torch.cuda.LongTensor(x_atom_index).size(),torch.cuda.LongTensor(x_bond_index).size(),torch.Tensor(x_mask).size())
        
        model.zero_grad()

        # Step 4. Compute your loss function. (Again, Torch wants the target wrapped in a variable)
        loss = 0.0
        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
#             validInds = np.where(y_val != -1)[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss += loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
            
        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()
        
def eval(model, dataset):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []

    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch) 

    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch,:]
        smiles_list = batch_df.smiles.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        atom_pred = atoms_prediction.data[:,:,1].unsqueeze(2).cpu().numpy()

        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
#             validInds = np.where((y_val=='0') | (y_val=='1'))[0]
#             print(validInds)
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
#             print(validInds)
            loss = loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
#             print(y_pred_adjust)
            y_pred_adjust = F.softmax(y_pred_adjust,dim=-1).data.cpu().numpy()[:,1]
            losses_list.append(loss.cpu().detach().numpy())
            try:
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_val_list[i] = []
                y_pred_list[i] = []
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
#             print(y_val,y_pred,validInds,y_val_adjust,y_pred_adjust)            
    eval_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(tasks))]
    eval_prc = [auc(precision_recall_curve(y_val_list[i], y_pred_list[i])[1],precision_recall_curve(y_val_list[i], y_pred_list[i])[0]) for i in range(len(tasks))]
#     eval_precision = [precision_score(y_val_list[i],
#                                      (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
#     eval_recall = [recall_score(y_val_list[i],
#                                (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    eval_loss = np.array(losses_list).mean()
    
    return eval_roc, eval_prc, eval_loss # eval_precision, eval_recall, 

best_param ={}
best_param["roc_epoch"] = 0
best_param["loss_epoch"] = 0
best_param["valid_roc"] = 0
best_param["valid_loss"] = 9e8

# config.logger.info("Training:")

for epoch in range(epochs):    
    train_roc, train_prc, train_loss = eval(model, train_df)
    valid_roc, valid_prc, valid_loss = eval(model, valid_df)

    train_roc_mean = np.array(train_roc).mean()
    valid_roc_mean = np.array(valid_roc).mean()
    
#     tensorboard.add_scalars('ROC',{'train_roc':train_roc_mean,'valid_roc':valid_roc_mean},epoch)
#     tensorboard.add_scalars('Losses',{'train_losses':train_loss,'valid_losses':valid_loss},epoch)

    if valid_roc_mean > best_param["valid_roc"]:
        best_param["roc_epoch"] = epoch
        best_param["valid_roc"] = valid_roc_mean
        if valid_roc_mean > 0.75:
            saved_model = 'model_'+prefix_filename+'_'+start_time+'_'+str(epoch)+'.pt'
            torch.save(model, os.path.join(wandb.run.dir, saved_model))             
    
    if valid_loss < best_param["valid_loss"]:
        best_param["loss_epoch"] = epoch
        best_param["valid_loss"] = valid_loss

    print("EPOCH:\t"+str(epoch)+'\n'\
        +"train_loss"+":"+str(train_loss)+'\n'\
        +"val_loss"+":"+str(valid_loss)+'\n'\
        +"train_roc"+":"+str(train_roc)+'\n'\
        +"valid_roc"+":"+str(valid_roc)+'\n'\
        +"train_prc"+":"+str(train_roc)+'\n'\
        +"valid_prc"+":"+str(valid_prc)+'\n'\
        )

    # config.logger.info(
    #     f"Epoch: {epoch+1} | "
    #     f"train_loss: {train_loss:.2f}, train_roc: {train_roc:.2f}, train_roc_mean: {train_roc_mean:.2f}, train_prc_mean: {train_prc_mean:.2f}, "
    #     f"val_loss: {valid_loss:.2f}, val_roc: {valid_roc:.2f}, valid_roc_mean: {valid_roc_mean:.2f}, valid_prc_mean: {valid_prc_mean:.2f}")
    
    wandb.log({
        "train_loss": train_loss,
        "val_loss": valid_loss,
        "train_roc": train_roc,
        "valid_roc": valid_roc,
        "train_prc": train_prc,
        "valid_prc": valid_prc})
    
    if (epoch - best_param["roc_epoch"] >6) and (epoch - best_param["loss_epoch"] >8):        
        break
        
    train(model, train_df, optimizer, loss_function)

# evaluate model
checkpoint = 'model_'+prefix_filename+'_'+start_time+'_'+str(best_param["roc_epoch"])+'.pt'
best_model = torch.load(os.path.join(wandb.run.dir, checkpoint))     

# best_model_dict = best_model.state_dict()
# best_model_wts = copy.deepcopy(best_model_dict)

# model.load_state_dict(best_model_wts)
# (best_model.align[0].weight == model.align[0].weight).all()

test_roc, test_prc, test_loss = eval(best_model, test_df)

print("best epoch:"+str(best_param["roc_epoch"])
      +"\n"+"test_loss:"+str(test_loss)
      +"\n"+"test_roc:"+str(test_roc)
      +"\n"+"test_prc:"+str(test_prc)
     )
    
# config.logger.info(
#     "Test performance:\n"
#     f"  test_loss: {test_loss:.2f}, test_roc: {test_roc:.2f}, test_prc: {test_prc:.2f}")