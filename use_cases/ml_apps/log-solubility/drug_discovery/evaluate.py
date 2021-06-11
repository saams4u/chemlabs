# evaluate.py - predict (infer) inputs (single/batch).
import copy
from train import *

# best_model = torch.load('saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(best_param["test_epoch"])+'.pt')     
best_model = pickle.load(open('saved_models/solubility_model_'+prefix_filename+'_'+start_time+'_'+str(best_param["test_epoch"])+'.pkl', 'rb'))
best_model_dict = best_model.state_dict() 
best_model_wts = copy.deepcopy(best_model_dict)

model.load_state_dict(best_model_wts)
(best_model.align[0].weight == model.align[0].weight).all()

test_MAE, test_MSE = eval(model, test_df)
print("best epoch:",best_param["test_epoch"],"\n","test MSE:",test_MSE)

