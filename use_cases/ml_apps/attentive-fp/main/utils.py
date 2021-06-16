# utils.py - utility functions to aid app operations.
import os

from datetime import datetime
from functools import wraps
from http import HTTPStatus

import csv
import json
import utils
import torch

from network.AttentiveFP.AttentiveLayers import Fingerprint

from photovoltaic_efficiency import radius, T, num_atom_features, num_bond_features
from photovoltaic_efficiency import fingerprint_dim, output_units_num, p_dropout, best_params, checkpoint
from photovoltaic_efficiency import raw_filename, feature_filename, filename, smilesList, prefix_filename, start_time
from photovoltaic_efficiency import smiles_tasks_df, feature_dicts, remained_smiles, remained_df, test_df, random_seed

def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r") as fp:
        json_obj = json.load(fp)
    return json_obj

def save_dict(d, filepath):
    """Save dict to a json file."""
    with open(filepath, 'w') as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)

def construct_response(f):
    """Construct a JSON response for an endpoint's results."""
    @wraps(f)
    def wrap(*args, **kwargs):
        results = f(*args, **kwargs)

        # Construct response
        response = {
            'message': results['message'],
            'method': request.method,
            'status-code': results['status-code'],
            'timestamp': datetime.now().isoformat(),
            'url': request.url,
        }

        # Add data
        if results['status-code'] == HTTPStatus.OK:
            response['data'] = results['data']

        return response
    return wrap

def get_run_components(run_dir):
    model = Fingerprint(radius, T, num_atom_features, num_bond_features,
            fingerprint_dim, output_units_num, p_dropout)
    model.load_state_dict(torch.load(os.path.join(run_dir, checkpoint)))

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    dataset = test_df

    return model, dataset

def get_best_run(project, metric, objective):
    # Get all runs
    api = wandb.Api()
    runs = api.runs(project)

    # Define objective
    if objective == 'maximize':
        best_metric_value = np.NINF
    elif objective == 'minimize':
        best_metric_value = np.inf

    # Get best run based on metric
    best_run = None
    for run in runs:
        if run.state == "finished":
            metric_value = run.summary[metric]
            if objective == 'maximize':
                if metric_value > best_metric_value:
                    best_run = run
                    best_metric_value = metric_value
            else:
                if metric_value < best_metric_value:
                    best_run = run
                    best_metric_value = metric_value

def load_run(run):
    run_dir = os.path.join(
        config.BASE_DIR, '/'.join(run.summary['run_dir'].split('/')[-2:]))

    # Create run dir if it doesn't exist
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    else:
        return run_dir

    # Load run files (if it exists, nothing happens)
    for file in run.files():
        file.download(replace=False, root=run_dir)

    return run_dir