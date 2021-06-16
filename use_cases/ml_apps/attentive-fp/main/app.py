import os
import json

from fastapi import FastAPI, Path
from fastapi.responses import RedirectResponse

import nest_asyncio
from pyngrok import ngrok
import uvicorn

from http import HTTPStatus
from pydantic import BaseModel

import config, utils

from photovoltaic_efficiency import eval, get_run_components

app = FastAPI(
    title="AttentiveFP",
    description="Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism",
    version="1.0.0",
)

project_list = ["bioactivity-muv", "malaria-bioactivity", "bioactivity-bace", 
             "photovoltaic-efficiency", "bioactivity-hiv", "log-solubility"]

test_metrics = ["test_loss", "test_MSE", "test_MAE", "test_roc", "test_prc"]

# Get best run
best_run = utils.get_best_run(project=f"mahjouri-saamahn/{project_list[3]}",
                              metric=test_metrics[1], objective="minimize")

# Load best run (if needed)
best_run_dir = utils.load_run(run=best_run)

# Get run components for prediction
model, dataset = get_run_components(run_dir=best_run_dir)

@utils.construct_response
@app.get("/")
async def _index():
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response

@app.get("/experiments")
async def _experiments():
    return RedirectResponse(f"https://wandb.ai/mahjouri-saamahn/{project_list[3]}")

class PredictPayload(BaseModel):
    pass

@utils.construct_response
@app.post("/predict")
async def _predict(payload: PredictPayload):
    atoms_prediction, mol_prediction, _, _ = eval(model=model, dataset=dataset)
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {"atoms prediction": atoms_prediction, "mols prediction": mol_prediction}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
