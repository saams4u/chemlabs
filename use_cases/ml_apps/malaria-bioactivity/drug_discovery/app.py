import os
import json

from fastapi import FastAPI, Path
from fastapi.responses import RedirectResponse

from http import HTTPStatus
from pydantic import BaseModel

from drug_discovery import config, utils

app = FastAPI(
    title="drug_discovery",
    description="Malaria Bioactivity: EC50 Analysis for Inhibiting Malaria",
    version="1.0.0",
)

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

class PredictPayload(BaseModel):
    pass

@utils.construct_response
@app.post("/predict")
async def _predict(payload: PredictPayload):
    pass
