"""
SageMaker inference script for the directed sentiment analysis model.
This script loads the model, handles input requests, performs predictions, and 
formats the output.
"""

import json
import torch
import mlflow.pytorch
from packages.lcf_pl_model import LCFS_BERT_PL

BERT_MODEL = 'bert-base-uncased'  # Or your BERT model
MAX_SEQ_LENGTH = 48
SRD = 9
LCF = 'cdw'

def model_fn(model_dir):
    """
    Loads the model from the saved MLflow model artifacts.

    Args:
        model_dir (str): The directory where the model artifacts are stored.

    Returns:
        LCFS_BERT_PL: The loaded model in evaluation mode.
    """
    model = mlflow.pytorch.load_model(f"{model_dir}/model")
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """
    Deserializes the input JSON data from the request body.

    Args:
        request_body (bytes): The raw request body.
        request_content_type (str): The content type of the request.

    Returns:
        dict: A dictionary containing the input tensors.

    Raises:
        ValueError: If the content type is not supported.
    """
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        for k, v in input_data.items():
            input_data[k] = torch.tensor(v)  # Convert to tensors
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Performs prediction using the loaded model.

    Args:
        input_data (dict): The input data as a dictionary of tensors.
        model (LCFS_BERT_PL): The loaded model.

    Returns:
        list: The prediction output as a list.
    """
    with torch.no_grad():
        predictions = model(input_data)
    return predictions.tolist()  # Convert to list for JSON serialization

def output_fn(prediction_output, accept):
    """
    Serializes the prediction output to JSON.

    Args:
        prediction_output (list): The prediction output.
        accept (str): The accepted content type.

    Returns:
        str: The JSON-serialized prediction output.

    Raises:
        ValueError: If the accept type is not supported.
    """
    if accept == "application/json":
        return json.dumps(prediction_output)
    raise ValueError(f"Unsupported accept type: {accept}")