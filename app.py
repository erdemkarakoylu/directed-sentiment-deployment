"""
FastAPI application for serving predictions from a directed sentiment analysis model 
deployed on SageMaker. This app receives text and target span, preprocesses it, sends it
to the SageMaker endpoint, and returns the prediction.
"""
import boto3
import json
import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from packages.lcf_textore_datamodule import DataModule
import configparser  # Import configparser


app = FastAPI()

class PredictionRequest(BaseModel):
    text: str
    target: str

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")
aws_region = config["aws"]["region"]
sagemaker_endpoint_name = config["sagemaker"]["endpoint_name"]

sagemaker_client = boto3.client("sagemaker-runtime", region_name=aws_region)
endpoint_name = sagemaker_endpoint_name

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predicts directed sentiment for a given text and target span using the 
    SageMaker endpoint.

    Args:
        request (PredictionRequest): The input request containing the text and target.

    Returns:
        dict: The prediction result from the SageMaker endpoint.

    Raises:
        HTTPException: If there's an error during prediction.
    """
    try:
        processed_input = preprocess_input(request.text, request.target)

        payload = json.dumps(
            {k: v.tolist() for k, v in processed_input.items()}
        ).encode("utf-8")

        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        return result

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


def preprocess_input(text, target):
    """
    Preprocesses the input text and target span for the model.

    Args:
        text (str): The input text.
        target (str): The target span.

    Returns:
        dict: A dictionary containing the preprocessed input tensors.
    """
    datamodule = load_datamodule()
    processed_input = datamodule.process_new_example(text, target)
    processed_input = {k: (
        v.clone().unsqueeze(0).to("cpu") if 'bert' in k else v)
        for k, v in processed_input.items()
        }
    processed_input['dep_distance_to_target'] = processed_input[
        'dep_distance_to_target'].clone().unsqueeze(0).to("cpu")
    return processed_input

def load_datamodule():
    """
    Loads and initializes the DataModule.

    Returns:
        DataModule: The initialized DataModule object.
    """
    dm = DataModule(
        model_name=BERT_MODEL, batch_size=TRAIN_BATCH_SIZE, num_workers=2,
        data_dir=DATA_PATH, max_seq_length=MAX_SEQ_LENGTH)
    return dm

BERT_MODEL = 'bert-base-uncased'  # Or your BERT model
DATA_PATH = "data/textore/ready/eval_samples_added_2_training"  # Or your data path
TRAIN_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 48
SRD = 9
LCF = 'cdw'
