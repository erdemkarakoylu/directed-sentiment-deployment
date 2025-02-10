"""
SageMaker training script for the directed sentiment analysis model.
This script loads the data, trains the model, logs the model with MLflow, and saves 
the model artifacts.
"""
import argparse
import os
import torch
import mlflow
import mlflow.pytorch
from packages.lcf_pl_model import LCFS_BERT_PL
from packages.lcf_textore_datamodule import DataModule
import pytorch_lightning as pl
import configparser
import boto3
import json


# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")
aws_region = config["aws"]["region"]  
bucket_name = config["s3"]["bucket_name"]
data_key = config["s3"]["data_key"]

BERT_MODEL = config['model']['bert_model']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", type=int, default=48)
    parser.add_argument("--srd", type=int, default=9)
    parser.add_argument("--lcf", type=str, default="cdw")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    # Load data from S3
    s3 = boto3.client('s3', region_name=aws_region)
    obj = s3.get_object(Bucket=bucket_name, Key=data_key)
    # Assuming your data is in JSON format (adapt as needed)
    train_data = json.loads(obj['Body'].read())

    # For local demo (replace with S3 loading if running in SageMaker):
    # with open("data/sample_data.json", "r") as f:
    #     train_data = json.load(f)

    dm = DataModule(
        model_name=BERT_MODEL,
        batch_size=args.batch_size,
        num_workers=2,
        data_dir=train_data,  # Pass the loaded data
        max_seq_length=args.max_seq_length
    )
    dm.setup()  # Make sure to call setup()

    model = LCFS_BERT_PL(
        BERT_MODEL,
        max_seq_length=args.max_seq_length,
        synthactic_distance_dependency=args.srd,
        local_context_focus=args.lcf
    )

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        trainer = pl.Trainer(max_epochs=3)  # Adjust as needed
        trainer.fit(model, datamodule=dm)

        mlflow.pytorch.log_model(model, "model")

        model_dir = os.environ["SM_MODEL_DIR"]
        torch.save(model.state_dict(), os.path.join(model_dir, "my_model.pth"))