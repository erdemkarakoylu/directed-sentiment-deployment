# Directed Sentiment Analysis Deployment

This repository provides a complete setup for deploying a directed sentiment analysis model using AWS SageMaker, FastAPI, and Docker. The model predicts the sentiment expressed towards a specific target within a given text.

## Features

* **SageMaker Integration:**  The model is trained and deployed using Amazon SageMaker, showcasing efficient model training and deployment in a cloud environment.
* **FastAPI API:** A FastAPI application provides a RESTful API for interacting with the deployed model, allowing users to send text and target information and receive sentiment predictions.
* **Dockerized Application:** The FastAPI application is containerized using Docker, ensuring portability and consistency across different environments.
* **MLflow Tracking:** MLflow is used to track experiments, log model parameters and metrics, and manage model versions, demonstrating best practices for MLOps.
* **Configuration Management:** Sensitive information (AWS credentials, endpoint names, etc.) is stored in a `config.ini` file that is excluded from version control, ensuring security and maintainability.

## Project Structure
├── app.py          # FastAPI application
├── Dockerfile      # Dockerfile for FastAPI
├── inference.py    # SageMaker inference script
├── train.py        # SageMaker training script
├── requirements.txt # Dependencies for FastAPI and SageMaker scripts
├── config.ini      # Configuration file (not tracked by Git)
└── data/           # (Optional) Small sample data for demo purposes
└── sample_data.json  # Example data


## Prerequisites

* **AWS Account:** An active AWS account with necessary permissions to create and manage SageMaker resources, ECR repositories, and ECS clusters.
* **Docker:** Docker installed on your local machine for building and testing the Docker image.
* **Python 3.9:** Python 3.9 or higher installed on your local machine.
* **Virtual Environment:** A virtual environment is recommended to manage project dependencies.

## Deployment Steps

1. **Clone the repository:** `git clone https://github.com/your-username/directed-sentiment-deployment.git`
2. **Create and activate a virtual environment:**
   ```bash
   python3.9 -m venv.venv
   source.venv/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. Configure AWS credentials: Configure your AWS credentials using the AWS CLI or environment variables.
5. Create an ECR repository: Create an Amazon ECR repository to store your Docker image.
6. Build and push the Docker image:


```bash
docker build -t directed-sentiment-api.
docker tag directed-sentiment-api:latest <your-account-id>.dkr.ecr.<your-region>[.amazonaws.com/directed-sentiment-api:latest](https://www.google.com/search?q=https://.amazonaws.com/directed-sentiment-api:latest)
docker push <your-account-id>.dkr.ecr.<your-region>[.amazonaws.com/directed-sentiment-api:latest](https://www.google.com/search?q=https://.amazonaws.com/directed-sentiment-api:latest)
```
7. Create a SageMaker training job: Use the AWS console or the SageMaker SDK to create a training job using the train.py script.
8. Deploy the model to a SageMaker endpoint: After training, deploy the model to a SageMaker endpoint using the inference.py script.
9. Create an ECS cluster: Create an Amazon ECS cluster with Fargate to run your Dockerized FastAPI application.
10. Create an ECS task definition: Define a task definition that uses your Docker image and exposes the necessary ports.
11. Create an ECS service: Create an ECS service that runs your task definition and configure an Application Load Balancer (ALB) to route traffic to your application.
12. Test the deployment: Send requests to your ALB's DNS name to test the deployed API.

## Usage
Once deployed, you can send POST requests to the /predict endpoint of your API with the following JSON payload:


```json
{
  "text": "The battery life of this phone is amazing, but the camera is disappointing.",
  "target": "battery life"
}
```
The API will return a JSON response containing the predicted sentiment.

## Cleanup

To avoid incurring charges, remember to delete the SageMaker endpoint, ECS service, ECR repository, and any other AWS resources you created during the deployment.