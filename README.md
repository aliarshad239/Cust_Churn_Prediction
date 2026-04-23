# Churn Collab

End-to-end customer churn ML system with two connected layers:

1. Local model development and training
2. Model serving and deployment through a FastAPI API

The training workflow keeps the full project dependency set in `requirements.txt`. The serving layer uses a separate `requirements-api.txt` so the Docker image stays focused on inference-only runtime needs.

## System Overview

### 1. Training Layer

This layer handles:

- raw dataset loading from `data/raw/`
- feature engineering and dataset preparation
- sklearn pipeline training
- evaluation and report generation
- model artifact export to `models/`

Key files:

- `config/params.yaml`: data, feature, and training configuration
- `src/features.py`: feature engineering and inference-time feature alignment
- `src/pipeline.py`: preprocessing and model pipeline construction
- `scripts/train.py`: training, evaluation, and artifact persistence
- `scripts/predict.py`: local batch/single-record prediction using saved artifacts

### 2. Serving Layer

This layer loads the saved model artifact and exposes inference over HTTP.

- `app/main.py`: FastAPI application with `/health` and `/predict`
- `app/schemas.py`: request schema
- `Dockerfile`: container image for the inference service
- `requirements-api.txt`: minimal dependencies required for serving only
- `deploy/`: AWS ECS/Fargate deployment definitions

## Project Structure

- `data/raw/`: raw dataset input
- `data/processed/`: processed training dataset output
- `models/`: saved model artifacts
- `reports/`: metrics and evaluation figures
- `examples/`: sample API payloads
- `scripts/`: training, prediction, and testing utilities
- `src/`: shared ML pipeline code used by both training and inference
- `app/`: FastAPI serving application
- `ui/`: Streamlit demo UI for local professor/demo usage
- `deploy/`: AWS deployment assets

## Local Setup

Create a virtual environment and install the full project stack for training and local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local ML Training

Run training locally:

```bash
python scripts/train.py --config config/params.yaml
```

Training outputs:

- `data/processed/train_ready.csv`
- `models/churn_model.joblib`
- `reports/metrics.json`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`

Training prints ROC-AUC, precision, recall, and F1 to the console and persists metrics to `reports/metrics.json`.

## Saved Model Artifacts

The primary deployment artifact is:

- `models/churn_model.joblib`

The inference service also relies on:

- `config/params.yaml`

The API uses the same feature preparation logic as training, so serving stays aligned with the trained pipeline and configuration.

## Local Prediction From Saved Model

Example JSON input:

```json
{
  "gender": "Female",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 10,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.1,
  "TotalCharges": 892.0
}
```

Run local prediction:

```bash
python scripts/predict.py --input sample.json --config config/params.yaml
```

CSV single-row input is also supported:

```bash
python scripts/predict.py --input sample.csv --config config/params.yaml
```

## FastAPI Inference Service

Run the API directly from the local Python environment:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test the running service:

```bash
python scripts/test_api.py
```

Available endpoints:

- `GET /health`
- `POST /predict`

## Streamlit Demo UI

The demo UI is a thin client over the existing FastAPI service. It collects core customer fields, sends them to `http://localhost:8000/predict`, and displays:

- churn probability
- prediction label
- threshold used

Run steps for the local demo:

1. Start the FastAPI API:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. In a second terminal, start the Streamlit UI:

```bash
streamlit run ui/app.py
```

3. Open the local Streamlit URL shown in the terminal, fill in the form, and click `Predict Churn`.

## Dockerized Deployment Layer

The container build is intentionally separate from the training environment:

- `requirements.txt` remains the full project/training dependency set
- `requirements-api.txt` is used only for the inference image

Build and run the inference container:

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

This keeps the deployment image smaller and avoids installing training-only libraries in the serving container.

## AWS-Ready Deployment

The API can run locally from bundled artifacts or pull the model/config from S3 at startup.

Set environment variables:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export ECR_REPO=churn-api
export S3_BUCKET=churn-collab-models
```

### 1. Upload model and config to S3

```bash
aws s3 mb s3://$S3_BUCKET --region $AWS_REGION
aws s3 cp models/churn_model.joblib s3://$S3_BUCKET/models/churn_model.joblib
aws s3 cp config/params.yaml s3://$S3_BUCKET/config/params.yaml
```

### 2. Build and push the API image to ECR

```bash
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $ECR_REPO .
docker tag $ECR_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

### 3. Create CloudWatch logs and ECS cluster

```bash
aws logs create-log-group --log-group-name /ecs/churn-api --region $AWS_REGION
aws ecs create-cluster --cluster-name churn-api-cluster --region $AWS_REGION
```

### 4. Register the ECS task definition

Update placeholders in `deploy/ecs-task-def.json`:

- `<AWS_ACCOUNT_ID>`
- `<AWS_REGION>`
- `<ECR_REPO>`
- `<S3_BUCKET>`

Then register:

```bash
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-def.json --region $AWS_REGION
```

### 5. Create the ECS Fargate service

Update placeholders in `deploy/ecs-service-def.json`:

- `<SUBNET_ID_1>`
- `<SUBNET_ID_2>`
- `<SECURITY_GROUP_ID>`

Then create the service:

```bash
aws ecs create-service --cli-input-json file://deploy/ecs-service-def.json --region $AWS_REGION
```

### 6. Call the deployed endpoint

Find the running task and public IP:

```bash
aws ecs list-tasks --cluster churn-api-cluster --service-name churn-api-service --region $AWS_REGION
aws ecs describe-tasks --cluster churn-api-cluster --tasks <TASK_ARN> --region $AWS_REGION
```

Test the deployed API:

```bash
curl -X POST http://<PUBLIC_IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/example_request.json
```

## Notes

- Training logic and training dependencies remain unchanged in `requirements.txt`.
- The Docker image now installs only inference dependencies from `requirements-api.txt`.
- If you choose to serve a model artifact that depends on additional runtime libraries beyond the default pipeline, add those packages to `requirements-api.txt` before building the image.
