# Churn Collab Project 1

End-to-end ML pipeline for customer churn prediction (local, reproducible, config-driven).

## Project Structure

- `config/params.yaml`: configuration for data, features, training
- `src/features.py`: feature engineering + dataset preparation
- `src/pipeline.py`: sklearn preprocessing + model pipeline
- `scripts/train.py`: train/evaluate/save artifacts
- `scripts/predict.py`: load model + predict single record
- `data/raw/`: raw dataset (input)
- `data/processed/`: processed dataset (output)
- `models/`: trained model artifacts
- `reports/`: metrics + figures

## Setup

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --config config/params.yaml
```

Outputs:
- `data/processed/train_ready.csv`
- `models/churn_model.joblib`
- `reports/metrics.json`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`

## Predict (Single Record)

JSON input example:

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

Save as `sample.json` and run:

```bash
python scripts/predict.py --input sample.json --config config/params.yaml
```

CSV input (single row) is also supported:

```bash
python scripts/predict.py --input sample.csv --config config/params.yaml
```

## Results Summary

Training prints ROC-AUC, precision, recall, and F1 to console and stores them in `reports/metrics.json`.
Figures are saved to `reports/figures/`.

XGBoost is optional; Logistic Regression works out-of-the-box on macOS without extra system libraries.

## Collab Project 2: FastAPI Inference Service

This deploys the existing trained model (`models/churn_model.joblib`) with the same preprocessing pipeline as Collab 1.
No retraining or feature changes are introduced.

### Local Run (Python)

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test locally:

```bash
python scripts/test_api.py
```

### Local Run (Docker)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### AWS Deployment (S3 + ECR + ECS Fargate)

Set environment variables (replace placeholders):

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export ECR_REPO=churn-api
export S3_BUCKET=churn-collab-models
```

#### 1) Upload model + config to S3

```bash
aws s3 mb s3://$S3_BUCKET --region $AWS_REGION
aws s3 cp models/churn_model.joblib s3://$S3_BUCKET/models/churn_model.joblib
aws s3 cp config/params.yaml s3://$S3_BUCKET/config/params.yaml
```

#### 2) Build and push image to ECR

```bash
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $ECR_REPO .
docker tag $ECR_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

#### 3) Create CloudWatch log group

```bash
aws logs create-log-group --log-group-name /ecs/churn-api --region $AWS_REGION
```

#### 4) Create ECS cluster

```bash
aws ecs create-cluster --cluster-name churn-api-cluster --region $AWS_REGION
```

#### 5) Register task definition

Update placeholders in `deploy/ecs-task-def.json`:
- `<AWS_ACCOUNT_ID>`, `<AWS_REGION>`, `<ECR_REPO>`, `<S3_BUCKET>`

Then register:

```bash
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-def.json --region $AWS_REGION
```

#### 6) Create ECS service (Fargate)

Update placeholders in `deploy/ecs-service-def.json`:
- `<SUBNET_ID_1>`, `<SUBNET_ID_2>`, `<SECURITY_GROUP_ID>`

Then create the service:

```bash
aws ecs create-service --cli-input-json file://deploy/ecs-service-def.json --region $AWS_REGION
```

#### 7) Find public endpoint

Get the public IP:

```bash
aws ecs list-tasks --cluster churn-api-cluster --service-name churn-api-service --region $AWS_REGION
aws ecs describe-tasks --cluster churn-api-cluster --tasks <TASK_ARN> --region $AWS_REGION
```

Use the `publicIPv4Address` from the ENI, then test:

```bash
curl -X POST http://<PUBLIC_IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/example_request.json
```
