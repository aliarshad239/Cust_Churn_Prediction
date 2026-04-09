# CODEX Plan — Collab 1 (Churn Prediction) — Up to pre-AWS

## Goal
Build an end-to-end churn prediction pipeline (data → features → train → evaluate → predict) using config/params.yaml.

## Must Produce
- Processed dataset saved to data/processed/
- Trained model artifact saved to models/
- Metrics saved to reports/metrics.json
- Figures saved to reports/figures/
- Predict script that loads model and predicts for a single customer
- README with run instructions and results

## Rules
- Use params.yaml for columns/features/paths
- Keep simple sklearn models; no deep learning
- No venv/, .git/, __MACOSX/ in outputs
- Scripts runnable from repo root

## Execution Order
1) Implement src/features.py
2) Implement src/pipeline.py
3) Implement scripts/train.py
4) Implement scripts/predict.py
5) Update README.md