**Wind Speed Forcasting Using Multivariate LSTM & SHAP Explainability**
 ## Project Summary
 This repository contains a complete, end to end pipeline for multivariate wind speed forecasting using a two-layer LSTM and SHAP explainability. The model uses the previous **24 hours** of meteorological data to forecast the next **7 hours** of wind speed. Output includes trained models, evaluation metrics by horizon,plots, SHAP feature importance, and an automated PDF report.

## highlights
**Input window:** 24 hours (n_past=24)
**Forecast horizon:** 7 hours (n_future=7)
**Model:** 2-layer LSTM(64-32),Dropout(0.2), Dense Output(7)
**Explainability:** SHAP (DeepExplainer preferred, KernelExplainer fallback)
**Outputs:** model `.h5`, per-horizon metrics CSV, SHAP values & plots, a PDF report

