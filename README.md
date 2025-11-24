🌬️ Wind Speed Forecasting with Multivariate LSTM & SHAP Explainability
📌 Overview

This project builds a multivariate time series forecasting model for predicting future wind speed using an LSTM deep learning architecture.
The model uses 24 hours of past meteorological data to forecast the next 7 hours.

To ensure transparency and interpretability, the project incorporates SHAP explainability, which highlights how each feature impacts model predictions.

The project includes:
✔ Complete preprocessing
✔ LSTM multivariate model
✔ Forecasting
✔ Evaluation
✔ SHAP analysis
✔ Automated PDF and text reports
📂 Dataset

The dataset contains multiple weather-related features such as:

WIND (Target Variable)

RAIN

T.MAX

T.MIN

T.MIN.G

IND.1

IND.2

DATE

The DATE column is transformed into a datetime index for time-series modeling.

⚙️ Project Pipeline
1️⃣ Data Preprocessing

Handles missing values using mean imputation

Converts DATE → datetime

Sorts by time

Removes extreme values using clipping

Normalizes all features with MinMaxScaler

2️⃣ Sequence Generation

A sliding-window method is used:

Past Input Window: n_past = 24 hours

Forecast Horizon: n_future = 7 hours

Features: 8 meteorological variables

3️⃣ LSTM Model Architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(n_past, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(n_future))


Training Enhancements

EarlyStopping

ModelCheckpoint

Final model saved as .h5

4️⃣ Evaluation Metrics

The following metrics are calculated per forecast horizon:

MAE

RMSE

MAPE

Results are saved as:

metrics_per_horizon.csv

5️⃣ Visualization

The project generates and saves:

Training vs Validation loss curve

Predicted vs Actual plots (for first 3 horizons)

SHAP feature importance visualizations

Stored in:

/output_models/

6️⃣ SHAP Explainability

The project uses:

SHAP DeepExplainer (primary)

Automatic fallback to KernelExplainer

Outputs include:

SHAP values per feature

Feature importance ranking

CSV and PNG visualizations

7️⃣ Automated Report Generation

A detailed report is created in two formats:

📄 report_summary.txt
📄 report.pdf — includes plots, metrics, SHAP charts

🧪 Technologies Used

Python

TensorFlow / Keras

SHAP

NumPy

Pandas

Matplotlib

Seaborn

Statsmodels

FPDF

📁 Project Output Structure
output_models/
│── best_lstm_model.h5
│── final_lstm_model.h5
│── training_validation_loss.png
│── pred_vs_actual_h1.png
│── pred_vs_actual_h2.png
│── pred_vs_actual_h3.png
│── shap_feature_importance.png
│── shap_feature_importance_kernel.png
│── shap_feature_importance.csv
│── shap_feature_importance_kernel.csv
│── metrics_per_horizon.csv
│── shap_values_kernel.csv
│── report_summary.txt
│── report.pdf

⭐ Key Highlights

✔ Complete multivariate forecasting pipeline

✔ Future wind prediction (7-step ahead)

✔ Accurate LSTM architecture

✔ Full SHAP Explainability

✔ Automated PDF reporting

✔ Ready for deployment & research usage

📧 Contact

For improvements, deployment help, or project extension — feel free to ask!
