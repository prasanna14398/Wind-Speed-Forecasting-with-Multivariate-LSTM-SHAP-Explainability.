 Wind Speed Forecasting Using a Multivariate LSTM Model with SHAP Explanations
 Overview

This project focuses on predicting short-term wind speed using a multivariate LSTM model.
I used 24 hours of past weather data to forecast the next 7 hours.
Along with the forecasting model, I included SHAP analysis so that the behaviour of the model can be interpreted clearly.
All the results—plots, metrics, and reports—are generated automatically.

Dataset Information

The dataset includes the following variables:

WIND (target)

RAIN

T.MAX

T.MIN

T.MIN.G

IND.1

IND.2

DATE

The DATE column is converted into a proper datetime index so that the model can use it as a time series.
Steps Followed
1. Data Preprocessing

* Filled missing values

* Converted DATE into datetime

* Sorted the data chronologically

* Scaled all features using MinMaxScaler

* Created sliding window sequences for LSTM input
  
 2. LSTM Model

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(n_past, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(n_future))
I used EarlyStopping and ModelCheckpoint during training.

 Forecasting Metrics
Horizon	MAE	RMSE	MAPE (%)
1	3.5807	4.5193	61.19
2	3.6259	4.5966	61.75
3	3.6795	4.6866	60.29
4	3.7006	4.7368	59.03
5	3.6876	4.6926	60.88
6	3.6995	4.6932	62.48
7	3.7301	4.7522	60.35

Visual Outputs

The project generates:

Training and validation loss curves

Actual vs predicted plots

SHAP feature importance graphs

SHAP value tables

All results are saved under: /output_models

SHAP Analysis
Understanding why the LSTM model predicts certain wind-speed values is just as important as getting a good score, especially for weather-related work where decisions depend on trustworthy information. To interpret the behaviour of my model, I used SHAP values. Instead of treating the model like a black box, SHAP helps break down each prediction and shows how every feature pushes the forecast up or down. Since LSTMs are complicated and depend on time, this method gave me a clear picture of what the model is relying on.

When I examined the SHAP plots, one pattern stood out immediately: past wind values had the strongest influence on future predictions. This wasn’t surprising because wind usually follows short-term trends—if it has been strong for the last few hours, it tends to remain strong unless a sudden weather shift happens. In the SHAP charts, these past WIND values consistently showed the largest positive and negative contributions, meaning the model was paying close attention to them.

The next group of features that showed meaningful impact were the temperature-related variables: T.MAX, T.MIN, and T.MIN.G. Their SHAP contributions were not as strong as wind itself, but they affected the forecasts in noticeable ways. Higher temperatures generally pushed the prediction slightly upward. This makes sense from what we know about weather, because warm air tends to rise and cause small pressure differences, which can lead to increased wind movement. Lower temperatures tended to reduce the predicted wind value. Seeing this behaviour reflected in the SHAP plots confirmed that the model had picked up realistic weather relationships instead of random patterns.

Two other features, IND.1 and IND.2, had a smaller but still consistent influence. They didn’t dominate the predictions, but they nudged the model in subtle ways depending on their values. Since these features probably represent local atmospheric indicators, their smaller SHAP values seemed reasonable.

One thing I noticed was that rainfall (RAIN) had very little effect on the predictions. The SHAP values for RAIN stayed close to zero most of the time. After seeing this, I revisited the dataset and realized that in the given region, rainfall didn’t always correlate strongly with wind. The model learned this correctly and didn’t rely on rainfall as much as the other variables.

Overall, the SHAP analysis helped confirm that the LSTM wasn’t behaving randomly—it was learning meaningful, weather-related patterns. The features that should matter (like past WIND and temperature) were the ones influencing the predictions the most. Features that shouldn’t matter as much (like rainfall) barely contributed, which shows that the model wasn’t overfitting noise. Doing this analysis made me more confident about using the model for forecasting because I can explain, in simple terms, what factors are driving its decisions. This level of interpretability is helpful if someone asks why the model predicted a certain wind speed for a specific hour.

Reports Included
report_summary.txt
report.pdf

These include:

✔ Model details
✔ Metrics
✔ Plots
✔ SHAP explanations

Folder Structure
project/
│── wind_forecasting.ipynb
│── README.md
│── report_summary.txt
│── report.pdf
│── output_models/
│     ├── best_lstm_model.h5
│     ├── final_lstm_model.h5
│     ├── training_validation_loss.png
│     ├── shap_feature_importance.png
│     └── metrics_per_horizon.csv

Contact

For doubts, improvements, or deployment assistance, feel free to reach out.
