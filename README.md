 Wind Speed Forecasting Using a Multivariate LSTM Model with SHAP Explanations
 Overview
This project focuses on predicting short-term wind speed using a multivariate LSTM model. The goal was not only to build a forecasting model but also to understand why the model makes certain predictions. For that reason, I also included SHAP explainability to analyze feature influence, especially during periods of high variability.

The model predicts the next 7 hours of wind speed using 48 past timesteps of meteorological features such as:

Wind speed

Rain

Max/Min temperature

Two indicator variables

Derived date/time components

Although several model variations were tested, the final results were obtained using the LSTM architecture described below.

Dataset Information
Before building the model, I checked whether the dataset met the characteristics typically expected in time-series forecasting.

✔ Trend

Wind speed and temperature showed long, smooth upward and downward drifts. These weren’t constant but developed over several days, indicating the presence of a trend.

✔ Seasonality

After plotting hourly averages, a repeating pattern became visible. For example:

Higher speeds during mid-afternoon

Lower, calmer periods around early morning
This daily cycle gave evidence of natural seasonality.

✔ Noise

Short-term random fluctuations appeared throughout the data. These irregularities are typical in environmental recordings, so I kept them as part of the modeling process instead of smoothing them out.

Because the dataset contained trend + seasonality + noise, it was suitable for the forecasting task.

Data Preprocessing Steps

Converted DATE into a proper datetime format

Sorted values chronologically

Clipped extreme spikes (to reduce measurement anomalies)

Applied MinMax scaling feature-wise

Created input/output sequences using:

Past window: 48 hours

Forecast horizon: 7 hours  

Model Architecture & Hyperparameter Choices

I tested several configurations before settling on the final model.
Initially, I thought a 24-step input would work, but the predictions became unstable—almost too reactive. Extending it to 48 steps gave smoother and more reliable patterns.

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(n_outputs))

* 64 LSTM units: I tried 32, 64, and 128.
128 tended to overfit quickly; 32 couldn’t capture multi-feature interactions.
64 provided the most balanced validation curves.

* 2 LSTM layers:
I initially tested 3 layers, but the third layer didn’t improve performance and slowed down training.
With 1 layer, the model underestimated sharp wind changes.

* Dropout 0.2:
Lower values (0.1) overfit after ~20 epochs.
Higher (0.4) damaged the learning of long-range patterns.

* Learning rate 0.001:
0.0001 converged too slowly, 0.005 produced unstable oscillations.

All tuning was done manually by checking training/validation loss.

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

One interesting thing I learned while interpreting the SHAP values is that the model behaves differently depending on whether the atmosphere is calm or rapidly changing.

✔ During High-Volatility Periods

When wind speed changed sharply:

The SHAP values for wind speed, wind direction, and gust-related variables showed large spikes.

The model leaned heavily on most recent observations to predict abrupt jumps.

✔ During Stable Periods

When the wind remained relatively constant:

SHAP values were smoother and more evenly spread across features.

Instead of one feature dominating, the LSTM seemed to consider the entire 48-step historical pattern.

This contrast helped me confirm that the LSTM model was responding appropriately to both steady and rapidly shifting conditions.

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
If you want help improving the model or exploring alternative architectures, feel free to reach out.

For doubts, improvements, or deployment assistance, feel free to reach out.
