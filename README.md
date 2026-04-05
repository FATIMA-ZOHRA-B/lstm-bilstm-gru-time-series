# lstm-bilstm-gru-time-series
Time series forecasting using LSTM, BiLSTM, and GRU (TensorFlow)
This project compares LSTM, BiLSTM, and GRU models for time series prediction.

## Models
- LSTM
- BiLSTM
- GRU

## Techniques
- MinMax scaling
- Sliding window (look_back=12)
- EarlyStopping
- Metrics: RMSE, MAE, MAPE, R²

## Run

```bash
pip install -r requirements.txt
python src/train.py
## Medium Article
https://medium.com/data-science-data-engineering/time-series-prediction-lstm-bi-lstm-gru-99334fc16d75
