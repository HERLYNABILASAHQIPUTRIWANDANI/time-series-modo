# model_utils.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import acf

# --- KONFIGURASI GLOBAL ---
N_LAGS = 7  # Jumlah hari sebelumnya (lags) yang digunakan sebagai fitur
TEST_SIZE_DAYS = 90 # Jumlah hari untuk data uji
MODEL_PATH = 'model/rf_model.pkl'
Z_SCORE = 1.96 # Faktor Z-score untuk 95% Confidence Interval

# ----------------------------------------------------
# A. FUNGSI UMUM DATA PRE-PROCESSING
# ----------------------------------------------------

def load_and_clean_data(file_path):
    """Memuat dan membersihkan data NO2."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File data tidak ditemukan: {file_path}")
        
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    # Mengisi nilai hilang (missing values) dengan forward fill
    df['NO2'] = df['NO2'].fillna(method='ffill')
    return df

def create_lags(data, n_lags):
    """Membuat fitur lagged dari data deret waktu."""
    df_lags = pd.DataFrame(data['NO2'])
    for i in range(1, n_lags + 1):
        df_lags[f'NO2_Lag_{i}'] = df_lags['NO2'].shift(i)
    df_lags.dropna(inplace=True)
    return df_lags.drop('NO2', axis=1), df_lags['NO2']

def split_data(X, y, test_size_days):
    """Membagi data latih dan uji secara kronologis."""
    return X.iloc[:-test_size_days], X.iloc[-test_size_days:], \
           y.iloc[:-test_size_days], y.iloc[-test_size_days:]

def get_last_data(df, n_lags):
    """Mengambil N_LAGS data historis terakhir."""
    return df['NO2'].tail(n_lags)

# ----------------------------------------------------
# B. FUNGSI RANDOM FOREST (RF)
# ----------------------------------------------------

def train_rf(X_train, y_train):
    """Melatih Random Forest."""
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, MODEL_PATH)
    return rf_model

def predict_rf_n_days(model, last_data_series, n_days, n_lags):
    """Prediksi RF multi-step rekursif dengan Confidence Interval 95%."""
    if model is None or len(last_data_series) != n_lags:
        return pd.DataFrame()

    current_features = last_data_series.values
    predictions_data = []
    last_date = last_data_series.index[-1]
    
    for _ in range(n_days):
        X_input = current_features.reshape(1, -1)
        
        # 1. Prediksi dari setiap pohon (untuk CI)
        all_tree_preds = [tree.predict(X_input)[0] for tree in model.estimators_]
        
        # 2. Hitung Statistik
        mean_prediction = np.mean(all_tree_preds)
        std_prediction = np.std(all_tree_preds)
        
        # 3. Hitung Batas 95% CI
        lower_bound = mean_prediction - Z_SCORE * std_prediction
        upper_bound = mean_prediction + Z_SCORE * std_prediction
        
        predictions_data.append({
            'NO2_Prediction': mean_prediction,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })

        # 4. Update fitur rekursif (gunakan mean_prediction sebagai Lag_1 hari berikutnya)
        current_features = np.roll(current_features, 1)
        current_features[0] = mean_prediction
        
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days, freq='D')
    return pd.DataFrame(predictions_data, index=forecast_dates)

# ----------------------------------------------------
# C. FUNGSI UTAMA (TRAINING & LOADING)
# ----------------------------------------------------

def prepare_and_train_all(file_path, n_lags, test_size_days):
    """Fungsi utama untuk melatih RF dan evaluasi."""
    df = load_and_clean_data(file_path)
    
    # --- RF Training ---
    X_rf, y_rf = create_lags(df, n_lags)
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = split_data(X_rf, y_rf, test_size_days)
    rf_model = train_rf(X_train_rf, y_train_rf)
    
    # Evaluasi metrik
    y_pred_rf = rf_model.predict(X_test_rf)
    mape_rf = mean_absolute_percentage_error(y_test_rf, y_pred_rf) * 100
    acf_rf = acf(y_test_rf - y_pred_rf, nlags=1, fft=True)[1]
    
    # Hasil
    last_data = get_last_data(df, n_lags)
    
    return {
        'rf_model': rf_model, 
        'full_df': df, 
        'last_data': last_data,
        'metrics_rf': {'mape': mape_rf, 'acf': acf_rf},
    }

def load_rf_model():
    """Memuat model RF yang sudah disimpan."""
    try:
        rf_model = joblib.load(MODEL_PATH)
        return rf_model
    except FileNotFoundError:
        return None