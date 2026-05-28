import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from scipy.stats import norm

# ==========================================
# 1. FUNCIONES ESTADÍSTICAS DE RIGOR
# ==========================================
def calcular_smape(actual, pred):
    return 100/len(actual) * np.sum(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + 1e-10))

def diebold_mariano_test(real, pred_1, pred_2):
    e1 = np.abs(real - pred_1)
    e2 = np.abs(real - pred_2)
    d = e1 - e2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var == 0: return 0, 1.0
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 1 - norm.cdf(dm_stat) 
    return dm_stat, p_value

# ==========================================
# 2. CARGA Y CONFIGURACIÓN
# ==========================================
ruta_data = Path.home() / "Documents" / "tesis" / "data limpia" / "nombre del csv"
df_raw = pd.read_csv(ruta_data)
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

canales = ['Listado nombres de canales']
resumen_tesis = []

for canal in canales:
    print(f"\n" + "="*60)
    print(f"BATTLE: SARIMAX vs LSTM - CANAL: {canal}")
    print("="*60)
    
    df_c = df_raw[df_raw['canal'] == canal].set_index('fecha').sort_index()
    y = df_c['volumen']
    X_exog = df_c[['es_quincena', 'es_feriado', 'es_fin_semana', 'dia_semana']]
    
    # Split 80/20 (Idéntico para ambos)
    split = int(len(df_c) * 0.8)
    y_train, y_test = y[:split], y[split:]
    X_train, X_test = X_exog[:split], X_exog[split:]

    # --- MODELO 1: SARIMAX (Baseline) ---
    print("Entrenando SARIMAX...")
    mod_sarimax = SARIMAX(y_train, exog=X_train, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    pred_sarimax = mod_sarimax.forecast(steps=len(y_test), exog=X_test)

    # --- MODELO 2: LSTM V3 (Independiente) ---
    print("Entrenando LSTM v4...")
    scaler = MinMaxScaler()
    data_s = scaler.fit_transform(df_c[['volumen', 'es_quincena', 'es_feriado', 'es_fin_semana', 'dia_semana']])
    
    look_back = 10
    X_lstm, y_lstm = [], []
    for i in range(len(data_s)-look_back):
        X_lstm.append(data_s[i:i+look_back, :])
        y_lstm.append(data_s[i+look_back, 0])
    
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    # Ajustar split por el look_back
    split_l = int(len(X_lstm) * 0.8)
    X_l_train, X_l_test = X_lstm[:split_l], X_lstm[split_l:]
    y_l_test = y_lstm[split_l:]

    model_lstm = Sequential([
        Input(shape=(X_l_train.shape[1], X_l_train.shape[2])),
        LSTM(64, activation='relu'), Dropout(0.2), Dense(32, activation='relu'), Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_l_train, y_lstm[:split_l], epochs=100, batch_size=16, verbose=0)
    
    pred_l_scaled = model_lstm.predict(X_l_test)
    # Reversión de escala
    dummy = np.zeros((len(pred_l_scaled), 5))
    dummy[:, 0] = pred_l_scaled.flatten()
    pred_lstm = scaler.inverse_transform(dummy)[:, 0]
    
    # Sincronizar longitudes para Diebold-Mariano (debido al look_back)
    real_final = y_test[-len(pred_lstm):].values
    sarimax_final = pred_sarimax[-len(pred_lstm):].values

    # --- MÉTRICAS DE COMPARACIÓN ---
    mae_s = mean_absolute_error(real_final, sarimax_final)
    mae_l = mean_absolute_error(real_final, pred_lstm)
    rmse_l = np.sqrt(mean_squared_error(real_final, pred_lstm))
    smape_l = calcular_smape(real_final, pred_lstm)
    
    # Ljung-Box sobre residuos LSTM
    residuos = real_final - pred_lstm
    lb_p = acorr_ljungbox(residuos, lags=[10], return_df=True)['lb_pvalue'].values[0]
    
    # TEST DE DIEBOLD-MARIANO
    dm_stat, p_val = diebold_mariano_test(real_final, sarimax_final, pred_lstm)

    resumen_tesis.append({
        'Canal': canal,
        'MAE_SARIMAX': mae_s,
        'MAE_LSTM': mae_l,
        'RMSE_LSTM': rmse_l,
        'sMAPE_LSTM': smape_l,
        'DM_p_value': p_val,
        'LjungBox_p': lb_p,
        'Gana_IA': 'SÍ' if p_val < 0.05 else 'NO'
    })

# ==========================================
# 3. CUADRO DE RESULTADOS FINAL PARA TESIS
# ==========================================
df_final_tesis = pd.DataFrame(resumen_tesis)
print("\n" + "!"*70)
print("CUADRO COMPARATIVO FINAL - RESULTADOS DE TESIS")
print("!"*70)
print(df_final_tesis.to_string(index=False))
