import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# =================================================================
# 1. CARGA DE DATOS Y DEFINICIÓN DE MÉTRICAS
# =================================================================
ruta_data = Path.home() / "Documents" / "tesis" / "data limpia" #/"nombre del csv"
df_raw = pd.read_csv(ruta_data)
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

def calcular_smape(actual, pred):
    return 100/len(actual) * np.sum(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + 1e-10))

# Canales a procesar
canales = [#listado de nombres de los canales según el set]
resultados_finales = []

# =================================================================
# 2. BUCLE DE ENTRENAMIENTO INDEPENDIENTE
# =================================================================
for canal in canales:
    print(f"\n" + "="*50)
    print(f"PROCESANDO CANAL: {canal.upper()}")
    print("="*50)
    
    # Filtrado por canal
    df_canal = df_raw[df_raw['canal'] == canal].copy()
    df_canal = df_canal.set_index('fecha').sort_index()
    
    # Variables: Volumen + Exógenas
    cols_features = ['volumen', 'es_quincena', 'es_feriado', 'es_fin_semana', 'dia_semana']
    data = df_canal[cols_features].values.astype('float32')
    
    # Escalamiento por canal (MinMax individual)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Ventana de tiempo (10 días como en V3 para ser justos)
    look_back = 10
    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:(i + look_back), :]) # Features
        y.append(data_scaled[i + look_back, 0])    # Target (Volumen)
    
    X, y = np.array(X), np.array(y)
    
    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Arquitectura Parsimoniosa V3
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Entrenamiento
    model.fit(X_train, y_train, epochs=100, batch_size=16, 
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)
    
    # Predicción y Reversión de Escala
    pred_scaled = model.predict(X_test)
    
    # Reversión manual del volumen (columna 0)
    # Creamos dummy para el inverse_transform
    dummy_pred = np.zeros((len(pred_scaled), len(cols_features)))
    dummy_pred[:, 0] = pred_scaled.flatten()
    y_pred_inv = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_real = np.zeros((len(y_test), len(cols_features)))
    dummy_real[:, 0] = y_test.flatten()
    y_real_inv = scaler.inverse_transform(dummy_real)[:, 0]
    
    # Cálculo de métricas
    mae = mean_absolute_error(y_real_inv, y_pred_inv)
    mape = np.mean(np.abs((y_real_inv - y_pred_inv) / np.where(y_real_inv == 0, 1, y_real_inv))) * 100
    smape_val = calcular_smape(y_real_inv, y_pred_inv)
    
    resultados_finales.append({
        'Canal': canal,
        'MAE': mae,
        'MAPE (%)': mape,
        'sMAPE (%)': smape_val
    })
    
    print(f"RESULTADO {canal}: MAE={mae:.2f} | sMAPE={smape_val:.2f}%")

# =================================================================
# 3. TABLA COMPARATIVA FINAL
# =================================================================
df_res = pd.DataFrame(resultados_finales)
print("\n" + "!"*60)
print("TABLA COMPARATIVA FINAL - LSTM V4 (INDEPENDIENTE)")
print("!"*60)
print(df_res.to_string(index=False))
print("="*60)
