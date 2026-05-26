import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# =================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CARGA DE DATOS
# =================================================================
ruta_data = Path.home() / "Documents" / "tesis" / "data limpia" / "dataset_final_para_modelos.csv"

print("--- Cargando Dataset Maestro ---")
df_raw = pd.read_csv(ruta_data)
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

# TRANSFORMACIÓN A FORMATO ANCHO (Pivote para Multicanal)
# Esto crea una columna por cada canal: Agencias, Agentes, ATMs, Digital
df_pivot = df_raw.pivot(index='fecha', columns='canal', values='volumen')

# Recuperar variables exógenas (Feriados, Quincenas, etc.)
df_exo = df_raw.groupby('fecha')[['es_quincena', 'es_feriado', 'es_fin_semana', 'dia_semana']].first()

# Dataset Final: 4 canales + 4 exógenas = 8 dimensiones
df_multi = pd.concat([df_pivot, df_exo], axis=1).sort_index().fillna(0)
n_canales = len(df_pivot.columns)
n_features = df_multi.shape[1]

print(f"Dataset multicanal listo. Dimensiones: {df_multi.shape}")
print(f"Canales a predecir: {list(df_pivot.columns)}")

# =================================================================
# 2. PREPROCESAMIENTO Y ESCALAMIENTO
# =================================================================
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_multi)

def create_windowed_dataset(data, look_back=7):
    """Transforma la serie en aprendizaje supervisado"""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])      # 7 días previos de TODA la matriz
        y.append(data[i + look_back, 0:n_canales]) # Predecimos solo los volúmenes del día 8
    return np.array(X), np.array(y)

look_back = 7 # Una semana de memoria técnica
X, y = create_windowed_dataset(data_scaled, look_back)

# Split Cronológico (80% Entrenamiento, 20% Prueba)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =================================================================
# 3. ARQUITECTURA DE LA RED NEURONAL (LSTM MULTIVARIANTE)
# =================================================================
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    LSTM(100, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(n_canales) # Salida: Una neurona por cada canal
])

model.compile(optimizer='adam', loss='mse')

# Callback para evitar sobreajuste (Overfitting)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n--- Iniciando Entrenamiento de la Neurona ---")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=16, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# =================================================================
# 4. EVALUACIÓN Y DESESCALAMIENTO (Inversa)
# =================================================================
predictions_scaled = model.predict(X_test)

def get_inverse_transform(scaled_data, original_scaler, n_cols):
    """Regresa los datos a escala transaccional real"""
    # Creamos un dummy array para que el scaler pueda revertir
    dummy = np.zeros((len(scaled_data), n_features))
    dummy[:, 0:n_cols] = scaled_data
    return original_scaler.inverse_transform(dummy)[:, 0:n_cols]

y_real = get_inverse_transform(y_test, scaler, n_canales)
y_pred = get_inverse_transform(predictions_scaled, scaler, n_canales)

# =================================================================
# 5. GENERACIÓN DE RESULTADOS POR CANAL
# =================================================================
canales = list(df_pivot.columns)

print("\n" + "="*40)
print("MÉTRICAS FINALES (LSTM MULTICANAL)")
print("="*40)

for i, canal in enumerate(canales):
    mae = mean_absolute_error(y_real[:, i], y_pred[:, i])
    # Evitamos divisiones por cero en el MAPE
    mape = np.mean(np.abs((y_real[:, i] - y_pred[:, i]) / np.where(y_real[:, i] == 0, 1, y_real[:, i]))) * 100
    
    print(f"CANAL: {canal:10} | MAE: {mae:8.2f} | MAPE: {mape:6.2f}%")

# =================================================================
# 6. VISUALIZACIÓN DE APRENDIZAJE (LOSS CURVE)
# =================================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Error Entrenamiento')
plt.plot(history.history['val_loss'], label='Error Validación')
plt.title('Curva de Aprendizaje de la Red Neuronal')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.show()

# =================================================================
# 7. GRÁFICA COMPARATIVA FINAL (Ejemplo: Agentes Bancarios)
# =================================================================
# Cambia el índice 'i' para ver otros canales (0: Agencias, 1: Agentes, etc.)
idx_canal = 1 
plt.figure(figsize=(12, 6))
plt.plot(y_real[:, idx_canal], label='Transacciones Reales', color='gray', alpha=0.5)
plt.plot(y_pred[:, idx_canal], label='Predicción LSTM', color='red', linestyle='--')
plt.title(f'Desempeño Multicanal LSTM: {canales[idx_canal]}')
plt.legend()
plt.show()
