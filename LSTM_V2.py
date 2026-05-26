import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. CARGA Y PREPARACIÓN 
ruta_data = Path.home() / "Documents" / "tesis" / "data limpia" / "dataset_final_para_modelos.csv"
df_raw = pd.read_csv(ruta_data)
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

df_pivot = df_raw.pivot(index='fecha', columns='canal', values='volumen')
df_exo = df_raw.groupby('fecha')[['es_quincena', 'es_feriado', 'es_fin_semana', 'dia_semana']].first()
df_multi = pd.concat([df_pivot, df_exo], axis=1).sort_index().fillna(0)

n_canales = len(df_pivot.columns)
n_features = df_multi.shape[1]

# 2. ESCALAMIENTO Y VENTANA (Ajuste de Look-back a 21 días)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_multi)

# Aumentamos look_back a 21 para capturar el ciclo de quincena completo
look_back = 21 

def create_windowed_dataset(data, lb):
    X, y = [], []
    for i in range(len(data) - lb):
        X.append(data[i:(i + lb), :])
        y.append(data[i + lb, 0:n_canales])
    return np.array(X), np.array(y)

X, y = create_windowed_dataset(data_scaled, look_back)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. ARQUITECTURA SIMPLIFICADA Y ESTABILIZADA
# Menos neuronas evitan que el modelo "alucine" con tan poca data
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    LSTM(32, activation='tanh', return_sequences=True), # Reducción de 100 a 32
    BatchNormalization(), # Estabiliza el aprendizaje entre canales
    Dropout(0.1),
    LSTM(16, activation='tanh'), # Reducción de 50 a 16
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dense(n_canales)
])

# Ajustamos el learning rate (tasa de aprendizaje) para que no sea tan errático
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print(f"\n--- Entrenando LSTM V2 (Look-back: {look_back}) ---")
history = model.fit(
    X_train, y_train, 
    epochs=150, # Más épocas pero con EarlyStopping
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 4. EVALUACIÓN E INVERSA
pred_scaled = model.predict(X_test)

def get_inverse(scaled, original_scaler, nc):
    dummy = np.zeros((len(scaled), n_features))
    dummy[:, 0:nc] = scaled
    return original_scaler.inverse_transform(dummy)[:, 0:nc]

y_real = get_inverse(y_test, scaler, n_canales)
y_pred = get_inverse(pred_scaled, scaler, n_canales)

# 5. RESULTADOS
canales = list(df_pivot.columns)
print("\n" + "="*40)
print("MÉTRICAS COMPARATIVAS (LSTM V2)")
print("="*40)

for i, canal in enumerate(canales):
    mae = mean_absolute_error(y_real[:, i], y_pred[:, i])
    print(f"CANAL: {canal:10} | MAE: {mae:8.2f}")
