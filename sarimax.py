import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================
ruta_data = Path.home() / "Documents" / "tesis" / "data limpia" / "nombre del csv"
CANAL_OBJETIVO = 'Canal'  # <--- Cambia segun canal

df = pd.read_csv(ruta_data)
df['fecha'] = pd.to_datetime(df['fecha'])

# Filtrado por canal y preparación del índice
df_model = df[df['canal'] == CANAL_OBJETIVO].copy()
df_model = df_model.set_index('fecha').sort_index()

# Asegurar frecuencia diaria (rellenar huecos si los hubiera con 0)
df_model = df_model.asfreq('D').fillna(0)

print(f"--- Entrenando SARIMAX para: {CANAL_OBJETIVO} ---")
print(f"Rango de datos: {df_model.index.min()} a {df_model.index.max()}")

# ==========================================
# 2. DEFINICIÓN DE VARIABLES
# ==========================================
# Variable objetivo
y = df_model['volumen']

# Matriz de variables exógenas (X)
# Incluimos quincena, feriado, fin de semana y el día de la semana
exog_cols = ['es_quincena', 'es_feriado', 'es_fin_semana', 'dia_semana']
X = df_model[exog_cols]

# Split Entrenamiento (80%) y Prueba (20%)
split = int(len(df_model) * 0.8)
y_train, y_test = y[:split], y[split:]
X_train, X_test = X[:split], X[split:]

# ==========================================
# 3. CONFIGURACIÓN Y AJUSTE DEL MODELO
# ==========================================
# Parámetros basados diagnóstico previo:
# order=(p,d,q), seasonal_order=(P,D,Q,S)
modelo = SARIMAX(y_train, 
                 exog=X_train,
                 order=(1, 1, 1), 
                 seasonal_order=(1, 1, 1, 7),
                 enforce_stationarity=False,
                 enforce_invertibility=False)

resultado = modelo.fit(disp=False)
print(resultado.summary())

# ==========================================
# 4. PREDICCIÓN Y MÉTRICAS
# ==========================================
# Predicción sobre el set de prueba
prediccion = resultado.get_forecast(steps=len(y_test), exog=X_test)
y_pred = prediccion.predicted_mean
intervalos = prediccion.conf_int()

# Cálculo de errores
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\nResultados de Validación:")
print(f"MAE: {mae:.2f} transacciones")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# ==========================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(y_train.index[-30:], y_train[-30:], label='Entrenamiento (últimos 30d)')
plt.plot(y_test.index, y_test, label='Real (Test)', color='gray', alpha=0.5)
plt.plot(y_test.index, y_pred, label='Predicción SARIMAX', color='red', linestyle='--')
plt.fill_between(y_test.index, intervalos.iloc[:, 0], intervalos.iloc[:, 1], color='pink', alpha=0.3)
plt.title(f"Predicción Multicanal: {CANAL_OBJETIVO} (SARIMAX + Exógenas)")
plt.legend()
plt.show()
