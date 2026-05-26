import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

# 1. Carga de datos
ruta_csv = Path.home() / "Documents" / "serie_diaria_atms_limpia.csv"
df = pd.read_csv(ruta_csv)
df['fecha_dt'] = pd.to_datetime(df['fecha_dt'])
df = df.sort_values('fecha_dt').set_index('fecha_dt')

# 2. Definición del punto de corte (80% Entrenamiento)
split_point = int(len(df) * 0.8)
train_data = df.iloc[:split_point]
test_data = df.iloc[split_point:]

print(f"Total registros: {len(df)}")
print(f"Registros Entrenamiento: {len(train_data)}")
print(f"Registros Prueba: {len(test_data)}")

# 3. Preparación de la señal: Primera diferencia (d=1)
# Esto remueve la tendencia para que la dependencia temporal sea clara
train_diff = train_data['volumen'].diff().dropna()

# 4. Generación de Correlogramas (ACF y PACF)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ACF (Ayuda a identificar el componente MA 'q')
plot_acf(train_diff, lags=40, ax=ax1, title="Autocorrelación (ACF) - Set de Entrenamiento (d=1)")

# PACF (Ayuda a identificar el componente AR 'p')
plot_pacf(train_diff, lags=40, ax=ax2, title="Autocorrelación Parcial (PACF) - Set de Entrenamiento (d=1)")

plt.tight_layout()
plt.show()

