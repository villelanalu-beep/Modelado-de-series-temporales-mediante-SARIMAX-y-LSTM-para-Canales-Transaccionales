import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine 

# 1. Configuración de conexión (Ajusta con tus credenciales de BAM)
def get_agencias_data():
    
    # Para fines de prueba
    df = pd.read_csv('agencias_volumen.csv') 
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

# 2. Ejecución de la inspección visual
df_agencias = get_agencias_data()

# 3. Graficación con "Wildcard" (Promedio móvil para ver la tendencia)
plt.figure(figsize=(15, 8))

# Señal original (Ruido + Estacionalidad + Tendencia)
plt.plot(df_agencias['fecha'], df_agencias['volumen'], 
         color='#007a33', alpha=0.4, label='Volumen Diario (Señal Original)')

# Promedio móvil de 7 días (Para filtrar la estacionalidad semanal y ver la tendencia)
df_agencias['tendencia_7d'] = df_agencias['volumen'].rolling(window=7).mean()
plt.plot(df_agencias['fecha'], df_agencias['tendencia_7d'], 
         color='#004b8d', linewidth=2, label='Promedio Móvil (7 días)')

plt.title('Inspección Gráfica: Demanda Transaccional - Canal Agencias', fontsize=14)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Volumen de Transacciones', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
