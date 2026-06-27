import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg

# =================================================================
# 1. CARGA DE DATOS
# =================================================================
ruta_data = Path.home() /
df_raw = pd.read_csv(ruta_data)

canales = ['Canal 1', 'Canal 2', 'Canal 3', 'Canal 4']
resultados_ou = []

# =================================================================
# 2. CÁLCULO DE PARÁMETROS OU POR CANAL
# =================================================================
for canal in canales:
    # Extraer volumen del canal
    serie = df_raw[df_raw['canal'] == canal]['volumen'].dropna().values
    
    # 1. Serie Estacionaria (Primera diferencia)
    serie_stat = np.diff(serie)
    
    # 2. Nivel mu (Media muestral de la serie estacionaria)
    mu = np.mean(serie_stat)
    
    # 3. Serie centrada en la media
    serie_centrada = serie_stat - mu
    
    # 4. Ajuste AR(1) sin tendencia (trend='n' porque ya está centrada)
    # Instrucción directa del asesor: AutoReg(x, lags=1)
    modelo_ar = AutoReg(serie_centrada, lags=1, trend='n').fit()
    
    # Extraer coeficiente Phi (phi_hat)
    phi = modelo_ar.params[0]
    
    # 5. Calcular métricas físicas
    theta = 1 - phi                          # Tasa de reversión a la media (Fricción)
    sigma = np.std(modelo_ar.resid)          # Amplitud del ruido (Innovación)
    tiempo_relajacion = 1 / theta if theta != 0 else np.inf  # 1 / theta
    
    resultados_ou.append({
        'Canal': canal,
        'Theta (Friccion)': theta,
        'Nivel Mu (Media)': mu,
        'Sigma (Ruido)': sigma,
        'Relajacion (1/Theta)': tiempo_relajacion
    })

# =================================================================
# 3. ENTREGABLE FINAL
# =================================================================
df_ou = pd.DataFrame(resultados_ou)
print("\n" + "=" * 80)
print("TABLA DE PARÁMETROS ORNSTEIN-UHLENBECK (FRICCIÓN FÍSICA)")
print("=" * 80)
print(df_ou.to_string(index=False))
print("=" * 80)
