import pandas as pd
import numpy as np
import antropy as ant
from scipy.stats import spearmanr, entropy
from pathlib import Path
import nolds

# --- 1. CARGA DE DATOS ---
ruta_data = Path.home()
df_raw = pd.read_csv(ruta_data)
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

# Definición de Canales
canales = ['Canal 1', 'Canal 2', 'Canal 3', 'Canal 4']

# Ingreso manual de metricas obtenidas en otro código para referencia
mejoras_mae_lstm = {
    'Canal 1': 0.0,   
    'Canal 2': 0.0,       
    'Canal 3': 0.0,             
    'Canal 4': 0.0  
}

resumen_tesis = []

# --- 2. CÁLCULO DE ENTROPÍAS POR CANAL ---
for canal in canales:
    # Aislar la serie y eliminar nulos
    serie = df_raw[canal].dropna().values
    
    # Uso de la serie en su primera diferencia para asegurar estacionariedad
    serie_diff = np.diff(serie)
    
    # Entropía de Permutación (Bandt-Pompe, m=4, tau=1)
    # normalizada para que todos los canales sean comparables de 0 a 1
    perm_ent = ant.perm_entropy(serie_diff, order=4, normalize=True)
    
    # Entropía Muestral (m=2)
    # antropy por defecto asume tolerancia r = 0.2 * std(serie), cumpliendo la orden del experto
    samp_ent = ant.sample_entropy(serie_diff, order=2)

    # Exponente de Lyapunov (Algoritmo de Rosenstein)
    # min_tsep aisla temporalmente los vectores vecinos. Usamos 10 (la ventana de tu LSTM)
    lyap_max = nolds.lyap_r(serie_diff, emb_dim=10, lag=1, min_tsep=10)
    
    resumen_tesis.append({
        'Canal': canal,
        'Entropia_Permutacion': perm_ent,
        'Entropia_Muestral': samp_ent,
        'Mejora_MAE_LSTM_pct': mejoras_mae_lstm[canal],
        'Lyapunov_Max_Data': lyap_max,
    })

# Convertir a DataFrame (Tu tabla final a exportar)
df_resultados_entropia = pd.DataFrame(resumen_tesis)
print("=== TABLA DE ENTROPÍAS BASALES ===")
print(df_resultados_entropia.to_string(index=False))

# --- 3. PRUEBA DE HIPÓTESIS (Reemplazo del Scatter Plot por restricción a visuales. Se sugiere usar Scatter Plot si es posible) ---
rho_perm, pval_perm = spearmanr(df_resultados_entropia['Entropia_Permutacion'], df_resultados_entropia['Mejora_MAE_LSTM_pct'])
rho_samp, pval_samp = spearmanr(df_resultados_entropia['Entropia_Muestral'], df_resultados_entropia['Mejora_MAE_LSTM_pct'])

print("\n=== CORRELACIÓN DE SPEARMAN (Rangos) ===")
print(f"Permutación vs Mejora MAE: rho = {rho_perm:.4f}, p-value = {pval_perm:.4f}")
print(f"Muestral vs Mejora MAE:    rho = {rho_samp:.4f}, p-value = {pval_samp:.4f}")
