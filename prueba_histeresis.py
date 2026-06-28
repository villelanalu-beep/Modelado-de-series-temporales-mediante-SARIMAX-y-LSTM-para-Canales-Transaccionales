import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from pathlib import Path

# =================================================================
# 1. CARGA DE DATOS Y FEATURE ENGINEERING DEL CICLO
# =================================================================
ruta_data = Path.home()
df_raw = pd.read_csv(ruta_data)
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

# Definir la ventana "Post-Bono" (Julio y Diciembre en Guatemala)
df_raw['mes'] = df_raw['fecha'].dt.month
df_raw['es_post_bono'] = df_raw['mes'].isin([7, 12]).astype(int)

# Definir la fase del ciclo quincenal (Días 0 a 14)
# Día 0 = Día de pago (15 o fin de mes). Días 1-14 = Distancia desde el pago.
def calcular_fase_quincenal(fecha):
    d = fecha.day
    if d == 15 or d == fecha.days_in_month:
        return 0
    elif d < 15:
        return d
    else:
        return d - 15

df_raw['fase_ciclo'] = df_raw['fecha'].apply(calcular_fase_quincenal)

canales = ['Canal 1', 'Canal 2', 'Canal 3', 'Canal 4']
resultados_histeresis = []

# =================================================================
# 2. BUCLE DE AUDITORÍA POR CANAL
# =================================================================
for canal in canales:
    df_c = df_raw[df_raw['canal'] == canal].copy()
    
    # --- PRUEBA 1 (ESTADÍSTICA): KOLMOGOROV-SMIRNOV ---
    # Condiciones emparejadas: Días ordinarios (es_quincena == 0, día laboral estándar)
    # Filtramos para comparar cómo se comporta un día normal si viene después de un mes de Bono vs uno normal.
    df_control = df_c[(df_c['es_quincena'] == 0) & (df_c['es_feriado'] == 0) & (df_c['es_fin_semana'] == 0)]
    
    vol_post_bono = df_control[df_control['es_post_bono'] == 1]['volumen'].values
    vol_ordinario = df_control[df_control['es_post_bono'] == 0]['volumen'].values
    
    # Prueba KS de 2 muestras
    ks_stat, p_val = ks_2samp(vol_post_bono, vol_ordinario)
    
    # --- PRUEBA 2 (FÍSICA): ÁREA DEL LAZO DE HISTÉRESIS ---
    # Demanda media por cada día del ciclo quincenal
    vol_medio_fase = df_c.groupby('fase_ciclo')['volumen'].mean()
    
    # Semiciclo Ascendente (Días 1->7 post-pago)
    ascendente = [vol_medio_fase.get(i, 0) for i in range(1, 8)]
    # Semiciclo Descendente (Días 14->8) mapeado a la misma distancia para cerrar el lazo
    descendente = [vol_medio_fase.get(15-i, 0) for i in range(1, 8)]
    
    # El área encerrada entre ambas curvas es la fuerza de la Histéresis
    area_lazo = np.sum(np.abs(np.array(ascendente) - np.array(descendente)))
    
    resultados_histeresis.append({
        'Canal': canal,
        'KS_Statistic': ks_stat,
        'KS_P_Value': p_val,
        'Area_Lazo': area_lazo
    })

    # --- GRÁFICO VISUAL (Aplicado solo a 1 canal para el estudio)
    if canal == 'Canal 1':
        plt.figure(figsize=(9, 5))
        plt.plot(range(1, 8), ascendente, label='Semiciclo Ascendente (Días 1→7 post-pago)', marker='o', color='firebrick')
        plt.plot(range(1, 8), descendente, label='Semiciclo Descendente (Días 14→8 pre-pago)', marker='s', color='steelblue')
        plt.fill_between(range(1, 8), ascendente, descendente, color='gray', alpha=0.2, label='Área de Histéresis')
        plt.title(f'Lazo de Histéresis de Demanda - {canal}\nPrueba de Dependencia de Trayectoria')
        plt.xlabel('Fase (Días de distancia desde el pago quincenal)')
        plt.ylabel('Volumen Promedio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# =================================================================
# 3. ENTREGABLE FINAL
# =================================================================
df_res = pd.DataFrame(resultados_histeresis)
print("\n" + "=" * 70)
print("TABLA DE HISTÉRESIS Y DEPENDENCIA DE TRAYECTORIA (KS)")
print("=" * 70)
print(df_res.to_string(index=False))
print("=" * 70)
