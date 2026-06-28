import pandas as pd
import numpy as np
import ruptures as rpt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# =================================================================
# 1. PREPARACIÓN DEL ECOSISTEMA (Físico vs Digital)
# =================================================================
ruta_data = Path.home() / 
df_raw = pd.read_csv(ruta_data)
df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

# Agrupación macro para la transición de fase
df_pivot = df_raw.pivot(index='fecha', columns='canal', values='volumen').fillna(0)
df_pivot['Fisico'] = df_pivot['Canal 1'] + df_pivot['Canal 2'] # Los "Puntos de Atención"
df_pivot['Virtual'] = df_pivot['Canal 3'] + df_pivot['Canal 4 ']
df_pivot['Total'] = df_pivot['Fisico'] + df_pivot['Virtual']

print("\n" + "="*70)
# =================================================================
# PIEZA 1: DATACIÓN DE RUPTURA ESTRUCTURAL (ALGORITMO PELT)
# =================================================================
print("1. DATACIÓN DE RUPTURA ESTRUCTURAL (Puntos Físicos)")
serie_fisica = df_pivot['Fisico'].values

# Modelo RBF (Radial Basis Function) detecta cambios en media y varianza
algo = rpt.Pelt(model="rbf").fit(serie_fisica)
rupturas_idx = algo.predict(pen=10) # pen=10 es la penalización estándar para evitar sobreajuste

fechas_ruptura = [df_pivot.index[i-1].strftime('%Y-%m-%d') for i in rupturas_idx if i < len(df_pivot)]
print(f"Fechas exactas de ruptura de régimen detectadas: {fechas_ruptura}")
print("-" * 70)

# =================================================================
# PIEZA 2: MODELO DE RÉGIMEN MARKOVIANO (ALTA VS BAJA VOLATILIDAD)
# =================================================================
print("2. MATRIZ DE TRANSICIÓN DE REGÍMENES (MarkovRegression)")
# Usamos la primera diferencia para aislar la volatilidad pura
serie_fisica_stat = np.diff(serie_fisica)

# Ajuste de 2 regímenes con varianza cambiante (switching_variance=True)
modelo_markov = sm.tsa.MarkovRegression(serie_fisica_stat, k_regimes=2, trend='n', switching_variance=True)
res_markov = modelo_markov.fit(disp=False)

# Extracción robusta de duraciones y probabilidades
duraciones = res_markov.expected_durations
p00 = 1 - (1 / duraciones[0])
p11 = 1 - (1 / duraciones[1])

print(f"Régimen 0 (Estabilidad):    P(0|0) = {p00:.4f} | Duración Esperada = {duraciones[0]:.2f} días")
print(f"Régimen 1 (Caos/Volatilidad): P(1|1) = {p11:.4f} | Duración Esperada = {duraciones[1]:.2f} días")
print("-" * 70)

# =================================================================
# PIEZA 3: PARÁMETRO DE ORDEN Y SUSCEPTIBILIDAD (PUNTO CRÍTICO)
# =================================================================
print("3. PUNTO CRÍTICO DE MIGRACIÓN DIGITAL")
# Ecuación del asesor: Phi(t) = virtual / total
df_pivot['phi'] = df_pivot['Virtual'] / df_pivot['Total']

# Susceptibilidad: Varianza móvil de Phi (ventana de 30 días)
df_pivot['susceptibilidad'] = df_pivot['phi'].rolling(window=30).var()

# El punto pseudo-crítico es donde la susceptibilidad (varianza) explota
punto_critico_idx = df_pivot['susceptibilidad'].idxmax()
phi_critico = df_pivot.loc[punto_critico_idx, 'phi']
max_susceptibilidad = df_pivot.loc[punto_critico_idx, 'susceptibilidad']

print(f"Fecha del Punto Crítico (Máxima Susceptibilidad): {punto_critico_idx.strftime('%Y-%m-%d')}")
print(f"Nivel de penetración digital Phi(t) en el punto crítico: {phi_critico:.4f}")
print("=" * 70)

# =================================================================
# GRÁFICA VISUAL (Exclusiva para validación local)
# =================================================================
plt.figure(figsize=(10, 5))
plt.plot(df_pivot.index, df_pivot['phi'], label=r'Parámetro de Orden $\phi(t)$ (Virtual / Total)', color='steelblue')
plt.plot(df_pivot.index, df_pivot['susceptibilidad'] * 50, label=r'Susceptibilidad $\chi$ (Varianza móvil escalada)', color='firebrick')
plt.axvline(punto_critico_idx, color='black', linestyle='--', label=f'Punto Crítico: {punto_critico_idx.strftime("%Y-%m-%d")}')
plt.title('Transición de Fase: Migración del Banco a Canales Virtuales')
plt.ylabel('Proporción')
plt.legend()
plt.tight_layout()
plt.show()
