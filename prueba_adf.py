from statsmodels.tsa.stattools import adfuller

def test_estacionariedad(serie, nombre):
    print(f"\n--- Prueba ADF: {nombre} ---")
    res = adfuller(serie.dropna())
    print(f"Estadístico ADF: {res[0]:.4f}")
    print(f"p-value: {res[1]:.4e}")
    if res[1] <= 0.05:
        print("Resultado: La serie es ESTACIONARIA (Rechaza H0)")
    else:
        print("Resultado: La serie NO es estacionaria (No rechaza H0)")

# Aplicar a tus sets de entrenamiento diferenciados
test_estacionariedad(train_atms['volumen'].diff(), "ATMs")
test_estacionariedad(train_digital['volumen'].diff(), "Digital")
test_estacionariedad(df_agencias['volumen'].diff(), "Agencias")
test_estacionariedad(df_agencias['volumen'].diff(), "Agentes Bancarios")
