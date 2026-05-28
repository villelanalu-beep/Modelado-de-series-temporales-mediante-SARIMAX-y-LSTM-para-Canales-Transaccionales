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
test_estacionariedad(train_canal1['volumen'].diff(), "Canal1")
test_estacionariedad(train_canal2['volumen'].diff(), "Canal2")
test_estacionariedad(df_canal3['volumen'].diff(), "Canal3")
test_estacionariedad(df_canal4['volumen'].diff(), "Canal4")
