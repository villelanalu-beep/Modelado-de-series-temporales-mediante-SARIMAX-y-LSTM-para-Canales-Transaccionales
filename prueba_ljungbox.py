from statsmodels.stats.diagnostic import acorr_ljungbox

def ejecutar_ljungbox(serie, titulo):
    print(f"\n{'='*40}")
    print(f"PRUEBA LJUNG-BOX: {titulo}")
    print(f"{'='*40}")
    
    # Ejecutamos la prueba sobre la serie diferenciada (d=1)
    # Probamos hasta 10 rezagos para ver la autocorrelación de corto plazo
    res_lb = acorr_ljungbox(serie.diff().dropna(), lags=[10], return_df=True)
    
    lb_stat = res_lb['lb_stat'].values[0]
    p_value = res_lb['lb_pvalue'].values[0]
    
    print(f"Estadístico de prueba: {lb_stat:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    # Interpretación para la tesis
    if p_value <= 0.05:
        print(f"\nCONCLUSIÓN: p-value < 0.05. Se RECHAZA la independencia serial.")
        print("Existe una estructura de dependencia significativa (No es ruido blanco).")
        print("Esto justifica el uso de modelos predictivos avanzados.")
    else:
        print(f"\nALERTA: p-value > 0.05. No se rechaza la independencia serial.")
        print("La serie se comporta como ruido blanco. Un modelo complejo podría no aportar valor.")

# Ejecución sobre el 80% de entrenamiento del canal
ejecutar_ljungbox(train_canal['volumen'], "Canal")
