from statsmodels.stats.diagnostic import het_arch

def ejecutar_heterocedasticidad(serie, titulo):
    print(f"\n{'='*40}")
    print(f"PRUEBA DE HETEROCEDASTICIDAD (ARCH): {titulo}")
    print(f"{'='*40}")
    
    # La prueba se realiza sobre la serie diferenciada (d=1) 
    # para evaluar si los cambios en el volumen tienen varianza constante
    serie_diff = serie.diff().dropna()
    
    # Ejecutamos la prueba de Engle
    # lm: estadístico de Lagrange, lmpv: p-value
    lm, lmpv, f_stat, f_pv = het_arch(serie_diff)
    
    print(f"Estadístico LM: {lm:.4f}")
    print(f"p-value (LM): {lmpv:.4e}")
    
    # Interpretación científica para la Sección VI.B.5
    if lmpv <= 0.05:
        print(f"\nCONCLUSIÓN: p-value < 0.05. Existe HETEROCEDASTICIDAD significativa.")
        print("La varianza no es constante (existen efectos ARCH).")
        print("Esto sugiere que el sistema tiene 'memoria en la volatilidad'.")
    else:
        print(f"\nCONCLUSIÓN: p-value > 0.05. No hay evidencia de heterocedasticidad.")
        print("La varianza es aproximadamente constante (Homocedasticidad).")

# Ejecución sobre el 80% de entrenamiento de ATMs
ejecutar_heterocedasticidad(train_atms['volumen'], "Cajeros Automáticos (ATMs)")
