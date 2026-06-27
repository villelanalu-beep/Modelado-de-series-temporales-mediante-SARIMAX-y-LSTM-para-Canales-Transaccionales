from sklearn.feature_selection import mutual_info_regression

#ESTE CÓDIGO SE INCLUYE COMO PARTE DE LA CONFIGURACIÓN DE LA LSTM, EN EL ESTUDIO SE APLICA LUEGO DE LA AUDITORÍA DE DE ENTROPÍA#

# =================================================================
    # INICIO: DEMONIO DE MAXWELL (COMPUERTA DE OLVIDO f_t)
    # =================================================================
    capa_lstm = model.get_layer(nombre_capa_lstm)
    W, U, b = capa_lstm.get_weights()
    unidades = capa_lstm.units

    # Aislar pesos de la compuerta de olvido (f)
    W_f = W[:, unidades:2*unidades]
    U_f = U[:, unidades:2*unidades]
    b_f = b[unidades:2*unidades]

    # Aislar x_t (día actual) y crear h_prev basal
    x_t = X_test[:, -1, :] 
    h_prev = np.zeros((len(X_test), unidades)) 

    # Reconstrucción matemática de la decisión
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    activacion_f_t = np.dot(x_t, W_f) + np.dot(h_prev, U_f) + b_f
    compuerta_olvido = sigmoid(activacion_f_t)
    
    # Señal media temporal (bar{f}_t)
    f_media_t = np.mean(compuerta_olvido, axis=1)

    # Indicador Exógeno (e_t) - Columnas: 1=Quincena, 2=Feriado
    e_t_quincena = X_test[:, -1, 1]
    e_t_feriado = X_test[:, -1, 2]
    e_t_eventos = np.clip(e_t_quincena + e_t_feriado, 0, 1)

    # Información Mutua I(f_t ; e_t)
    mi_score = mutual_info_regression(f_media_t.reshape(-1, 1), e_t_eventos, random_state=42)[0]

    # Medias condicionales para el reporte escrito
    f_media_eventos = np.mean(f_media_t[e_t_eventos == 1])
    f_media_normal = np.mean(f_media_t[e_t_eventos == 0])
    
    # Gráfico (Solo saltará para Agencias, ciérralo para que el bucle continúe)
    if canal == 'Agencias':
        plt.figure(figsize=(10, 4))
        plt.plot(f_media_t, label='Apertura Media $\\bar{f}_t$', color='blue')
        plt.fill_between(range(len(e_t_eventos)), 0, 1, where=(e_t_eventos==1), color='red', alpha=0.3, label='Evento ($e_t$)')
        plt.title(f'Demonio de Maxwell - {canal} | Info Mutua: {mi_score:.4f}')
        plt.legend()
        plt.show()
    # =================================================================
    # FIN DEMONIO DE MAXWELL
    # =================================================================

#PARA LA MUESTRA DE RESULTADOS SE DEBE AGREGAR AL APPEND RESULATDOS DEL LSTM, COMO SE MUESTRA A CONTINUACIÓN#
# Guardar resultados
    resultados_finales.append({
        'Canal': canal,
        'MAE': mae,
        'MAPE (%)': mape,
        'sMAPE (%)': smape_val,
        'Media_Entropia_H': media_H,
        'Varianza_Entropia_H': var_H,
        'Info_Mutua_ft': mi_score,
        'f_media_Normal': f_media_normal,
        'f_media_Eventos': f_media_eventos
    })
