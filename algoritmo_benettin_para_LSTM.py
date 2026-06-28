# ESTE CÓDIGO ESTA PENSADO PARA UTILIZARSE DENTRO DEL MODELO DE LSTM, SE PRESENTA POR SEPARADO PARA AÑADIRLO COMO OPCIONAL YA QUE AUMENTA EL COSTO COMPUTACIONAL#
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN # Necesaria para el baseline

#COLOCAR SIGUIENTE LUEGO DEL PRED_SCALER EN LSTM PARA SU APLICACIÓN#
# =================================================================
    # INICIO: DIAGNÓSTICO DE LYAPUNOV (NIVEL 2)
    # =================================================================
    print("Calculando Jacobiano Recurrente de LSTM V3")
    
    # 1. Extraer la celda real de tu LSTM entrenada
    capa_lstm = [l for l in model.layers if isinstance(l, LSTM)][0]
    celda_lstm = capa_lstm.cell
    unidades_lstm = capa_lstm.units

    def calcular_lyapunov_celda(celda, tipo, X_seq, unidades):
        Q = np.eye(unidades)
        lyap_sum = np.zeros(unidades)
        
        if tipo == 'LSTM':
            h_prev = [tf.zeros([1, unidades]), tf.zeros([1, unidades])]
        else:
            h_prev = [tf.zeros([1, unidades])]
            
        T_len = len(X_seq)
        for t in range(T_len):
            x_t = tf.convert_to_tensor(X_seq[t, -1, :][np.newaxis, :], dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(h_prev)
                out, h_new = celda(x_t, states=h_prev)
                
            if tipo == 'LSTM':
                # Derivamos el Cell State (c_t) que es la verdadera memoria
                J_t = tape.jacobian(h_new[1], h_prev[1])[0, :, 0, :]
            else:
                J_t = tape.jacobian(h_new[0], h_prev[0])[0, :, 0, :]
                
            A = np.dot(J_t.numpy(), Q)
            Q, R = np.linalg.qr(A)
            lyap_sum += np.log(np.abs(np.diag(R)) + 1e-12)
            h_prev = h_new
            
        return np.max(lyap_sum / T_len)

    # Lyapunov de LSTM
    lambda_max_lstm_v4 = calcular_lyapunov_celda(celda_lstm, 'LSTM', X_test, unidades_lstm)

    # 2. Entrenar y evaluar un Baseline Tonto (SimpleRNN) en silencio para comparar
    print("Calculando Baseline RNN Simple para comparación...")
    modelo_rnn = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        SimpleRNN(units=unidades_lstm, activation='tanh'),
        Dense(1)
    ])
    modelo_rnn.compile(optimizer='adam', loss='mse')
    modelo_rnn.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0) # Solo 10 epocas para probar falla
    
    celda_rnn = [l for l in modelo_rnn.layers if isinstance(l, SimpleRNN)][0].cell
    lambda_max_rnn_base = calcular_lyapunov_celda(celda_rnn, 'RNN', X_test, unidades_lstm)
    # =================================================================
    # FIN DIAGNÓSTICO LYAPUNOV
    # =================================================================


#AGREGAR SIGUIENTE BLOQUE EN EL APPEND DE RESULTADOS
        'Lyapunov_LSTM_V4': lambda_max_lstm_v4,
        'Lyapunov_RNN_Base': lambda_max_rnn_base
