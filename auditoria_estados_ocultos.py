#OPCION 1: EXTERNO A LA CONFIRGURACIÓN DE LA LSTM#
#--------------------------------------------------------------------------------#
from tensorflow.keras.models import Model

# --- 1. EXTRACCIÓN DE LA MATRIZ DE ACTIVACIONES ---
# Crear un modelo intermedio que escupa los estados ocultos en lugar de la predicción final
modelo_oculto = Model(inputs=modelo_lstm.input, 
                      outputs=modelo_lstm.get_layer('lstm_1').output)

# X_test_canal tensor de prueba preparado para la red (forma: Muestras, Pasos, Features)
# Predict genera la matriz h_t
activaciones = modelo_oculto.predict(X_test_canal)

# Array 2D para analizar el paso t
matriz_ht = activaciones.reshape(-1, activaciones.shape[-1]) # Forma: (T_total, Neuronas)

# --- 2. CÁLCULO DE ENTROPÍA DE SHANNON POR PASO TEMPORAL ---
bins_cuantiles = 20
entropias_ht = []

for t in range(matriz_ht.shape[0]):
    vector_activacion_t = matriz_ht[t, :]
    
    # Discretizar en B=20 bins empíricos
    histograma, _ = np.histogram(vector_activacion_t, bins=bins_cuantiles, density=False)
    
    # Calcular probabilidades empíricas (p)
    probabilidades = histograma / np.sum(histograma)
    
    # Calcular Entropía de Shannon H(h_t) para este instante
    h_t_entropia = entropy(probabilidades, base=2)
    entropias_ht.append(h_t_entropia)

entropias_ht = np.array(entropias_ht)

# --- 3. RESUMEN ESTADÍSTICO (Reemplazo de la gráfica de trayectoria, se recomienda usar plot si es posible) ---
media_H = np.mean(entropias_ht)
var_H = np.var(entropias_ht)
cv_H = np.std(entropias_ht) / media_H if media_H != 0 else 0

print("\n=== DINÁMICA DE ESTADOS OCULTOS H(h_t) ===")
print(f"Canal analizado: [Nombre del Canal]")
print(f"Media de Entropía Interna (mu): {media_H:.4f}")
print(f"Varianza de Entropía Interna (sigma^2): {var_H:.4f}")
print(f"Coeficiente de Variación (CV): {cv_H:.4f}")



#OPCIÓN 2: INTEGRADO A LA LSTM (ESTA VERSIÓN FUE LA ELEGIDA EN EL ESTUDIO)#
#SE DEBE INTEGRAR COMO PARTE DEL APPEND DE RESULTADOS EN LA LSTM#
#----------------------------------------------------------------------------------------#
        # =================================================================
        # INICIO AUDITORÍA DE ENTROPÍA DE ESTADOS OCULTOS (H_t)
        # =================================================================
        nombre_capa_lstm = None
        for layer in model.layers:
            if 'lstm' in layer.name.lower():
                nombre_capa_lstm = layer.name
                break
        
        modelo_oculto = Model(inputs=model.inputs, outputs=model.get_layer(nombre_capa_lstm).output)
        activaciones = modelo_oculto.predict(X_test, verbose=0) # verbose=0 para no ensuciar la consola
        
        # Extraer matriz (tomando el último paso de la secuencia)
        if len(activaciones.shape) == 3:
            matriz_ht = activaciones[:, -1, :] 
        else:
            matriz_ht = activaciones 
            
        T_muestras, n_neuronas = matriz_ht.shape
        B_bins = 20
        matriz_ht_discreta = np.zeros_like(matriz_ht)
        
        # Discretización vertical con protección contra neuronas muertas (ReLU)
        for j in range(n_neuronas):
            serie_neurona = matriz_ht[:, j]
            try:
                bins_cuantiles = pd.qcut(serie_neurona, q=B_bins, labels=False, duplicates='drop')
                matriz_ht_discreta[:, j] = bins_cuantiles
            except ValueError:
                # Si la neurona murió (varianza 0), la asignamos al bin 0 para no frenar el código
                matriz_ht_discreta[:, j] = 0
                
        # Cálculo horizontal
        entropias_ht = []
        for t in range(T_muestras):
            estado_t = matriz_ht_discreta[t, :]
            conteos = np.bincount(estado_t.astype(int), minlength=B_bins)
            probabilidades = conteos / n_neuronas
            h_t_entropia = entropy(probabilidades, base=2)
            entropias_ht.append(h_t_entropia)
            
        entropias_ht = np.array(entropias_ht)
        media_H = np.mean(entropias_ht)
        var_H = np.var(entropias_ht)
        # =================================================================
        # FIN AUDITORÍA DE ENTROPÍA
        # =================================================================

