import streamlit as st
import joblib
import numpy as np
# Título de la aplicación
st.title("Predictor")
st.header("Predecir el precio de alquiler ")
# Instrucciones para el usuario
st.write("Ingrese las características ")
# Front

metros_cuadrados = float(st.text_input("metros cuadrados", value="16"))
n_baños= float(st.text_input("Número de baños", value="1"))
# Solicita al usuario que elija entre "Sí" y "No"
tiene_ascensor = st.selectbox('¿Tiene ascensor? :',("Sí", "No"))
# Asigna valores enteros a cada opción
if tiene_ascensor== "Sí":
   valor = 1
else:
   valor =0
tiene_ascensor = valor
n_habitaciones= int(st.text_input("Número de habitaciones", value="1"))
# Solicita al usuario que elija entre "Sí" y "No"
tiene_estacionamiento = st.selectbox('¿Tiene ascensor? :',("Sí", "No"))
# Asigna valores enteros a cada opción
if tiene_estacionamiento== "Sí":
   valor =1
else:
   valor =0
tiene_estacionamiento= valor
 # Backend
# Creamos el array de entrada
X_list = [ metros_cuadrados,n_baños , tiene_ascensor, n_habitaciones, tiene_estacionamiento ]
X = np.array(X_list, dtype=np.float64)
X = X.reshape(1,-1)
# Botón para ejecutar el modelo
if st.button("Predecir"):
 if len(X) > 0:
 # Cargar el modelo y los parámetros de normalización guardados
    scaler= joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
    scaler_y= joblib.load('scaler_y.joblib')
 # Mostrar las primeras filas del DataFrame cargado
    X_scaled = scaler.transform(X)
 # Realizar predicciones con el modelo
    predicciones = model.predict(X_scaled)
    predicciones= scaler_y.inverse_transform(predicciones)
 # Mostrar las predicciones
    st.write("Predicciones del precio de renta:")
    st.write(predicciones)            