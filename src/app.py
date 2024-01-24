import streamlit as st
from neuron import Neuron

st.set_page_config(layout="wide")

st.image('src/neurona.jpg', width=150)

st.header('Simulador de neurona')

numxw = st.slider('Elige el número de entradas/pesos que tendrá la neurona',1,10)

st.markdown('### Pesos')
col1 = st.columns(numxw)

w = []
for i in range(numxw):
   with col1[i]:
      w.append(st.number_input(f'w{i}'))

st.text(f'w={w}')

st.markdown('### Entradas')

col2 = st.columns(numxw)

x = []
for i in range(numxw):
   with col2[i]:
      x.append(st.number_input(f'x{i}'))

st.text(f'x={x}')

subcol1, subcol2 = st.columns(2)
with subcol1:
   st.subheader('Sesgo')
   b = st.number_input('Introduce el valor del sesgo', value=100)

with subcol2:
   st.subheader('Función de activación')
   funciones_activacion = {'Sigmoide': 'sigmoid', 'ReLU': 'relu', 'Tangente hiperbólica': 'tanh'}
   funcion = st.selectbox('Elige la función de activación', list(funciones_activacion.keys()))

if st.button('Calcular la salida', type='primary'):
   neurona = Neuron(weights=w, bias=b, func=funciones_activacion[funcion])
   output = neurona.run(x)
   st.write('La salida es', output)
