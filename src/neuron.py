import numpy as np

class Neuron:
  def __init__(self, weights, bias, func): # Inicializamos la clase
    self.weights = weights
    self.bias = bias
    self.func = func

  @staticmethod # F.Act Sigmoide (método estático)
  def __sigmoid(x): # Método privado
    return 1 / (1 + np.exp(-x))

  @staticmethod # F.Act ReLU (método estático)
  def __relu(x): # Método privado
    return np.maximum(0, x)

  @staticmethod # F.Act Tangente Hiperbólica (método estático)
  def __tanh(x): # Método privado
    return np.tanh(x)

  def change_bias(self, new_bias): # Función para cambiar el valor del sesgo
    self.bias = new_bias

  def change_weights(self, new_weights): # Función para cambiar el valor de los pesos
    self.weights = new_weights

  def run(self, x):
    if len(self.weights) != len(x): # La cantidad para el array de valores de entrada y para los pesos ha de ser la misma.
      print('Los valores de entrada y los pesos son de distinto tamaño')
    else:
      neuron_operation = sum(np.multiply(self.weights, x)) + self.bias # Operación para la neurona
      activation_function = getattr(self, f"_Neuron__{self.func}")  # Guarda la función de activación
      y = activation_function(neuron_operation) # Llama a la función de activación y realiza la operación
      return y