from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import io

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo
model = load_model('Reconocimiento_Señales_Transito.h5')

# Diccionario para las etiquetas
labels = {
    0: 'Parada De Bus',
    1: 'No Pase',
    2: 'No Parqueo',
    3: 'Pare',
    4: 'No Girar a la Izquierda',
    5: 'Prohibido Parquear',
    6: 'Ceda El Paso',
    7: 'Prohibido Girar Derecha',
    8: 'Prohibido Girar Izquierda',
    9: 'Detección Electrónica',
    10: 'Prohibido Dejar Pasajeros',
    11: 'Velocidad Máxima',
    12: 'Maltrato Animal',
    13: 'Arroyo',
    14: 'Tráfico Bicicletas',
    15: 'Zona De Peatones',
    16: 'Reductor De Velocidad',
    17: 'Zona Escolar'
}

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen del formulario
    file = request.files['image']
    
    # Leer la imagen en memoria y convertirla a un formato que el modelo pueda usar
    image = Image.open(io.BytesIO(file.read()))
    image = image.resize((300, 300))
    image_array = np.array(image) / 255.0  # Normalizar la imagen
    image_array = np.expand_dims(image_array, axis=0)  # Agregar la dimensión del lote

    # Realizar la predicción
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    predicted_label = labels[predicted_class]

    return jsonify({'label': predicted_label})

# Iniciar la aplicación Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
