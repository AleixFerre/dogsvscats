import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Cargar el modelo entrenado
loaded_model = tf.keras.models.load_model('modelo_gatos_perros.h5')

# Hacer predicciones en una nueva imagen
img_path = 'data/train/cat/1220.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = loaded_model.predict(img_array)
print("Predicción:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Es un perro.")
else:
    print("Es un gato.")
