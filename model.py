import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rutas de los datos
train_data_dir = 'data/train'
test_data_dir = 'data/test'

# Configuración del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preprocesamiento de datos
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Entrenar el modelo
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10
)

# Guardar el modelo
model.save('modelo_gatos_perros.h5')
