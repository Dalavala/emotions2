import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from temp import Model

# Определяем классы эмоций
class_names = ['angry', 'boredom', 'contempt', 'disgust', 'embarrassment', 
               'fear', 'happy', 'neutral', 'sad', 'surprised']

# Загружаем модель
model = load_model('my_model.keras')

# Функция для загрузки и предварительной обработки изображения
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Функция для предсказания класса эмоции
def predict_emotion(img_path):
    img_array = load_and_preprocess_image(img_path)
    preds = model.predict(img_array)
    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = class_names[predicted_class_index]
    probability = preds[0][predicted_class_index]
    return predicted_class_name, probability