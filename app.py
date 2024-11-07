from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model import predict_emotion

app = Flask(__name__)

# Папка для загрузки изображений
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Получаем предсказание от модели
        predicted_class_name, probability = predict_emotion(file_path)

        return render_template('index.html', 
                               predicted_class=predicted_class_name, 
                               probability=probability,
                               filename=filename)

if __name__ == '__main__':
    app.run(debug=True)