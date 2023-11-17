from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)

# Load your pre-trained model
model = load_model('food_classification_model.h5')

label_index_dict = {
    'Baked Potato': 0,
    'Crispy Chicken': 1,
    'Donut': 2,
    'Fries': 3,
    'Hot Dog': 4,
    'Sandwich': 5,
    'Taco': 6,
    'Taquito': 7,
    'apple_pie': 8,
    'burger': 9,
    'butter_naan': 10,
    'chai': 11,
    'chapati': 12,
    'cheesecake': 13,
    'chicken_curry': 14,
    'chole_bhature': 15,
    'dal_makhani': 16,
    'dhokla': 17,
    'fried_rice': 18,
    'ice_cream': 19,
    'idli': 20,
    'jalebi': 21,
    'kaathi_rolls': 22,
    'kadai_paneer': 23,
    'kulfi': 24,
    'masala_dosa': 25,
    'momos': 26,
    'omelette': 27,
    'paani_puri': 28,
    'pakode': 29,
    'pav_bhaji': 30,
    'pizza': 31,
    'samosa': 32,
    'sushi': 33
}

# Function to return the index of the image label
def find_max(preds):
    return np.argmax(preds)

# Function to preprocess the user-provided image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(351, 351))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_label(pred_index):
    index_label_dict = {v: k for k, v in label_index_dict.items()}
    label = index_label_dict.get(pred_index, "Label not found")
    return label

# Function to perform classification on the user-provided image
def classify_image(img_path):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    decoded_prediction_index = find_max(predictions) 
    prediction_label = get_label(decoded_prediction_index)
    return str(prediction_label)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            img = file
            img_path = "static/" + img.filename	
            img.save(img_path)
            result = classify_image(img_path)
        else:
            result = "Invalid file format. Please upload an image."

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)