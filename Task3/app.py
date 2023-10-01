import os
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import base64
from werkzeug.utils import secure_filename


print(os.getcwd())


app = Flask(__name__)
app._static_folder = os.path.abspath("templates")

# Load the existing model
loaded_model = joblib.load('C:/Users/admin/Desktop/Facial-Recognition-System/face_recognizer.pkl')

# Path to save uploaded images
UPLOAD_FOLDER = 'C:/Users/admin/Desktop/Facial-Recognition-System/data/yalefaces_resized'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_face_data(face_data):
    # Decode base64 image data and convert it to a numpy array
    image_bytes = base64.b64decode(face_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Assuming the image is in color

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to match the input size expected by the model
    target_size = (100, 100)  # Specify the target width and height
    resized_img = cv2.resize(gray_img, target_size)

    # Normalize the pixel values to be between 0 and 1
    normalized_img = resized_img.astype('float32') / 255.0

    # Flatten the image into a 1D array
    flattened_img = normalized_img.flatten()

    # Return the preprocessed image data
    return flattened_img.reshape(1, -1)  # Reshape to match the input shape expected by the model


# Function to update the model with a new employee's face data
def update_model_with_new_employee(face_data, label):
    # Preprocess face_data (resize, normalize, etc.) similar to how you did during training
    processed_face_data = preprocess_face_data(face_data)

    # Load existing face embeddings and labels
    face_embeddings = np.load('face_embeddings.npy')
    face_labels = np.load('face_labels.npy')

    # Append new face embedding and label
    face_embeddings = np.append(face_embeddings, processed_face_data, axis=0)
    face_labels = np.append(face_labels, label)  # You can use name, email, or any unique identifier as the label

    # Update the label encoder with the new labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(face_labels)

    # Fine-tune the existing model with the new data
    loaded_model.fit(face_embeddings, encoded_labels)
    
    # Save the updated model, face embeddings, and label encoder
    joblib.dump(loaded_model, 'face_recognizer.pkl')
    np.save('face_embeddings.npy', face_embeddings)
    np.save('face_labels.npy', face_labels)

    
import sqlite3
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''
          CREATE TABLE IF NOT EXISTS users 
          (id INTEGER PRIMARY KEY AUTOINCREMENT, 
          name TEXT NOT NULL, 
          email TEXT NOT NULL, 
          face_data TEXT NOT NULL)
          ''')
conn.commit()
conn.close()

@app.route('/register', methods=['GET'])
def render_registration_page():
    return render_template('registration.html')

# Endpoint for registering a new user
@app.route('/register', methods=['POST'])
def register_user():
    name = request.form['name']
    email = request.form['email']
    
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'message': 'No file part'})
    
    file = request.files['image']
    #print('Received Image Data:', file)
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Read the image file and preprocess it
        image_bytes = file.read()
        print('Received Image Data:', image_bytes)
        face_data = base64.b64encode(image_bytes).decode('utf-8')  # Convert image to base64
        
        # Perform registration logic here (e.g., save user info to database)
        # Store user information in the database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (name, email, face_data) VALUES (?, ?, ?)', (name, email, face_data))
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'User registered successfully!'})

# Render face recognition page
@app.route('/recognize', methods=['GET'])
def recognize():
    return render_template('recognition.html')

# Endpoint for updating the model with new employee data
@app.route('/update_model', methods=['POST'])
def update_model():
    print("Received update_model request")
    try:
        data = request.get_json()
        base64_image = data['image']  # Get the base64 image data from the request
        #print('Base64 Image:', base64_image)
        if not base64_image:
            return jsonify({'error': 'Empty or invalid image data'}), 400
        # Convert the base64 image data to a numpy array
        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Assuming the image is in color

        if img is not None:
            # Preprocess face_data (resize, normalize, etc.) similar to how you did during training
            processed_face_data = preprocess_face_data(img)

            # Load existing face embeddings and labels
            face_embeddings = np.load('face_embeddings.npy')
            face_labels = np.load('face_labels.npy')

            # Append new face embedding and label
            face_embeddings = np.append(face_embeddings, processed_face_data, axis=0)
            face_labels = np.append(face_labels, data['name'])  # You can use name, email, or any unique identifier as the label

            # Update the label encoder with the new label (name, email, etc.)
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(face_labels)

            # Fine-tune the existing model with the new data
            loaded_model.fit(face_embeddings, encoded_labels)

            # Save the updated model, face embeddings, and label encoder
            joblib.dump(loaded_model, 'face_recognizer.pkl')
            np.save('face_embeddings.npy', face_embeddings)
            np.save('face_labels.npy', face_labels)

            return jsonify({'message': 'Model updated successfully!'})
        else:
            return jsonify({'error': 'Invalid or empty image data'}), 400

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


# Endpoint for recognizing a face
@app.route('/recognize', methods=['POST'])
def recognize_face():
    # Check if the request contains a file named 'image'
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    data = request.get_json()
    face_data = data['image']  # Face data captured from the user

    # Preprocess face_data similar to how you did during training
    processed_face_data = preprocess_face_data(face_data)

    # Load face embeddings and label encoder
    face_embeddings = np.load('face_embeddings.npy')
    label_encoder = LabelEncoder()
    face_labels = np.load('face_labels.npy')
    encoded_labels = label_encoder.fit_transform(face_labels)

    # Use the loaded model to predict the label
    predicted_label = loaded_model.predict(processed_face_data)[0]

    # Get the corresponding name from the label encoder
    predicted_name = label_encoder.classes_[predicted_label]
    
    return jsonify({'predicted_name': predicted_name})



if __name__ == '__main__':
    app.run(debug=True)
