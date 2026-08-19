import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import CNN
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import base64
import io


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu')))
model.eval()

def prediction(image_path, plant_type="Auto-Detect"):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.transpose(image_array, (2, 0, 1))
    input_data = torch.tensor(image_array, dtype=torch.float32) / 255.0
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()[0]
    
    if plant_type != "Auto-Detect":
        valid_indices = []
        for i, name in enumerate(disease_info['disease_name']):
            if name.startswith(plant_type):
                valid_indices.append(i)
        
        if valid_indices:
            masked_output = np.full_like(output, -np.inf)
            masked_output[valid_indices] = output[valid_indices]
            index = np.argmax(masked_output)
            return index

    index = np.argmax(output)
    return index

def prediction_top3(image, plant_type="Auto-Detect"):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.transpose(image_array, (2, 0, 1))
    input_data = torch.tensor(image_array, dtype=torch.float32) / 255.0
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    
    if plant_type != "Auto-Detect":
        valid_indices = []
        for i, name in enumerate(disease_info['disease_name']):
            if name.startswith(plant_type):
                valid_indices.append(i)
        
        if valid_indices:
            mask = torch.full_like(output[0], -float('inf'))
            mask[valid_indices] = output[0][valid_indices]
            probabilities = F.softmax(mask, dim=0)
            k = min(3, len(valid_indices))
            topk_prob, topk_catid = torch.topk(probabilities, k)
            return topk_prob.detach().numpy(), topk_catid.detach().numpy()

    probabilities = F.softmax(output[0], dim=0)
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    return top3_prob.detach().numpy(), top3_catid.detach().numpy()


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        plant_type = request.form.get('plant_type', 'Auto-Detect')
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path, plant_type)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = f"/static/uploads/{filename}" , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    image_data = data['image_data']
    plant_type = data.get('plant_type', 'Auto-Detect')
    
    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        top3_prob, top3_catid = prediction_top3(image, plant_type)
        
        results = []
        for i in range(len(top3_catid)):
            idx = int(top3_catid[i])
            prob = float(top3_prob[i]) * 100
            
            disease_name = str(disease_info['disease_name'][idx])
            prevent = str(disease_info['Possible Steps'][idx])
            supp_name = str(supplement_info['supplement name'][idx])
            
            results.append({
                'disease_name': disease_name,
                'probability': round(prob, 2),
                'prevent': prevent,
                'supplement_name': supp_name if supp_name != 'nan' else 'None'
            })
            
        return jsonify({'success': True, 'predictions': results})
    except Exception as e:
        print("Error processing frame:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
