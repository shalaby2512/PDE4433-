from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import models, transforms
from PIL import Image
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import re

app = Flask(__name__)
pyo.init_notebook_mode()

# Load the pre-trained model which was done in the notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 9)
model.load_state_dict(torch.load('best_efficientnetb0.pth', map_location=device))
model = model.to(device)
model.eval()

class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

# Transform and start to preprocess uploaded images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to extract timestamp from filename
def extract_timestamp(filename):
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    return None

# This is for the Home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle multiple image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return 'No files uploaded', 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return 'No selected files', 400

    # Dictionary to store counts for each category
    waste_counts = defaultdict(int)

    # Process each uploaded image
    for file in files:
        try:
            # Preprocess the uploaded image
            image = Image.open(file).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                predicted_label = class_names[preds.item()]

            # Increment the count for the predicted category
            waste_counts[predicted_label] += 1

            # Create a directory based on the predicted label which is to be stored in the static folder
            category_folder = os.path.join('static', predicted_label.lower())
            os.makedirs(category_folder, exist_ok=True)

            # Create a timestamped filename to avoid overwriting
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            img_path = os.path.join(category_folder, filename)
            
            # Save the image as a file
            file.seek(0) 
            with open(img_path, 'wb') as f:
                f.write(file.read())

        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            continue

    # Render the result page with the waste counts
    return render_template('result.html', waste_counts=dict(waste_counts))

# This is the Dashboard route
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    static_dir = 'static'
    
    # Get all categories
    categories = [d for d in os.listdir(static_dir) if os.path.isdir(os.path.join(static_dir, d))]
    
    # Collect data for all files
    data = []
    for category in categories:
        category_path = os.path.join(static_dir, category)
        for filename in os.listdir(category_path):
            if os.path.isfile(os.path.join(category_path, filename)):
                timestamp = extract_timestamp(filename)
                data.append({'Category': category, 'Filename': filename, 'Timestamp': timestamp})
    
    df = pd.DataFrame(data)
    
    # Handle form submission
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    if start_date and end_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    
    # Calculate category counts
    category_counts = df['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    # Create interactive bar chart using Plotly
    fig = px.bar(category_counts, x='Category', y='Count',
                 title='Waste Classification Distribution',
                 labels={'Category': 'Waste Category', 'Count': 'Number of Items'},
                 color='Category',  
                 hover_data=['Count'],  
                 text='Count')
    
    fig.update_layout(
        xaxis_title="Waste Category",
        yaxis_title="Number of Items",
        title_x=0.5,
        template="plotly_white"
    )
    
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    
    plot_html = pyo.plot(fig, output_type='div')
    
    return render_template('dashboard.html', plot_html=plot_html)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
