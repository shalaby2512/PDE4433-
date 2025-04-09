# 🗑️ Design and Implementation of an AI-Powered Autonomous Waste Sorting Robot Using Deep Learning

A web application built with Flask that classifies uploaded waste images into nine distinct categories using a fine-tuned **EfficientNet-B0** model. It also provides an interactive dashboard for data visualization of classified waste based on timestamps.

Git Repo: https://github.com/shalaby2512/PDE4433-

The Git Repo will include all files and Demo Video alike. 

<div align="center">
  <h3>🎯 Categories</h3>
  <code>Cardboard</code> • <code>Food Organics</code> • <code>Glass</code> • <code>Metal</code> • <code>Miscellaneous Trash</code> • <code>Paper</code> • <code>Plastic</code> • <code>Textile Trash</code> • <code>Vegetation</code>
</div>

---

## ✨ Features

- 📥 Upload multiple images at once for classification
- 🧠 Deep learning with PyTorch and EfficientNet-B0
- 📊 Interactive dashboard built with Plotly
- 🗂️ Automatically organizes uploaded images by predicted category
- 📆 Timestamp-based image saving and filtering

---

## 🛠️ Technology Stack

- **Backend**:
  - Python 3.10+
  - Flask
- **Machine Learning**:
  - PyTorch
  - EfficientNet-B0 (fine-tuned)
- **Frontend & Visualization**:
  - HTML, CSS
  - Plotly for interactive charts
  - Jinja2 templates

---

## 🧠 Model Overview

The model is built using **EfficientNet-B0** pre-trained on ImageNet and fine-tuned for 9 waste categories.

```text
EfficientNet-B0
├── Global Average Pooling
├── Fully Connected Layer (9 outputs)
└── Softmax Activation
```

- Input Size: 224x224
- Optimizer: Adam
- Loss Function: Cross Entropy
- Accuracy: ~92% on test data

---

## 📁 Project Structure

```bash
waste_classifier/
├── app.py                   # Main Flask app
├── code.ipynb               # Jupyter notebook for training
├── best_efficientnetb0.pth  # Trained model
├── static/
│   ├── plastic/
│   ├── paper/
│   └── ... (other folders auto-created)
├── templates/
│   ├── index.html
│   ├── result.html
│   └── dashboard.html
└── requirements.txt
```

---

## 🚀 Running the Project

### 🔁 Workflow Overview

1. 📓 Train the model in **Jupyter Notebook**
2. 💾 Save the model as `best_efficientnetb0.pth`or any other prefered name 
3. 🖥️ Run the Flask app using **Command Prompt**

---

### ✅ Step-by-Step Guide

#### 1. 🔬 Train the Model

- Open Jupyter Notebook
- Run the notebook file `code.ipynb`
- This will:
  - Load and prepare the dataset
  - Train the EfficientNet-B0 model
  - Save the model as `best_efficientnetb0.pth`

> ⚠️ Make sure this file is saved in the same folder as `app.py`.

---

#### 2. 🖥️ Run the Flask Web App

##### 📂 A. Navigate to Your Folder

Open Command Prompt or Terminal:

```bash
cd path\to\your\waste_classifier
```

##### 🧪 B. (Optional) Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux
```

##### 📦 C. Install Required Libraries

```bash
pip install -r requirements.txt
```

##### 🚀 D. Launch the App

```bash
python app.py
```

Then go to your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📱 How to Use

1. 🏠 On the homepage, upload one or more waste images
2. 🧠 The model predicts and categorizes them
3. 💾 Images are saved into the appropriate folders
4. 📊 Go to `/dashboard` to see data visualized by category and date range

---

## 📈 Dashboard Features

- Interactive bar charts using **Plotly**
- Selectable `start_date` and `end_date` filters
- Real-time update of category counts

---

## 📦 Waste Categories

```text
['Cardboard', 'Food Organics', 'Glass', 'Metal',
 'Miscellaneous Trash', 'Paper', 'Plastic',
 'Textile Trash', 'Vegetation']
```