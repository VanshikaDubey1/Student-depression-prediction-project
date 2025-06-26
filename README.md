🎓 Student Depression Prediction ProjectAdd commentMore actions

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning web application that predicts the risk of depression in students using behavioral and academic data. Built with a Flask backend and HTML frontend.

---

## 📌 Features

- 🧠 Predicts *student depression risk* (Low / High)
- ✅ Uses a *Random Forest Classifier*
- 🌐 Frontend hosted on *GitHub Pages*
- 📦 Python-powered backend using *Flask + Scikit-learn*
- 💾 Local .csv dataset and pre-trained model support

---

## 📁 Project Structure

```bash
Student-depression/
├── app.py                        # Flask backend
├── docs/
│   └── index.html                # Frontend UI (hosted via GitHub Pages)
├── student_depression_dataset.csv   # Dataset (local only)
├── feature_columns.pkl           # Stored feature columns
├── label_encoders.pkl            # Encoders for categorical data
├── depression_model.pkl          # Trained ML model
├── Student-depression.ipynb      # Notebook for model training
├── .venv/                        # Virtual environment (optional)
└── README.md                     # Project documentation


---

🚀 Getting Started

🔧 Clone and Set Up

git clone https://github.com/VanshikaDubey1/Student-depression-prediction-project.git
cd Student-depression-prediction-project

📦 Create and Activate Virtual Environment

python -m venv .venv
.\.venv\Scripts\activate

📥 Install Required Packages

pip install flask scikit-learn pandas joblib


---

▶ Run the Application

python app.py

📍 Open in browser: http://127.0.0.1:5000


---



<h3>🧠 Student Depression Illustration</h3>
<img src="student depression.jpg.png" alt="Student Depression" width="600"/>





🌐 Live Frontend Demo

✨ Hosted with GitHub Pages
🔗 View Live Demo

> Note: Backend (Flask) must run locally or be deployed separately.




---

🧠 About the Model

Algorithm: Random Forest Classifier

Input: Student behavior & academic-related features

Output: Depression risk classification

Trained using scikit-learn with preprocessing and label encoding



---

🗃 Dataset Info

Ensure that the file student_depression_dataset.csv is present in the root folder.
The dataset should contain features like attendance, GPA, screen time, sleep patterns, etc.


---

✨ Future Enhancements

🌍 Host backend on Render or Railway

🔐 Add user authentication

📈 Improve prediction accuracy with more features

🎨 Redesign UI with Bootstrap or React

