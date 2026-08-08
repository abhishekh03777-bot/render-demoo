# 🏏 IPL Team Win Prediction

A Machine Learning web application that predicts the **winning probability of an IPL team** based on real-time match conditions such as batting team, bowling team, current score, overs, wickets, and target.

## 🚀 Project Overview

This project uses Machine Learning to estimate the probability of a team winning an IPL match during the second innings.

The model takes important match features as input and predicts:

* 🏆 Winning probability
* 📉 Losing probability

The project is deployed as a web application using **Flask**, allowing users to enter match details and get predictions through an interactive interface.

## ✨ Features

* Predicts IPL match winning probability
* Interactive web interface
* Takes current match conditions as input
* Displays winning and losing probabilities
* Machine Learning pipeline for preprocessing and prediction
* Flask-based backend
* Simple and user-friendly UI

## 🧠 Machine Learning

The model is trained using IPL match data and uses features such as:

* Batting Team
* Bowling Team
* City
* Target Score
* Current Score
* Overs Completed
* Wickets Lost

From these features, the model calculates the current match situation and predicts the probability of winning.

## 🛠️ Tech Stack

**Programming Language**

* Python

**Machine Learning**

* Scikit-learn
* Pandas
* NumPy


**Tools**

* Jupyter Notebook
* VS Code
* Git & GitHub

## 📂 Project Structure

```text
IPL-Win-Prediction/
│
├── app.py
├── model.pkl
├── pipe.pkl
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── notebooks/
│   └── model_training.ipynb
└── README.md
```

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone <your-repository-link>
cd IPL-Win-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

### 6. Open in browser

```text
http://127.0.0.1:5000/
```

## 📊 Prediction Workflow

```text
User Input
    ↓
Data Preprocessing
    ↓
Feature Transformation
    ↓
Trained ML Model
    ↓
Win Probability
    ↓
Loss Probability
```

## 🎯 Example

The user provides the current match situation:

```text
Batting Team: Chennai Super Kings
Bowling Team: Mumbai Indians
Target: 180
Current Score: 120
Overs: 15
Wickets Lost: 4
```

The application processes these features and returns the predicted winning and losing probabilities.

## 📌 Future Improvements

* Add more recent IPL data
* Improve model accuracy
* Add multiple ML models and compare performance
* Add live IPL match data using an API
* Improve UI/UX
* Deploy the application publicly
* Add prediction history and match analytics

## 👨‍💻 Author

**Abhishek Kumar**

B.Tech Artificial Intelligence Student

Interested in:

* Machine Learning
* Deep Learning
* NLP
* LLMs
* Computer Vision

---

⭐ If you find this project useful, consider giving the repository a star!
