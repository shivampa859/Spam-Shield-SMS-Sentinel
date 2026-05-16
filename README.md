# Spam-Shield-SMS-Sentinel

Spam-Shield-SMS-Sentinel is an end-to-end Machine Learning and Natural Language Processing (NLP) project designed to classify SMS messages as Spam or Ham (Not Spam). The project uses text preprocessing techniques, TF-IDF vectorization, and a trained machine learning model to detect spam messages with high accuracy.

---

# Project Overview

The main objective of this project is to build an intelligent SMS spam detection system capable of identifying unwanted and fraudulent messages automatically.

This project demonstrates:

- Natural Language Processing (NLP)
- Text preprocessing
- Feature extraction using TF-IDF
- Machine Learning model training
- Model serialization using Pickle
- Deployment-ready application structure

---

# Features

- Detects spam SMS messages
- Real-time text prediction
- NLP-based preprocessing pipeline
- TF-IDF vectorization
- Trained machine learning model
- Lightweight and fast prediction system
- Deployment-ready structure

---

# Tech Stack

## Programming Language
- Python

## Libraries and Frameworks
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Flask / Streamlit
- Pickle

## Machine Learning Concepts
- NLP
- Text Classification
- TF-IDF Vectorization
- Supervised Learning

---

# Project Structure

```bash
Spam-Shield-SMS-Sentinel/
│
├── app.py
├── requirements.txt
├── model.pkl
├── vectorizer.pkl
├── README.md
├── .gitignore
│
├── templates/
├── static/
│
└── dataset/
```

---

# Workflow of the Project

## 1. Data Collection

The SMS dataset contains labeled messages:
- Spam
- Ham (Not Spam)

Example:

| Message | Label |
|----------|-------|
| Congratulations! You won a lottery | Spam |
| Hey, are you coming today? | Ham |

---

## 2. Data Preprocessing

The text preprocessing pipeline includes:

- Lowercasing
- Removing punctuation
- Removing stopwords
- Tokenization
- Stemming

Example:

Input:
```text
Congratulations!!! You won $1000 prize.
```

Processed Output:
```text
congratul won prize
```

---

## 3. Feature Extraction

TF-IDF Vectorizer is used to convert text into numerical vectors.

TF-IDF helps in:
- Finding important words
- Reducing importance of common words
- Improving model performance

---

# Model Training

The machine learning model is trained on transformed TF-IDF vectors.

Possible algorithms used:
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine

Training Steps:

```text
Dataset
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization (.pkl)
```

---

# Model Serialization

After training:

- `model.pkl` stores the trained ML model
- `vectorizer.pkl` stores the TF-IDF vectorizer

These files are loaded during prediction.

---

# Application Flow

```text
User Input SMS
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Trained ML Model
        ↓
Prediction
        ↓
Spam / Ham Result
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/Spam-Shield-SMS-Sentinel.git
```

## Navigate to Project Folder

```bash
cd Spam-Shield-SMS-Sentinel
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
python app.py
```

---

# Example Prediction

## Input

```text
Congratulations! You have won a free vacation ticket.
```

## Output

```text
Spam
```

---

# Machine Learning Pipeline

```text
Raw SMS Text
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
Stemming
      ↓
TF-IDF Vectorization
      ↓
ML Model Prediction
      ↓
Spam / Ham Classification
```

---

# Advantages of the Project

- Fast prediction speed
- Lightweight deployment
- High accuracy for spam detection
- Real-world NLP implementation
- Easy to integrate into messaging systems

---

# Future Improvements

Possible future enhancements:

- Deep Learning integration
- LSTM / RNN models
- Transformer-based NLP models
- Real-time API deployment
- Mobile application integration
- Multi-language spam detection

---

# Deployment

This project can be deployed using:

- Render
- Railway
- Heroku
- Streamlit Cloud
- AWS
- Azure

---

# Learning Outcomes

Through this project, the following concepts are implemented and understood:

- NLP preprocessing techniques
- Text vectorization
- Feature engineering
- ML model training
- Model evaluation
- Pickle serialization
- Web application deployment

---

# Author

Shivam Patel

---

# License

This project is created for educational and learning purposes.
