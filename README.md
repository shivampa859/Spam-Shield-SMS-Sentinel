# Spam Shield SMS Sentinel

Spam Shield SMS Sentinel is a machine learning-based system designed to detect and filter **spam SMS messages** from genuine ones.  
It uses text preprocessing, feature extraction, and classification algorithms to protect users from unwanted or malicious SMS.

---

## Features
- Detects **spam** vs **ham** SMS messages.
- Uses **Natural Language Processing (NLP)** techniques.
- Machine learning model for classification.
- Easy to integrate with mobile/desktop applications.
- Lightweight and fast for real-time detection.

---

## 🛠️ Tech Stack
- **Python 3.8+**
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Scikit-learn** – ML models
- **NLTK / spaCy** – NLP preprocessing
- **Matplotlib / Seaborn** – Visualization

---

## 📊 Dataset
- **Name:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** ~5,572 SMS messages
- **Labels:**  
  - `spam` – Unwanted promotional, phishing, or fraudulent messages  
  - `ham` – Genuine messages

---

## ⚙️ How It Works
1. **Data Cleaning:** Remove stopwords, punctuation, special characters.
2. **Text Vectorization:** Convert text to numerical format using **TF-IDF** or **Bag of Words**.
3. **Model Training:** Train a **Naive Bayes** or **Logistic Regression** classifier.
4. **Prediction:** Classify new SMS as spam or ham.
5. **Evaluation:** Check accuracy, precision, recall, F1-score.

---

## 📈 Model Performance
| Metric       | Score |
|--------------|-------|
| Accuracy     | 98%   |
| Precision    | 97%   |
| Recall       | 96%   |
| F1-Score     | 96%   |

---

## ▶️ Installation & Usage
```bash
# Clone the repository
git clone https://github.com/YourUsername/Spam-Shield-SMS-Sentinel.git
cd Spam-Shield-SMS-Sentinel

# Install dependencies
pip install -r requirements.txt

# Run training
python src/model.py

# Predict a sample message
python src/predict.py "Congratulations! You've won a free ticket."
