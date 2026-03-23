# 🫁 Lung Cancer Risk Assessment System

A Machine Learning-based web application that predicts the likelihood of lung cancer using patient demographics and symptom-based inputs. The system provides real-time predictions through an interactive and visually enhanced Streamlit interface.

---

## 📌 Project Overview

This project focuses on early detection of lung cancer risk by analyzing patient data such as age, gender, and various health-related symptoms. A trained Support Vector Machine (SVM) model is used to classify whether a patient is at high or low risk, helping in early awareness and decision-making.

---

## 🚀 Features

* Predicts lung cancer risk (High / Low)
* Interactive and modern UI built with Streamlit
* Real-time prediction with confidence scores
* Uses patient symptoms and demographic data
* Clean and user-friendly interface with custom styling

---

## 🧠 Machine Learning Model Used

* Support Vector Machine (SVM)

---

## 📊 Input Features

* Age
* Gender
* Smoking
* Yellow Fingers
* Anxiety
* Peer Pressure
* Chronic Disease
* Fatigue
* Allergy
* Wheezing
* Alcohol Consumption
* Coughing
* Shortness of Breath
* Swallowing Difficulty
* Chest Pain

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## ⚙️ Project Structure

```
├── app.py
├── svm_lung_cancer_model.joblib
├── age_scaler.joblib
├── Project_lungcancer.ipynb
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/SATYA012904/lung-cancer-prediction.git
cd lung-cancer-prediction
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
streamlit run app.py
```

---

## 📈 Model Details

* Model: Support Vector Machine (SVM)
* Data preprocessing includes:

  * Feature encoding
  * Scaling using Standard Scaler
* Confidence scores generated using:

  * Probability (if available)
  * Sigmoid function (fallback method)

---

## 💡 Key Highlights

* Strong use of symptom-based feature engineering
* Clean preprocessing pipeline for model input
* Modern UI with custom CSS styling
* Real-time prediction with confidence metrics
* Useful for educational and healthcare awareness purposes

---

## 📌 Future Improvements

* Add more advanced models (Random Forest, Deep Learning)
* Improve dataset size and quality
* Deploy on cloud platforms
* Add feature importance visualization
* Integrate real-world medical datasets

---

## ⚠️ Disclaimer

This application is developed for educational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment.

---

## 👨‍💻 Author

Satyabrata Sahu
B.Tech Computer Science Student

---

## 📜 License

This project is for educational purposes only.
