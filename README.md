# 🏠 End-to-End House Price Prediction

**Developed by: Moghal Tanveer**  
**Technology: Python | Machine Learning | XGBoost | Flask**

## 📌 Project Overview

The **House Price Prediction** project is an end-to-end Machine Learning application that predicts house prices based on property features such as bedrooms, bathrooms, living area, location, and other housing details.

The project uses an **XGBoost Regression model** to learn patterns from historical housing data. A Flask-based web application provides a user-friendly interface where users can enter house details and receive an estimated price.

## 🎯 Objectives

- Analyze and preprocess housing datasets.
- Handle missing numerical and categorical values.
- Train an XGBoost regression model.
- Evaluate model performance using MAE, RMSE, and R² score.
- Save the trained Machine Learning pipeline.
- Develop a web interface using Flask.
- Predict house prices using user-provided information.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Saving:** Joblib
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **IDE:** PyCharm

## 📂 Project Structure

```text
HOUSE_PRICE_PREDICTION/
│
├── models/
│   └── house_price_model.pkl
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│
├── templates/
│   └── index.html
│
├── app.py
├── main.py
├── data.csv
├── requirements.txt
└── README.md
```

## ⚙️ Machine Learning Workflow

1. Load the housing dataset.
2. Remove unnecessary columns such as `date` and `street`.
3. Clean and preprocess the data.
4. Separate numerical and categorical features.
5. Split the dataset into training and testing sets.
6. Train the XGBoost regression model.
7. Evaluate model performance.
8. Save the complete trained pipeline.
9. Use Flask to connect the model with the website.
10. Predict house prices through the web interface.

## 📊 Model Performance

The model was evaluated using the following metrics:

| Evaluation Metric | Result |
|---|---:|
| MAE | $94,751.58 |
| RMSE | $197,159.29 |
| R² Score | 0.7387 |
| R² Percentage | 73.87% |

*Note: These are the results from the current training run. R² percentage is not classification accuracy.*

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/moghaltanveer90-bot/HOUSE_PRICE_PREDICTION.git
```

### 2. Open the Project Folder

```bash
cd HOUSE_PRICE_PREDICTION
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Model

```bash
python main.py
```

### 5. Run the Flask Application

```bash
python app.py
```

### 6. Open the Website

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## 💡 Application Features

- User-friendly web interface.
- House details input form.
- Machine Learning-based price prediction.
- Real-time prediction through Flask.
- Saved model pipeline for reuse.

## 👨‍💻 Developer

**Moghal Tanveer**

- Email: moghaltanveer90@gmail.com
- Mobile: 7680014278

---

⭐ If you find this project useful, feel free to explore the repository and share your feedback.

**Developed with Python and Machine Learning.**
