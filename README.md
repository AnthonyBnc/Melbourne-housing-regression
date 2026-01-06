# Melbourne Housing Price Prediction (Regression & Deployment)

This project implements a complete machine learning workflow to predict residential house prices using a subset of the Melbourne Housing dataset.  
It covers data preprocessing, exploratory data analysis, regression modelling, feature importance analysis, and deployment of a live web demo.

---

## 🔗 Live Demo (Hosted Application)

The trained model has been deployed as an interactive web application using **Gradio** and **Hugging Face Spaces**.

👉 **Live URL:**  
https://huggingface.co/spaces/AnthonyBnc/melbourne-house-price-predictor

Users can input property features and receive real-time house price predictions.

---

## 📊 Dataset

- **Source:** Melbourne Housing Dataset (Kaggle)
- **Subset:** ~150 records filtered to three suburbs
- **Target Variable:** `Price` (AUD)
- **Feature Types:**
  - Numerical (e.g., Rooms, Distance, Landsize)
  - Categorical (e.g., Suburb, Property Type)

---

## 🧠 Machine Learning Workflow

### 1. Dataset Inspection
- Checked dataset shape, data types, summary statistics, and missing values
- Identified numerical and categorical features

### 2. Data Cleaning & Preprocessing
- Removed rows with missing target values
- Applied:
  - Median imputation and standard scaling for numerical features
  - Most-frequent imputation and one-hot encoding for categorical features
- Implemented using `ColumnTransformer` and `Pipeline` for reproducibility

### 3. Exploratory Data Analysis (EDA)
- Price distribution analysis
- Price comparison across suburbs
- Correlation analysis for numerical features

### 4. Model Development
Three regression models were implemented:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

### 5. Model Evaluation
- 80/20 train–test split
- 5-fold cross-validation on training data
- Evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² score

### 6. Hyperparameter Tuning
- GridSearchCV used to tune selected model parameters
- Best model selected based on cross-validation RMSE

### 7. Feature Importance
- Feature influence analysed using **Permutation Importance**
- Importance reported on original input features for interpretability
- Location- and size-related attributes were found to have the strongest impact on price predictions

---

## 🚀 Model Deployment

- The final trained model pipeline was saved using `joblib`
- A Gradio-based web application was developed to:
  - Accept user inputs for property features
  - Generate real-time price predictions
- The app is publicly hosted using **Hugging Face Spaces**

---

## 📁 Repository Structure
.
├── app.py # Gradio application for deployment
├── requirements.txt # Python dependencies
├── melbourne_price_model.joblib # Trained regression model pipeline
├── feature_columns.csv # Feature schema used during training
├── ui_reference.csv # Reference data for UI defaults and choices
├── Task8_2_Melbourne_Housing.ipynb# Jupyter notebook (model development)
└── README.md # Project documentation

---

## 🛠️ Technologies Used

- Python 3
- pandas, numpy
- scikit-learn
- joblib
- Gradio
- Hugging Face Spaces

---

```bash
pip install -r requirements.txt
python app.py
