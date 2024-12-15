# Laptop Price Prediction 💻📈
## Overview
The Laptop Price Prediction project aims to predict the price range of laptops based on various features like brand, processor type, RAM, storage, display size, and GPU. It uses machine learning models to analyze patterns in the dataset and provide accurate predictions.

This project can assist buyers in comparing laptops or help businesses understand pricing trends in the market.

## Key Features
- ### Data Preprocessing:

  - Cleaned and encoded raw data for machine learning models.
  - Handled missing values, outliers, and scaling of numerical data.
- ### Exploratory Data Analysis (EDA):

  - Analyzed trends in laptop pricing based on features like RAM, brand, and GPU.
  - Visualized relationships between features and prices.
- ### Machine Learning Models:

  - Regression models (Linear Regression, Random Forest, Gradient Boosting) for price prediction.
  - Evaluated models using metrics like R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- ### Interactive Prediction:

  - Built an optional user interface using Streamlit for real-time predictions.
## Technologies Used
- ### Programming Language: Python
- ### Libraries and Frameworks:
  - #### Pandas
  - #### NumPy
  - #### Scikit-learn
  - #### Matplotlib
  - #### Seaborn
  - #### Streamlit (for UI)

## Dataset
- ### The dataset contains details of laptops, including:

  - Brand
  - Processor type (e.g., Intel i5, AMD Ryzen 7)
  - RAM size and type (e.g., DDR4)
  - Storage size and type (e.g., SSD, HDD)
  - GPU (e.g., NVIDIA GTX, Integrated)
  - Display size and resolution
  - Operating System
  - **Source:** Public datasets (e.g., Kaggle, e-commerce websites)

## How It Works
- ### Data Preprocessing:

  - Cleaned the dataset and encoded categorical features.
  - Scaled numerical features for better model performance.
- ### Exploratory Data Analysis (EDA):

  - Visualized the impact of RAM, GPU, and other features on price.
  - Identified correlations between features.
- ### Model Development:

  - Trained regression models to predict prices.
  - Input: Laptop features (e.g., processor, RAM, GPU).
  - Output: Predicted price range.
- ### Interactive UI:

  - Users can input laptop specifications through the Streamlit interface to get price predictions.

## Future Enhancements
- Add live data integration from e-commerce websites for dynamic predictions.
- Incorporate advanced deep learning models for better predictions.
- Include additional features like battery life and weight.
- Deploy the project on a cloud platform like AWS or Heroku.
