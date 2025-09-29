# Heart Disease Prediction

This project focuses on analyzing, predicting, and visualizing heart disease risk using machine learning models.  
It includes a web app built with Streamlit for easy user interaction.

# Project Structure
Heart_Disease_Project/
│── data/ # Dataset files
│── notebooks/ # Jupyter notebooks for EDA & model training
│── models/ # Saved model (.pkl)
│── ui/ # Streamlit app 
│── README.md # Project documentation
│── .gitignore # Files to ignore in git

# Models Used
Supervised:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)
Unsupervised:
- K-Means
- Hierarchical Clustering

# Dataset
The dataset is based on the UCI Heart Disease dataset, commonly used in medical ML research.

# How to run
1. Clone the repository:
   ``` bash
   git clone https://github.com/mariam818/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   
2. Install dependencies:
pip install -r requirements.txt

3. Run the Streamlit app:
cd ui
streamlit run ui.py
