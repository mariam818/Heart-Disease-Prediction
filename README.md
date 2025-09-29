# Heart Disease Prediction

This project focuses on analyzing, predicting, and visualizing heart disease risk using machine learning models. 
The workflow involves data preprocessing, feature selection, dimensionality reduction (PCA), model training, evaluation, and deployment. Classification models like Logistic Regression, Decision Trees, Random Forest, and SVM , alongside
K-Means and Hierarchical Clustering for unsupervised learning.
It includes a web app built with Streamlit for easy user interaction.

# Project Structure
Heart_Disease_Project/
├── data/                        
│   └── heart_disease.csv
│
├── models/                        
│   └── final_model.pkl
│
├── notebooks/                    
│   ├──01_data_preprocessing.ipynb
│   ├──02_pca_analysis.ipynb
│   ├──03_feature_selection.ipynb
│   ├──04_supervised_learning.ipynb
│   ├──05_unsupervised_learning.ipynb
│   ├──06_hyperparameter_tuning.ipynb
│
├── results/
│   ├──evaluation_metrics.txt
│   
├── ui/                         
│   └── ui.py
|
├── requirements.txt              
├── README.md                    
├── .gitignore                   

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

# Features 
 the following features are used for prediction:
- age: Age of the patient  
- sex: Gender (1 = male, 0 = female)  
- cp: Chest pain type (0–3)  
- trestbps: Resting blood pressure (mm Hg)  
- chol: Serum cholesterol (mg/dl)  
- fbs: Fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)  
- restecg: Resting electrocardiographic results (0–2)  
- thalach: Maximum heart rate achieved  
- exang: Exercise-induced angina (1 = yes; 0 = no)  
- oldpeak: ST depression induced by exercise relative to rest  
- slope: Slope of the peak exercise ST segment (0–2)  
- ca: Number of major vessels (0–3) colored by fluoroscopy  
- thal: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) 

# How to run
1. Clone the repository:
   ``` bash
   git clone https://github.com/mariam818/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   
2. Install dependencies:
pip install -r requirements.txt

3. Run Jupyter notebooks (for analysis and training):
Open and execute each notebook in order under the notebooks/ directory.

3. Run the Streamlit app:
cd ui
streamlit run ui.py
