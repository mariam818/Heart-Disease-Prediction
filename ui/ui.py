import os
import pickle
import pandas as pd
import streamlit as st

# 1. Page Config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

def get_model_path():
    """
    Get the correct model path based on your project structure
    Current structure: Heart_Disease_Project/ui/ui.py
    Model location: Heart_Disease_Project/models/final_model.pkl
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This gives ui/ folder
    project_root = os.path.dirname(current_dir)  # This gives Heart_Disease_Project/
    model_path = os.path.join(project_root, "models", "final_model.pkl")
    return model_path

MODEL_PATH = get_model_path()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    
    # Show directory structure for debugging
    st.subheader("Current Directory Structure")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(project_root):
        for root, dirs, files in os.walk(project_root):
            level = root.replace(project_root, '').count(os.sep)
            if level > 2:  
                continue
            indent = ' ' * 2 * level
            st.write(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.endswith('.pkl'):
                    st.write(f"{subindent}📁 {file} (PKL file)")
    
    # File upload fallback
    st.sidebar.header("Alternative: Upload Model File")
    uploaded_file = st.sidebar.file_uploader("Upload final_model.pkl", type=['pkl'])
    if uploaded_file is not None:
        MODEL_PATH = "uploaded_model.pkl"
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Model uploaded successfully!")
    else:
        st.stop()

# 3. Load trained model
import joblib 
try:
    # Try loading model
    model = joblib.load(MODEL_PATH)
    st.success(f"Model loaded successfully!")

    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        st.sidebar.info(f"Model expects {len(expected_features)} features")
    else:
        st.error("Model doesn't have feature names")
        st.stop()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.warning("If the model is missing , drag & drop the correct file below:")

    # Sidebar file upload
    st.sidebar.header("📂 Upload Model File")
    uploaded_file = st.sidebar.file_uploader("Upload final_model.pkl", type=['pkl'])
    if uploaded_file is not None:
        MODEL_PATH = "uploaded_model.pkl"
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            model = joblib.load(MODEL_PATH)
            st.success("Model uploaded and loaded successfully!")
        except Exception as e2:
            st.error(f"Couldn’t load model: {e2}")
            st.stop()
    else:
        st.stop()


# 4. Sidebar Inputs
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.number_input("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                             format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                                   "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                              format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2], 
                                  format_func=lambda x: ["Normal", "ST-T Abnormality", 
                                                        "Left Ventricular Hypertrophy"][x])
    thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1], 
                                format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope", [0, 1, 2], 
                                format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3], 
                               format_func=lambda x: ["Normal", "Fixed Defect", 
                                                     "Reversible Defect", "Other"][x])

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame(data, index=[0])

df_user = user_input_features()

# 5. Feature alignment function
def align_features(df_user, expected_features):
    df_aligned = pd.DataFrame()
    
    # Create one-hot encoded features for categorical variables
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    
    for col in categorical_cols:
        if col in df_user.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df_user[col], prefix=col)
            # Ensure all possible categories are present
            for i in range(4):  # 0-3 for all categoricals
                dummy_col = f"{col}_{i}"
                if dummy_col in expected_features:
                    df_aligned[dummy_col] = dummies.get(dummy_col, 0)
    
    # Add numerical features directly
    numerical_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca']
    for col in numerical_cols:
        if col in expected_features:
            df_aligned[col] = df_user[col]
    
    # Add any missing expected features with 0
    for feature in expected_features:
        if feature not in df_aligned.columns:
            df_aligned[feature] = 0
    
    # Ensure correct order
    return df_aligned[expected_features]

# 6. Main Interface
st.title(" Heart Disease Prediction App")
st.write("Adjust the parameters in the sidebar and click 'Predict' to see the results.")

# Display user inputs
st.subheader("User Input Features")
st.dataframe(df_user)

# 7. Prediction
if st.button("Predict"):
    try:
        # Align features
        df_final = align_features(df_user, expected_features)
        
        # Make prediction
        prediction = model.predict(df_final)
        prediction_proba = model.predict_proba(df_final)
        
        # Display results
        st.subheader("Result")
        if prediction[0] == 1:
            st.error("🫀 Heart Disease Detected")
        else:
            st.success("No Heart Disease !!")
            
        st.subheader("Probability")
        prob_df = pd.DataFrame({
            'No Heart Disease': [f"{prediction_proba[0][0]:.3f}"],
            'Heart Disease': [f"{prediction_proba[0][1]:.3f}"]
        })
        st.dataframe(prob_df)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        

        
