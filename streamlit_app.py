import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import shap
from sklearn.preprocessing import MinMaxScaler

# Define feature metadata for Early_Detection models
FEATURE_METADATA = {
    'Age': {'type': 'numerical', 'min': 0, 'max': 120, 'default': 50},
    'Years_of_Smoking': {'type': 'numerical', 'min': 0, 'max': 100, 'default': 0},
    'Cigarettes_per_Day': {'type': 'numerical', 'min': 0, 'max': 100, 'default': 0},
    'Survival_Years': {'type': 'numerical', 'min': 0, 'max': 50, 'default': 0},
    'Gender': {'type': 'categorical', 'options': {'Female': 0, 'Male': 1}, 'default': 'Female'},
    'Smoker': {'type': 'categorical', 'options': {'No': 0, 'Yes': 1}, 'default': 'No'},
    'Passive_Smoker': {'type': 'categorical', 'options': {'No': 0, 'Yes': 1}, 'default': 'No'},
    'Family_History': {'type': 'categorical', 'options': {'No': 0, 'Yes': 1}, 'default': 'No'},
    'Cancer_Stage': {'type': 'categorical', 'options': {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}, 'default': 'Stage I'},
    'Adenocarcinoma_Type': {'type': 'categorical', 'options': {'None': 0, 'Type A': 1, 'Type B': 2}, 'default': 'None'},
    'Air_Pollution_Exposure': {'type': 'categorical', 'options': {'Low': 0, 'Medium': 1, 'High': 2}, 'default': 'Low'},
    'Occupational_Exposure': {'type': 'categorical', 'options': {'No': 0, 'Yes': 1}, 'default': 'No'},
    'Indoor_Pollution': {'type': 'categorical', 'options': {'Low': 0, 'Medium': 1, 'High': 2}, 'default': 'Low'},
    'Healthcare_Access': {'type': 'categorical', 'options': {'Poor': 0, 'Fair': 1, 'Good': 2}, 'default': 'Fair'},
    'Developed_or_Developing': {'type': 'categorical', 'options': {'Developing': 0, 'Developed': 1}, 'default': 'Developing'}
}

# Set page title and layout
st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")
st.title("Model Evaluation Dashboard")

# Load saved model and results
@st.cache_data
def load_model_data():
    try:
        with open('model_results.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Error: model_results.pkl not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model_results.pkl: {e}")
        return None

model_data = load_model_data()
if model_data is None:
    st.stop()

# Extract data
results_dict = model_data["results"]
models_dict = model_data["models"]
predictions_dict = model_data["predictions"]
y_test = model_data["y_test"]
x_train = model_data["x_train"]
x_test = model_data["x_test"]

# Verify x_test columns match FEATURE_METADATA
expected_features = list(FEATURE_METADATA.keys())
if set(x_test.columns) != set(expected_features):
    st.warning(f"Feature mismatch: x_test contains {list(x_test.columns)}, expected {expected_features}")

# Fit a MinMaxScaler on x_train (since original scaler is not saved)
scaler = MinMaxScaler()
scaler.fit(x_train)

# Sidebar: Model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Select a Model", list(results_dict.keys()))

# Main area: Evaluation metrics
st.header(f"{model_choice} Evaluation Metrics")
model_results = results_dict[model_choice]

# Use column layout for metrics
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{model_results['Accuracy']:.4f}")
col2.metric("Precision", f"{model_results['Precision']:.4f}")
col3.metric("Recall", f"{model_results['Recall']:.4f}")
col1.metric("F1 Score", f"{model_results['F1 Score']:.4f}")
col2.metric("ROC-AUC", f"{model_results['ROC-AUC']:.4f}")

# Confusion matrix
st.header(f"Confusion Matrix for {model_choice}")
y_pred = predictions_dict[model_choice]["y_pred"]
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix - {model_choice}")
st.pyplot(fig)
plt.close(fig)

# Debug: Print confusion matrix values
st.write(f"Confusion Matrix Values for {model_choice}:\n{cm}")

# SHAP values
st.header(f"SHAP Values for {model_choice}")
try:
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        explainer = shap.TreeExplainer(models_dict[model_choice])
        shap_values = explainer(x_test.iloc[:50])
        # For binary classification, select SHAP values for the positive class
        if len(shap_values.shape) == 3:  # Binary classification case
            shap_values = shap_values[..., 1]  # Select positive class
    else:  # Logistic Regression or Ridge Logistic Regression
        explainer = shap.Explainer(models_dict[model_choice], x_train)
        shap_values = explainer(x_test.iloc[:50])
        # For linear models, ensure SHAP values are for the positive class
        if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
            shap_values = shap_values[..., 1]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, x_test.iloc[:50], show=False)
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.error(f"Error generating SHAP plot for {model_choice}: {str(e)}")
    st.write("Debug Info:")
    st.write(f"Model type: {type(models_dict[model_choice])}")
    st.write(f"x_test shape: {x_test.iloc[:50].shape}")
    st.write(f"x_test columns: {list(x_test.columns)}")

# Model comparison table
st.header("Model Comparison")
results_df = pd.DataFrame([{
    "Model": model_name,
    "Accuracy": results["Accuracy"],
    "Precision": results["Precision"],
    "Recall": results["Recall"],
    "F1 Score": results["F1 Score"],
    "ROC-AUC": results["ROC-AUC"]
} for model_name, results in results_dict.items()])
st.dataframe(results_df.style.format("{:.4f}", subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]))

# Prediction section
st.header("Predict Early Detection")
st.write("Enter feature values to predict Early Detection of lung cancer.")

# Input form for features
input_data = {}
with st.form(key="prediction_form"):
    st.subheader("Input Features")
    for feature, metadata in FEATURE_METADATA.items():
        if metadata['type'] == 'numerical':
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=float(metadata['min']),
                max_value=float(metadata['max']),
                value=float(metadata['default']),
                step=0.1,
                help=f"Enter a value between {metadata['min']} and {metadata['max']}"
            )
        else:  # categorical
            options = list(metadata['options'].keys())
            selected_option = st.selectbox(
                f"{feature}",
                options,
                index=options.index(metadata['default']),
                help=f"Select one of: {', '.join(options)}"
            )
            input_data[feature] = metadata['options'][selected_option]
    
    submit_button = st.form_submit_button("Predict")

# Perform prediction
if submit_button:
    input_df = pd.DataFrame([input_data])
    try:
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        input_df_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
        
        # Make prediction
        selected_model = models_dict[model_choice]
        prediction = selected_model.predict(input_df_scaled)
        probability = selected_model.predict_proba(input_df_scaled)[0, 1] if hasattr(selected_model, "predict_proba") else None
        
        st.success(f"Prediction: {'Early Detection' if prediction[0] == 1 else 'No Early Detection'}")
        if probability is not None:
            st.info(f"Probability of Early Detection: {probability:.4f}")
        else:
            st.warning("The selected model does not support probability prediction.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")