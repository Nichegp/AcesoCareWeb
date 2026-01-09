import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pytesseract
import re


def render():
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    heart_data = pd.read_csv('./data/heart.csv')
    patients_data = heart_data.drop(columns='target', axis=1)
    disease_prediction = heart_data['target']

    patients_data_train, patients_data_test, disease_prediction_train, disease_prediction_test = train_test_split(
        patients_data, disease_prediction, test_size=0.2, stratify=disease_prediction, random_state=2
    )

    # Scale features to help Logistic Regression (lbfgs) converge reliably
    # Fit scaler on the underlying numpy arrays so we don't depend on DataFrame feature names
    scaler = StandardScaler()
    patients_data_train = scaler.fit_transform(patients_data_train.values)
    patients_data_test = scaler.transform(patients_data_test.values)
    # Increase max_iter to avoid lbfgs convergence warnings
    model = LogisticRegression(max_iter=1000)
    model.fit(patients_data_train, disease_prediction_train)
    patients_data_train_prediction = model.predict(patients_data_train)
    accuracy_of_train_data = accuracy_score(patients_data_train_prediction, disease_prediction_train)
    disease_prediction_test_prediction = model.predict(patients_data_test)
    test_data_accuracy = accuracy_score(disease_prediction_test_prediction, disease_prediction_test)

    left_col, right_col = st.columns([2, 1])  # wider left, narrower right
    with right_col:
        st.markdown(
            f'<div style="background-color: #e8f5e8; border-left: 3px solid #4caf50; padding: 5px; margin: 6px 0; border-radius: 4px;">'
            f'<p style="color:#2e7d32; margin:0; font-weight:600; font-size:14px;">📊 Model Performance</p>'
            f'<p style="color: #388e3c; margin: 2px 0 0 0; font-weight: 500; font-size: 12px;"> Data Accuracy: {accuracy_of_train_data:.2%}</p></div>',
            unsafe_allow_html=True
        )
    with left_col:
        st.markdown(
            f'<h2 style=font-size:20px; margin-bottom:10px;"> Please choose how you would like to provide your medical details:</h2>',
            unsafe_allow_html=True)
        input_mode = st.radio(
            "Select how you want to enter patient data:",
            ["📷 Scan My Report", "⌨️ I can manually enter my data"],horizontal=True
        )
        if input_mode == "📷 Scan My Report":
            uploaded_img = st.file_uploader("Upload Medical Report", type=["jpg", "jpeg", "png"])
            if uploaded_img:
                img = Image.open(uploaded_img)
                st.image(img, caption="Scanned Image Preview", width=400)
                extracted_text = pytesseract.image_to_string(img)
                def extract(pattern, text):
                    match = re.search(pattern, text, re.IGNORECASE)
                    return float(match.group(1)) if match else 0
                
                # Extract values (OCR-based)
                age = extract(r"age[: ]+(\d+)", extracted_text)
                sex = extract(r"sex[: ]+(\d+)", extracted_text)
                cp = extract(r"cp[: ]+(\d+)", extracted_text)
                trestbps = extract(r"bp[: ]+(\d+)", extracted_text)
                chol = extract(r"chol[: ]+(\d+)", extracted_text)
                fbs = extract(r"fbs[: ]+(\d+)", extracted_text)
                restecg = extract(r"ecg[: ]+(\d+)", extracted_text)
                thalach = extract(r"hr[: ]+(\d+)", extracted_text)
                exang = extract(r"exang[: ]+(\d+)", extracted_text)
                oldpeak = extract(r"oldpeak[: ]+(\d+\.?\d*)", extracted_text)
                slope = extract(r"slope[: ]+(\d+)", extracted_text)
                ca = extract(r"ca[: ]+(\d+)", extracted_text)
                thal = extract(r"thal[: ]+(\d+)", extracted_text)
                
                st.subheader("Auto-Filled Form Below (Editable)")
                
                # STEP 3 — UI fields auto-fill
                age = st.number_input("Age", value=int(age))
                sex = st.number_input("Sex", value=int(sex))
                cp = st.number_input("Chest Pain (cp)", value=int(cp))
                trestbps = st.number_input("Resting BP", value=int(trestbps))
                chol = st.number_input("Cholesterol", value=int(chol))
                fbs = st.number_input("FBS", value=int(fbs))
                restecg = st.number_input("Rest ECG", value=int(restecg))
                thalach = st.number_input("Max HR", value=int(thalach))
                exang = st.number_input("Exercise Angina", value=int(exang))
                oldpeak = st.number_input("Oldpeak", value=float(oldpeak))
                slope = st.number_input("Slope", value=int(slope))
                ca = st.number_input("CA count", value=int(ca))
                thal = st.number_input("Thal score", value=int(thal))

                if st.button("CardioAnalyze"):
                    input_data = np.array([
                        age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal
                    ]).reshape(1, -1)
                    input_data_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_data_scaled)
                    if prediction[0] == 0:
                        st.success("✅ Amazing News! Low chance of Heart Disease!")
                    else:
                        st.error("⚠️ High chance of Heart Disease. To find nearby hospitals, use the sidebar on the left.")
        
        else:
            # Manual data entry form
            gender_mapping = {
                "Male": 0,
                "Female": 1
            }
            cp_mapping = {
                "Typical Angina": 0,
                "Atypical Angina": 1,
                "Non-Anginal Pain": 2,
                "Asymptomatic": 3
            }
            fbs_mapping = {
                "Fasting Blood Sugar > 120 mg/dl": 1,
                "Otherwise": 0
            }
            restecg_mapping = {
                "Normal": 0,
                "ST-T wave abnormality": 1,
                "Left ventricular hypertrophy": 2
            }
            exang_mapping = {
                "Yes": 1,
                "No": 0
            }
            slope_mapping = {
                "Upsloping": 0,
                "Flat": 1,
                "Downsloping": 2
            }
            thal_mapping = {
                "Normal": 0,
                "Fixed Defect": 1,
                "Reversable Defect": 2
            }

            with st.form("heart_form"):
                name = st.text_input("Enter the Name: ")
                age = st.number_input("Enter the age: ", min_value=0, max_value=120, step=1)
                gender = st.selectbox("Enter the Gender: ", list(gender_mapping.keys()))
                cp = st.selectbox("Enter the chest pain type: ", list(cp_mapping.keys()))
                trestbps = st.number_input("Enter the resting blood pressure: ")
                chol = st.number_input("Enter the serum cholesterol: ")
                fbs = st.selectbox("Enter the fasting blood sugar: ", list(fbs_mapping.keys()))
                restecg = st.selectbox("Enter the resting electrocardiographic results: ", list(restecg_mapping.keys()))
                thalach = st.number_input("Enter the maximum heart rate achieved: ")
                exang = st.selectbox("Enter the exercise induced angina: ", list(exang_mapping.keys()))
                oldpeak = st.number_input("Enter the ST depression induced by exercise relative to rest: ")
                slope = st.selectbox("Enter the slope of the peak exercise ST segment: ", list(slope_mapping.keys()))
                ca = st.number_input("Enter the number of major vessels colored by flouroscopy: ")
                thal = st.selectbox("Enter the thalassemia: ", list(thal_mapping.keys()))
                submitHeart = st.form_submit_button("CardioAnalyze")
                if submitHeart:
                    gender_value = gender_mapping[gender]
                    cp_value = cp_mapping[cp]
                    fbs_value = fbs_mapping[fbs]
                    restecg_value = restecg_mapping[restecg]
                    exang_value = exang_mapping[exang]
                    slope_value = slope_mapping[slope]
                    thal_value = thal_mapping[thal]
                        
                    input_data = [age, gender_value, cp_value, trestbps, chol, fbs_value, restecg_value, thalach, exang_value, oldpeak, slope_value, ca, thal_value]
                    input_data_as_numpy_array = np.asarray(input_data)
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                        # Apply the same scaling used during training
                    input_data_scaled = scaler.transform(input_data_reshaped)
                    prediction = model.predict(input_data_scaled)
                    print(prediction)
                    if prediction[0] == 0:
                        st.success("✅ Amazing News! Low chance of Heart Disease!")
                    else:
                        st.error("⚠️ High chance of Heart Disease. To find nearby hospitals, use the sidebar on the left.")
            