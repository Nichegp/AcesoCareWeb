import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from PIL import Image
import pytesseract
import re


def render():
    # ---------------- LOAD DATA ----------------
    parkinsons_data = pd.read_csv('./data/parkinsons.csv')

    # Data split
    X = parkinsons_data.drop(columns=['status', 'name'], axis=1)
    Y = parkinsons_data['status']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Data scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training
    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)

    # Accuracy check
    train_prediction = model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, train_prediction)
    test_prediction = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, test_prediction)
    prefill = {}
    def normalize_ocr_numbers(text):
        text = text.lower()

        # normalize dash characters
        text = text.replace("–", "-").replace("—", "-")

        # attach minus sign to number
        text = re.sub(r"-\s+(\d)", r"-\1", text)

        # join split decimals: -145 555000 → -145.555000
        text = re.sub(r"(-?\d+)\s+(\d{6})", r"\1.\2", text)

        # safe cleanup
        text = re.sub(r"[^0-9a-zA-Z\s\.\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ---------------- FORM ----------------
    def parkinsons_form(prefill=None):
        if prefill is None:
            prefill = {}
        mdvp_fo = st.text_input("MDVP:Fo(Hz)", value=prefill.get("mdvp_fo_hz", "0.0"))
        mdvp_fhi = st.text_input("MDVP:Fhi(Hz)", value=prefill.get("mdvp_phi_hz", "0.0"))
        mdvp_flo = st.text_input("MDVP:Flo(Hz)", value=prefill.get("mdvp_flo_hz", "0.0"))
        jitter_pct = st.text_input("MDVP:Jitter(%)", value=prefill.get("mdvp_jitter_percent", "0.0"))
        jitter_abs = st.text_input("MDVP:Jitter(Abs)", value=prefill.get("mdvp_jitter_abs", "0.0"))
        rap = st.text_input("MDVP:RAP", value=prefill.get("mdvp_rap", "0.0"))
        ppq = st.text_input("MDVP:PPQ", value=prefill.get("ppq", "0.0"))
        ddp = st.text_input("Jitter:DDP", value=prefill.get("jitter_ddp", "0.0"))
        shimmer = st.text_input("MDVP:Shimmer", value=prefill.get("mdvp_shimmer", "0.0"))
        shimmer_db = st.text_input("MDVP:Shimmer(dB)", value=prefill.get("mdvp_shimmer_db", "0.0"))
        apq3 = st.text_input("Shimmer:APQ3", value=prefill.get("shimmer_apq3", "0.0"))
        apq5 = st.text_input("Shimmer:APQ5", value=prefill.get("shimmer_apq5", "0.0"))
        apq = st.text_input("MDVP:APQ", value=prefill.get("mdvp_apq", "0.0"))
        dda = st.text_input("Shimmer:DDA", value=prefill.get("shimmer_dda", "0.0"))
        nhr = st.text_input("NHR", value=prefill.get("nhr", "0.0"))
        hnr = st.text_input("HNR", value=prefill.get("hnr", "0.0"))
        rpde = st.text_input("RPDE", value=prefill.get("rpde", "0.0"))
        dfa = st.text_input("DFA", value=prefill.get("dfa", "0.0"))
        spread1 = st.text_input("Spread1", value=prefill.get("spread1", "0.0"))
        spread2 = st.text_input("Spread2", value=prefill.get("spread2", "0.0"))
        d2 = st.text_input("D2", value=prefill.get("d2", "0.0"))
        pre = st.text_input("PPE", value=prefill.get("pre", "0.0"))

        return np.array([
            mdvp_fo, mdvp_fhi, mdvp_flo, jitter_pct, 
            jitter_abs,
            rap, ppq, ddp, shimmer, shimmer_db,
            apq3, apq5, apq, dda, nhr,hnr,
            rpde, dfa, spread1, spread2, d2, pre
        ]).reshape(1, -1)
    left_col, right_col = st.columns([2, 1])
    with right_col:
        st.markdown(
            f'<div style="background-color: #e8f5e8; border-left: 3px solid #4caf50; padding: 5px; margin: 6px 0; border-radius: 4px;">'
            f'<p style="color:#2e7d32; margin:0; font-weight:600; font-size:14px;">📊 Model Performance</p>'
            f'<p style="color: #388e3c; margin: 2px 0 0 0; font-weight: 500; font-size: 12px;"> Data Accuracy: {train_accuracy:.2%}</p></div>',
            unsafe_allow_html=True
        )
    with left_col:
        st.markdown(
            f'<h2 style=font-size:20px; margin-bottom:10px;"> Please choose how you would like to provide your medical details:</h2>',
            unsafe_allow_html=True)
        input_mode = st.radio(
            "Select how you want to enter patient data:",
            ["📷 Scan Parkinsons Report", "⌨️ Manually Enter Data"],horizontal=True
        )
    if input_mode == "📷 Scan Parkinsons Report":
        uploaded = st.file_uploader(" Upload Medical Report", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, width=250)
            extracted_text = pytesseract.image_to_string(img).lower()
            extracted_text = re.sub(r"[¢°)—]", " ", extracted_text)
            extracted_text = re.sub(r"\.", " ", extracted_text)
            extracted_text = re.sub(r"[^0-9a-zA-Z\s\.\-]", " ", extracted_text)
            extracted_text = re.sub(r"\s+", " ", extracted_text).strip()
            extracted_text = normalize_ocr_numbers(extracted_text)

            def extract(pattern, text, default=None):
                m = re.search(pattern, text, re.I)
                return m.group(1).strip() if m else default
            patient_id = extract(r"patient id\s*[:\-]?\s*([a-z0-9\-]+)", extracted_text),
            name       = extract(r"name\s*[:\-]?\s*([a-z ]+)", extracted_text)
            age        = extract(r"age\s*[:\-]?\s*(\d+)", extracted_text)
            sex        = extract(r"sex\s*[:\-]?\s*(male|female)", extracted_text)
            date       = extract(r"date\s*[:\-]?\s*([0-9]{2}-[a-z]{3}-[0-9]{4})", extracted_text)
            numbers    = re.findall(r"-?\d+\.\d+", extracted_text)
            params = [
                    "mdvp_fo_hz",
                    "mdvp_phi_hz",
                    "mdvp_flo_hz",
                    "mdvp_jitter_percent",
                    "mdvp_jitter_abs",
                    "mdvp_rap",
                    "jitter_ddp",
                    "mdvp_shimmer",
                    "mdvp_shimmer_db",
                    "shimmer_apq3",
                    "shimmer_apq5",
                    "mdvp_apq",
                    "shimmer_dda",
                    "nhr",
                    "hnr",
                    "rpde",
                    "dfa",
                    "spread1",
                    "spread2",
                    "d2",
                    "pre",
                    "ppq",
                ]
            prefill = dict(zip(params, numbers))
            st.subheader("Auto-Filled Form Below (Editable)")
            input_data = parkinsons_form(prefill)
            if st.button("Parkinson Check"):
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                std_data = scaler.transform(input_data_reshaped)
                prediction = model.predict(std_data)
                if prediction[0] == 0:
                    st.success("✅ Amazing News! Low chance of Parkinson's Disease!")
                else:
                    st.error("⚠️ High chance of Parkinson's Disease. Please consult a doctor using the sidebar on the left.")
    else:
        # Manual data entry form
        st.subheader("Enter Medical Details")
        input_data = parkinsons_form()
    # ---------------- PREDICT ----------------
        if st.button("Parkinson Check"):
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            std_data = scaler.transform(input_data_reshaped)
            prediction = model.predict(std_data)
            if prediction[0] == 0:
                st.success("✅ Amazing News! Low chance of Parkinson's Disease!")
            else:
                st.error("⚠️ High chance of Parkinson's Disease. Please consult a doctor using the sidebar on the left.")
