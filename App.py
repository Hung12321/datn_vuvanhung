import streamlit as st
import pickle
import numpy as np
import os

# Load các model từ file pickle
with open('model/GaussianNB_model.pkl', 'rb') as file:
    gnb_model = pickle.load(file)

with open('model/RandomForestClassifier_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('model/SVM_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('model/min_max_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Tạo giao diện Streamlit
st.title('Crop Prediction')

# Lựa chọn mô hình
model_option = st.selectbox("Choose Model", ["None", "GaussianNB", "RandomForestClassifier", "Support Vector Machine"])

# Đọc các thông số accuracy từ file
accuracy_dict = {}
with open('accuracy.txt', 'r') as file:
    for line in file:
        model_name, accuracy = line.strip().split(' with accuracy: ')
        accuracy_dict[model_name] = float(accuracy)

# Kiểm tra xem mô hình đã được chọn chưa
if model_option != "None":
    st.title('Vui lòng nhập các dữ liệu tương ứng')
    # Hàng ngang cho việc nhập số
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        N = st.number_input('N', step=None)
    with col2:
        P = st.number_input('P', step=None)
    with col3:
        k = st.number_input('k', step=None)
    with col4:
        temperature = st.number_input('Temperature', step=None)

    col5, col6, col7 = st.columns(3)
    with col5:
        humidity = st.number_input('Humidity', step=None)
    with col6:
        ph = st.number_input('PH', step=None)
    with col7:
        rainfall = st.number_input('Rainfall', step=None)

    # Button để dự đoán
    if st.button('Predict'):
        # Tạo một array từ các giá trị nhập vào
        data = np.array([[N, P, k, temperature, humidity, ph, rainfall]])

        # Chuẩn hóa dữ liệu đầu vào
        scaled_data = scaler.transform(data)

        # Dự báo với model tương ứng
        if model_option == "GaussianNB":
            st.write(f"Accuracy of GaussianNB: {accuracy_dict['GaussianNB']}")
            prediction = gnb_model.predict(scaled_data)
        elif model_option == "RandomForestClassifier":
            st.write(f"Accuracy of RandomForestClassifier: {accuracy_dict['RandomForestClassifier']}")
            prediction = rf_model.predict(scaled_data)
        elif model_option == "Support Vector Machine":
            st.write(f"Accuracy of Support Vector Machine: {accuracy_dict['Support Vector Machine']}")
            prediction = svm_model.predict(scaled_data)

        # Hiển thị kết quả dự báo
        # st.write(f"Gợi ý cây trồng: {prediction[0]}")

        # Function to get image path based on crop name
        crop_dict = {
            1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya',
            7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes',
            12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram',
            17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans',
            21: 'chickpea', 22: 'coffee'
        }
        
        crop_name = crop_dict.get(prediction[0], "Unknown")
        st.write(f"Gợi ý cây trồng: {crop_name}")

        image_path = os.path.join("Caytrong", crop_name + ".jpg")
        if os.path.exists(image_path):
            st.image(image_path, caption='Hình ảnh cho cây gợi ý', use_column_width=True)
        else:
            st.write("Không có cây nào tương ứng")
