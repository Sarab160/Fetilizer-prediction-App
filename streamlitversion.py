import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("Fertilizer Prediction.csv")

# print(df.info())
# # print(df["Crop Type"].unique)
# # print(np.unique(df["Crop Type"]))
# # print(df.shape)
# # sns.pairplot(data=df)
# # plt.show()
# print(df.columns)

# -------------------------------
# Features & Encoding
# -------------------------------
x = df[['Temparature', 'Humidity ', 'Moisture',
        'Nitrogen', 'Potassium', 'Phosphorous']]

le = LabelEncoder()
y = le.fit_transform(df["Fertilizer Name"])

oe = OrdinalEncoder()
fe = df[["Soil Type", "Crop Type"]]
encode_array = oe.fit_transform(fe)
encodedata = pd.DataFrame(encode_array, columns=fe.columns)

final_x = pd.concat([x, encodedata], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    final_x, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌱 Fertilizer Prediction App")

st.sidebar.header("User Input Features")

# User input fields
temp = st.sidebar.number_input("Temperature", value=25)
humidity = st.sidebar.number_input("Humidity", value=60)
moisture = st.sidebar.number_input("Moisture", value=40)
nitrogen = st.sidebar.number_input("Nitrogen", value=20)
potassium = st.sidebar.number_input("Potassium", value=20)
phosphorous = st.sidebar.number_input("Phosphorous", value=20)
soil_type = st.sidebar.selectbox("Soil Type", df["Soil Type"].unique())
crop_type = st.sidebar.selectbox("Crop Type", df["Crop Type"].unique())

# Encode user input
user_encoded = oe.transform([[soil_type, crop_type]])
user_data = pd.DataFrame([[temp, humidity, moisture, nitrogen, potassium, phosphorous]],
                         columns=['Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])
user_final = pd.concat([user_data, pd.DataFrame(user_encoded, columns=["Soil Type", "Crop Type"])], axis=1)

# Prediction button
if st.button("🔮 Predict Fertilizer"):
    pred = lr.predict(user_final)[0]
    fertilizer_name = le.inverse_transform([pred])[0]
    st.success(f"Recommended Fertilizer: **{fertilizer_name}**")

# Accuracy and metrics
if st.checkbox("📊 Show Model Accuracy & Metrics"):
    y_pred = lr.predict(x_test)
    acc_test = lr.score(x_test, y_test)
    acc_train = lr.score(x_train, y_train)
    st.write(f"✅ Training Accuracy: {acc_train:.2f}")
    st.write(f"✅ Testing Accuracy: {acc_test:.2f}")
    st.write(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.2f}")
    st.write(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.2f}")
    st.write(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.2f}")

# Graphs button
if st.button("📉 Show Confusion Matrix"):
    y_pred = lr.predict(x_test)
    cf = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cf, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
