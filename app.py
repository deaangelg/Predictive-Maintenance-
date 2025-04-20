import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("Predicción de Falla en Equipos - Mantenimiento Predictivo")

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("predictive_maintenance.csv")

df = load_data()

# Preprocesamiento
X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Failure Type']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=2025)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown(f"**Precisión del modelo:** `{accuracy:.2%}`")

# Inputs del usuario
st.subheader("Ingrese los datos del equipo:")
air_temp = st.number_input("Temperatura del aire [K]", min_value=280.0, max_value=400.0, value=300.0)
process_temp = st.number_input("Temperatura del proceso [K]", min_value=280.0, max_value=400.0, value=310.0)
rpm = st.number_input("Velocidad rotacional [rpm]", min_value=0, max_value=5000, value=1500)
torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
wear = st.number_input("Desgaste de herramienta [min]", min_value=0, max_value=500, value=100)

# Diccionario de imágenes
imagenes_falla = {
    "Power Failure": "Power Failure/power_failure.png",
    "Tool Wear Failure": "Power Failure/tool_wear_failure.png",
    "Overstrain Failure": "Power Failure/overstrain_failure.png",
    "Random Failures": "Power Failure/random_failure.png",
    "Heat Dissipation Failure": "Power Failure/heat_dissipation_failure.png",
    "No Failure": "Power Failure/no_failure.png"
}

# Botón de predicción
if st.button("Predecir tipo de falla"):
    input_data = pd.DataFrame([[air_temp, process_temp, rpm, torque, wear]],
                              columns=X.columns)
    
    prediction_encoded = model.predict(input_data)[0]
    prediction = le.inverse_transform([prediction_encoded])[0]

    # Mostrar resultado
    if prediction == "No Failure":
        st.success("No se predice ninguna falla en el equipo.")
    else:
        st.error(f"Se predice una falla: **{prediction}**")

    # Mostrar imagen correspondiente
    if prediction in imagenes_falla:
        st.image(imagenes_falla[prediction], caption=prediction)
    else:
        st.warning("No se encontró una imagen para esta falla.")
