import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/diabetes_model.sav', 'rb'))

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float32)  # Convert input to float
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    st.title('Diabetes Prediction')
    
    Pregnancies = st.text_input('Number of pregnancies', value='0')
    Glucose = st.text_input('Glucose level', value='0')
    BloodPressure = st.text_input('Blood pressure', value='0')
    SkinThickness = st.text_input('Skin thickness', value='0')
    Insulin = st.text_input('Insulin level', value='0')
    BMI = st.text_input('BMI value', value='0')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value', value='0')
    Age = st.text_input('Age of a person', value='0')

    diagnosis = ''
    if st.button('Diabetes Test Result'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = diabetes_prediction([float(i) for i in input_data])  # Convert input to float
    st.success(diagnosis)

if __name__ == '__main__':
    main()
