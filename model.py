import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_data = pd.read_csv("diabetes.csv")

# Split the data into features and target variable
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Define the Streamlit app
def app():
    # Set the app title
    st.title("Diabetes Prediction App")

    # Add input fields for the independent variables
    st.header("Enter the following details:")
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, step=1)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, step=1)
    bmi = st.number_input("Body Mass Index BMI", min_value=0, step=1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0, step=1)
    age = st.number_input("Age", min_value=0, step=1)

    # Create a button to make predictions
    if st.button("Predict"):
        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree_function],
            "Age": [age]
        })

        # Make predictions using the trained classifier
        predictions = clf.predict(user_input)

        # Display the prediction result
        st.subheader("Prediction Result:")
        if predictions[0] == 0:
            st.write("The person is not diabetic.")
        else:
            st.write("The person is diabetic.")

# Run the app
if __name__ == "__main__":
    app()
