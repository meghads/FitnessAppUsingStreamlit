import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import streamlit as st

# Load the dataset
data = pd.read_csv('training_datanew.csv')

# Fill missing values
data.fillna('None', inplace=True)

# Preprocess categorical columns
label_encoders = {}
categorical_cols = ['Gender', 'Lifestyle Type', 'Allergies', 'Medical Conditions',
                    'Stress Levels', 'Diet Type', 'Physical Limitations']

for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Encode 'Goal' and meal choice columns
goal_encoder = LabelEncoder()
data['Goal'] = goal_encoder.fit_transform(data['Goal'])

meal_encoders = {'Breakfast Choices': LabelEncoder(), 'Lunch Choices': LabelEncoder(), 'Dinner Choices': LabelEncoder()}
for meal in meal_encoders.keys():
    data[meal] = meal_encoders[meal].fit_transform(data[meal])

# Select features and labels
X = data[['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'Goal', 'Lifestyle Type',
          'Allergies', 'Medical Conditions', 'Stress Levels', 'Diet Type', 'Physical Limitations']]
y_regression = data[['Sleep Duration (hours/day)', 'Water Intake (liters/day)', 'Daily Steps Goal',
                     'Carbohydrates (%)', 'Proteins (%)', 'Fats (%)', 'Fiber Intake (grams/day)',
                     'Sugar Intake Limit (grams/day)', 'Sodium Intake Limit (mg/day)',
                     'Daily Fruit Intake Goal', 'Daily Vegetable Intake Goal']]

# For categorical outputs, we create separate labels
y_classification = data[['Preferred Foods', 'Avoided Foods', 'Preferred Exercise Types',
                         'Exercise Frequency (days/week)', 'Average Exercise Duration (minutes/day)']]

# Split data
X_train, X_test, y_train_regression, y_test_regression = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Define and train regression models for continuous parameters
regression_models = {
    'Sleep Duration (hours/day)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Water Intake (liters/day)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Daily Steps Goal': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Carbohydrates (%)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Proteins (%)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Fats (%)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Fiber Intake (grams/day)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Sugar Intake Limit (grams/day)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Sodium Intake Limit (mg/day)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Daily Fruit Intake Goal': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Daily Vegetable Intake Goal': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
}

# Define and train classification models for categorical parameters
classification_models = {
    'Preferred Foods': RandomForestClassifier(n_estimators=200, random_state=42),
    'Avoided Foods': RandomForestClassifier(n_estimators=200, random_state=42),
    'Preferred Exercise Types': RandomForestClassifier(n_estimators=200, random_state=42),
    'Exercise Frequency (days/week)': RandomForestClassifier(n_estimators=200, random_state=42),
    'Average Exercise Duration (minutes/day)': RandomForestClassifier(n_estimators=200, random_state=42)
}

# Train regression models
for target, model in regression_models.items():
    model.fit(X_train, y_train_regression[target])

# Train classification models
for target, model in classification_models.items():
    model.fit(X_train_class, y_train_class[target])

# Streamlit interface
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Prediction", "Contact Us"])

if page == "Home":
    st.title("Welcome to the Personalized Health Recommendation System - FitBot")
    st.write("""
        FitBot is designed to help you achieve your health and fitness goals by providing 
        personalized recommendations based on your individual needs and preferences. 
        Whether you're looking to lose weight, gain muscle, or maintain your current health, 
        our system can provide tailored suggestions to support your journey.
        
        Enter your details in the Prediction section to get started with your personalized recommendations!
    """)

elif page == "About":
    st.title("About FitBot")
    st.write("""
        FitBot is a Personalized Health Recommendation System uses advanced machine learning algorithms 
        to analyze user input data and generate customized health recommendations. Our system 
        considers various factors including age, gender, weight, lifestyle, dietary preferences, 
        and health goals to provide comprehensive suggestions for diet, exercise, and wellness.
        
        Our aim is to empower individuals to make informed health choices and to provide a 
        user-friendly experience in managing their health and fitness journeys. 
    """)

elif page == "Prediction":
    st.title("FitBot - Personalized Health Recommendation System")

    # User inputs
    age = st.number_input("Enter your age:", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Select your gender:", options=["Male", "Female", "Other"])
    height = st.number_input("Enter your height (cm):", min_value=50, max_value=250, value=170)
    weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=300, value=70)
    goal = st.selectbox("Select your goal:", options=["Weight Loss", "Muscle Gain", "Maintenance"])
    lifestyle = st.selectbox("Select your lifestyle type:", options=["Sedentary", "Very Active", "Active"])
    allergies = st.selectbox("Select any allergies:", options=["None", "Lactose", "Peanuts", "Gluten"])
    medical_conditions = st.selectbox("Select any medical conditions:", options=["None", "Hypertension", "Diabetes", "Asthma"])
    stress_levels = st.selectbox("Select your stress levels:", options=["Low", "Moderate", "High"])
    diet_type = st.selectbox("Select your diet type:", options=["Vegetarian", "Balanced", "Keto", "Vegan"])
    physical_limitations = st.selectbox("Select any physical limitations:", options=["None", "Back Pain", "Knee Pain"])

    # Button to submit user data
    if st.button("Get Recommendations"):
        # Prepare user data
        user_data = {
            "Age": age,
            "Gender": gender,
            "Height (cm)": height,
            "Weight (kg)": weight,
            "Goal": goal,
            "Lifestyle Type": lifestyle,
            "Allergies": allergies,
            "Medical Conditions": medical_conditions,
            "Stress Levels": stress_levels,
            "Diet Type": diet_type,
            "Physical Limitations": physical_limitations,
        }

        # Function for recommending based on user input
        def recommend_health(user_data):
            # Convert user data to DataFrame
            input_df = pd.DataFrame(user_data, index=[0])

            # Preprocess input
            for col in categorical_cols:
                if col in input_df.columns:
                    input_df[col] = label_encoders[col].transform(input_df[col])
            input_df['Goal'] = goal_encoder.transform(input_df['Goal'])

            # Make predictions for regression parameters
            regression_predictions = {target: round(model.predict(input_df)[0]) for target, model in regression_models.items()}

            # Make predictions for classification parameters
            classification_predictions = {target: model.predict(input_df)[0] for target, model in classification_models.items()}

            # Combine predictions
            return {**regression_predictions, **classification_predictions}

        # Get recommendations
        recommendations = recommend_health(user_data)

        # Display recommendations
        st.subheader("Personalized Health Recommendations:")
        for key, value in recommendations.items():
            st.write(f"{key}: {value}")

elif page == "Contact Us":
    st.title("Contact Us")
    st.write("""
        We would love to hear from you! If you have any questions, feedback, or suggestions, 
        please feel free to reach out to us.
        
        Email: support@healthrecommendation.com  
        Phone: +1-800-123-4567  
        
        Your health is our priority, and we are here to help you on your journey to better well-being!
    """)
