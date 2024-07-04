import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Function to load the vectorizer
def load_vectorizer(vectorizer_path):
    if os.path.exists(vectorizer_path):
        return joblib.load(vectorizer_path)
    else:
        st.error(f"Vectorizer file not found at {vectorizer_path}")
        return None

# Define absolute paths
base_path = os.path.dirname(__file__)
vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")

# Load the vectorizer
vectorizer = load_vectorizer(vectorizer_path)

# List of available models
model_paths = {
    "Logistic Regression": os.path.join(base_path, "logistic_regression_model.pkl"),
    "Naive Bayes": os.path.join(base_path, "naive_bayes_model.pkl"),
    "Support Vector Machine": os.path.join(base_path, "support_vector_machine_model.pkl")
}

# Category mapping (Example: Replace with your actual categories)
category_mapping = {
    0: "Business",
    1: "Education",
    2: "Entertainment",
    3: "Sports",
    4: "Technology"
}

# Main function to build the Streamlit app
def main():
    """News Classifier App with Streamlit"""
    st.set_page_config(page_title="News Classifier App", page_icon=":newspaper:")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Go to", ["Home", "Information", "Prediction"])

    # Main content based on selection
    if page_selection == "Home":
        show_home_page()
    elif page_selection == "Information":
        show_information_page()
    elif page_selection == "Prediction":
        show_prediction_page()

def show_home_page():
    st.title("Welcome to News Classifier App!")
    st.write("""
        Explore different sections of the app using the sidebar navigation.
    """)

    # Beautiful words and pictures
    st.header("Discover the Power of Machine Learning in News Classification")
    st.image("https://via.placeholder.com/800x400", use_column_width=True)
    st.markdown("""
        Machine learning models can classify news articles into various categories,
        helping users to quickly navigate through vast amounts of information.
        
        Explore the "Prediction" section to see it in action!
    """)

def show_information_page():
    st.title("Information")
    st.info("General information about the app and its functionality.")

    # Example content
    st.markdown("""
        This app demonstrates the use of machine learning models for news classification.
        
        - Choose different models to predict the category of a news article.
        - Explore how text is processed and classified in real-time.
    """)
    st.image("https://via.placeholder.com/600x400", caption="Streamlit Logo", use_column_width=True)

def show_prediction_page():
    st.title("Prediction")
    st.info("Predict the category of a news article using machine learning models.")

    # Model selection
    model_choice = st.selectbox("Choose Model", list(model_paths.keys()))

    # Creating a text box for user input
    news_text = st.text_area("Enter Text", "Type Here")

    if st.button("Classify"):
        if vectorizer is not None:
            # Transforming user input with vectorizer
            vect_text = vectorizer.transform([news_text]).toarray()

            # Load the selected model
            model_path = model_paths[model_choice]
            if os.path.exists(model_path):
                predictor = joblib.load(open(model_path, "rb"))

                # Make prediction
                prediction = predictor.predict(vect_text)

                # Get the category name
                category_name = category_mapping.get(prediction[0], "Unknown category")

                # When model has successfully run, will print prediction
                st.success(f"Text Categorized as: {category_name}")
            else:
                st.error(f"Model file not found at {model_path}")
        else:
            st.error("Vectorizer could not be loaded. Classification cannot proceed.")

# Required to let Streamlit instantiate our web app
if __name__ == '__main__':
    main()





# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Print current working directory for debugging
#st.write("Current working directory:", os.getcwd())

# Function to load the vectorizer
def load_vectorizer(vectorizer_path):
    if os.path.exists(vectorizer_path):
        return joblib.load(vectorizer_path)
    else:
        st.error(f"Vectorizer file not found at {vectorizer_path}")
        return None

# Define absolute paths
base_path = os.path.dirname(__file__)
vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")

# Load the vectorizer
vectorizer = load_vectorizer(vectorizer_path)

# List of available models
model_paths = {
    "Logistic Regression": os.path.join(base_path, "logistic_regression_model.pkl"),
    "Naive Bayes": os.path.join(base_path, "naive_bayes_model.pkl"),
    "Support Vector Machine": os.path.join(base_path, "support_vector_machine_model.pkl")
}

# Category mapping (Example: Replace with your actual categories)
category_mapping = {
    0: "Business",
    1: "Education",
    2: "Entertainment",
    3: "Sports",
    4: "Technology"
}

# Main function to build the Streamlit app
def main():
    """News Classifier App with Streamlit"""
    
    # Creates a main title and subheader on your page
    st.title("News Classifier")
    st.subheader("Analysing news articles")
    
    # Creating sidebar with selection box
    options = ["Prediction", "Home"]
    selection = st.sidebar.selectbox("Choose Option", options)
    
    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        
        # Model selection
        model_choice = st.selectbox("Choose Model", list(model_paths.keys()))
        
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")
        
        if st.button("Classify"):
            if vectorizer is not None:
                # Transforming user input with vectorizer
                vect_text = vectorizer.transform([news_text]).toarray()
                
                # Load the selected model
                model_path = model_paths[model_choice]
                if os.path.exists(model_path):
                    predictor = joblib.load(open(model_path, "rb"))
                    
                    # Make prediction
                    prediction = predictor.predict(vect_text)
                    
                    # Get the category name
                    category_name = category_mapping.get(prediction[0], "Unknown category")
                    
                    # When model has successfully run, will print prediction
                    st.success(f"Text Categorized as: {category_name}")
                else:
                    st.error(f"Model file not found at {model_path}")
            else:
                st.error("Vectorizer could not be loaded. Classification cannot proceed.")

# Required to let Streamlit instantiate our web app
if __name__ == '__main__':
    main()