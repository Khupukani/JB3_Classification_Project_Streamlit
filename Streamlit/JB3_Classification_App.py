import streamlit as st
import joblib
import os

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

# Function to check if the file exists and return the correct path
def get_file_path(relative_path):
    file_path = os.path.join(base_path, relative_path)
    if os.path.exists(file_path):
        return file_path
    else:
        st.error(f"File not found: {file_path}")
        return None

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
    st.title("The News Spot Classifier App!")
    st.write("""
        Explore different sections of the app using the sidebar navigation.
    """)

    # Section 1: Introduction
    st.header("Introduction")
    st.markdown("""
        In today's world, information flows faster than ever before. 
        This app uses machine learning to classify news articles into categories, 
        helping you navigate through vast amounts of information with ease.
    """)
    video1_path = get_file_path("Video1.mp4")
    if video1_path:
        st.video(video1_path)

    # Section 2: How It Works
    st.header("How It Works")
    st.markdown("""
        Our models analyze the text of news articles and predict which category they belong to. 
        Explore the "Prediction" section to see it in action!
    """)
    video2_path = get_file_path("Video2.mp4")
    if video2_path:
        st.video(video2_path)

    # Section 3: Benefits
    st.header("Benefits")
    st.markdown("""
        - Quickly find relevant news
        - Stay informed on topics that matter to you
        - Save time with automated categorization
    """)
    video3_path = get_file_path("Video3.mp4")
    if video3_path:
        st.video(video3_path)

    # Section 4: Real-Life Examples
    st.header("Real-Life Examples")
    st.markdown("""
        See how news classification can be applied in various industries:
        - **Business**: Analyzing market trends
        - **Education**: Enhancing learning materials
        - **Entertainment**: Curating media content
        - **Sports**: Tracking game statistics
        - **Technology**: Monitoring tech innovations
    """)
    video4_path = get_file_path("Video4.mp4")
    if video4_path:
        st.video(video4_path)

def show_information_page():
    st.title("Information")
    st.info("General information about the app and its functionality.")

    # Example content
    st.markdown("""
        This app demonstrates the use of machine learning models for news classification.
        
        - Choose different models to predict the category of a news article.
        - Explore how text is processed and classified in real-time.
    """)
    image_path = get_file_path("Picture2.JPG")
    if image_path:
        st.image(image_path, caption="Streamlit Logo", use_column_width=True)

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
