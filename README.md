## 6. Streamlit<a class="anchor" id="streamlit"></a>

### Why Streamlit?

We chose to deploy our application using Streamlit due to its simplicity and effectiveness in creating interactive web applications directly from Python scripts. Streamlit allows us to quickly build and deploy our machine learning models with minimal setup, enabling us to focus more on data analysis and model development.

### Deployment Steps

#### Steps Taken

1. **Model Development and Pickling**: We developed and trained multiple machine learning models including Logistic Regression, Naive Bayes, Support Vector Machine (SVM), Random Forest, and Gradient Boosting. Each model was serialized using pickle for easy deployment.

2. **Streamlit Integration**: We integrated these pickled models into a Streamlit web application (`base_app.py`). This application allows users to input text and receive predictions on the category of news articles.

3. **Dependency Management**: We listed all necessary dependencies in `requirements.txt` to ensure smooth installation on different environments.

#### Accessing Our App

To access our Streamlit web application:

1. **Local Deployment**:
   - Clone the repository to your local machine.
   - Navigate to the `2401FTDS_Classification_Project/Streamlit/` directory.
   - Install necessary libraries:
     ```bash
     pip install -U streamlit numpy pandas scikit-learn
     ```
   - Run the Streamlit app:
     ```bash
     streamlit run base_app.py
     ```
   - View the app in your browser at `http://localhost:8501`.

### File Description

Within the Streamlit directory (`2401FTDS_Classification_Project/Streamlit/`), the key files include:

- `base_app.py`: Definition of the Streamlit application with integrated models for predicting news article categories.
- `gradient_boosting_model.pkl`, `logistic_regression_model.pkl`, `naive_bayes_model.pkl`, `random_forest_model.pkl`, `support_vector_machine_model.pkl`: Pickled models serialized from our machine learning model development.
- `tfidf_vectorizer.pkl`: Serialized TF-IDF vectorizer used for text preprocessing.

#### Running the Streamlit Web App Locally

Ensure you have Python and necessary libraries installed. Navigate to the directory containing `base_app.py` and run
