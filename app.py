# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

# App Configuration
st.set_page_config(
    page_title="Amazon Alexa Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title
st.title("Amazon Alexa Sentiment Analysis")
st.markdown("### An NLP Approach to Feedback Classification")
st.write("This app uses machine learning to classify feedback as **Positive** or **Negative**.")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('data/amazon_alexa.tsv', sep='\t')
    return data

data = load_data()

# Display dataset details
if st.checkbox("Show dataset details"):
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("### Dataset Shape")
    st.write(data.shape)

    st.write("### Column Details")
    st.write(data.columns)

    st.write("### Missing Values")
    st.write(data.isnull().sum())

# Ensure all values in the 'verified_reviews' column are strings
data['verified_reviews'] = data['verified_reviews'].astype(str).fillna("")

# Calculate review length
data['review_length'] = data['verified_reviews'].apply(len)

# Data visualizations
if st.checkbox("Show Data Visualizations"):
    # Feedback Distribution
    st.write("### Feedback Distribution")
    feedback_dist = data['feedback'].value_counts()
    fig1 = px.pie(
        values=feedback_dist.values,
        names=feedback_dist.index,
        title="Feedback Distribution",
        hole=0.3,  # For a donut-style pie chart
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig1)

    # Review Length Distribution
    st.write("### Review Length Distribution")
    # Ensure all values in 'verified_reviews' are strings
    data['verified_reviews'] = data['verified_reviews'].astype(str).fillna("")
    data['review_length'] = data['verified_reviews'].apply(len)
    fig2 = px.histogram(
        data,
        x='review_length',
        nbins=50,
        title="Review Length Distribution",
        labels={'review_length': 'Review Length'},
        color_discrete_sequence=['blue']
    )
    fig2.update_layout(
        xaxis_title="Review Length",
        yaxis_title="Frequency",
        bargap=0.2
    )
    st.plotly_chart(fig2)

# Word Cloud Visualization
if st.checkbox("Generate Word Cloud"):
    st.write("### Word Cloud for Reviews")
    # Combine all reviews into a single text
    reviews_text = " ".join(data['verified_reviews'].astype(str).tolist())
    # Generate the word cloud
    wordcloud = WordCloud(
        background_color='white',
        max_words=100,
        colormap='viridis',
        width=800,
        height=400
    ).generate(reviews_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


# Label encoding and oversampling
@st.cache_data
def preprocess_data(data):
    le = LabelEncoder()
    data['variation'] = le.fit_transform(data.variation)

    df_class_0 = data[data['feedback'] == 0]
    df_class_1 = data[data['feedback'] == 1]
    df_class_0_over = df_class_0.sample(len(df_class_1), replace=True, random_state=42)
    df_sampled = pd.concat([df_class_0_over, df_class_1], axis=0)
    return df_sampled

df_sampled = preprocess_data(data)

# Tokenize and pad sequences
@st.cache_data
def tokenize_pad(data):
    # Convert all reviews to strings and handle NaN values
    data['verified_reviews'] = data['verified_reviews'].astype(str).fillna("")
    
    tokenizer = Tokenizer(num_words=15212, lower=True, oov_token='UNK')
    tokenizer.fit_on_texts(data['verified_reviews'])
    sequences = tokenizer.texts_to_sequences(data['verified_reviews'])
    padded_sequences = pad_sequences(sequences, maxlen=80, padding='post')
    return tokenizer, padded_sequences

tokenizer, X_pad = tokenize_pad(df_sampled)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pad, df_sampled['feedback'], test_size=0.3, random_state=42)

# Apply SMOTE
@st.cache_data
def smote_resample(X, y):
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return X_resampled, y_resampled

X_resampled, y_resampled = smote_resample(X_train, y_train)

# Train models
@st.cache_resource
def train_random_forest(X_resampled, y_resampled):
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_resampled, y_resampled)
    return rf_classifier

rf_classifier = train_random_forest(X_resampled, y_resampled)

# Evaluate model
if st.checkbox("Evaluate model performance"):
    y_pred = rf_classifier.predict(X_test)

    st.write("### Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("### Accuracy Score")
    st.write(accuracy_score(y_test, y_pred))

if st.checkbox("Show Feature Importance"):
    # Ensure the model and feature names are defined
    if 'rf_classifier' in locals() or 'rf_classifier' in globals():
        feature_importances = rf_classifier.feature_importances_
        # Replace with your actual feature names
        feature_names = X_resampled.columns if isinstance(X_resampled, pd.DataFrame) else [f"Feature {i}" for i in range(len(feature_importances))]

        top_features = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False).head(10)

        fig3 = px.bar(
            top_features,
            x='Feature',
            y='Importance',
            title="Top 10 Feature Importances",
            labels={'Feature': 'Feature Name', 'Importance': 'Importance Score'},
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig3)
    else:
        st.warning("Random Forest model not trained yet.")

if st.checkbox("Compare Models"):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).sort_values(by='Accuracy', ascending=False)
    st.write("### Model Comparison")
    st.dataframe(results_df)
    fig4 = px.bar(
        results_df,
        x=results_df.index,
        y='Accuracy',
        title="Model Accuracy Comparison",
        labels={"index": "Model", "Accuracy": "Accuracy Score"},
        color='Accuracy',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig4)


# Feedback classification
st.sidebar.header("User Feedback Classification")
user_feedback = st.sidebar.text_input("Enter your review here:")

if st.sidebar.button("Classify Feedback"):
    feedback_seq = tokenizer.texts_to_sequences([user_feedback])
    feedback_padded = pad_sequences(feedback_seq, maxlen=80, padding='post')
    prob = rf_classifier.predict_proba(feedback_padded)[0]
    sentiment = "Positive" if prob[1] > prob[0] else "Negative"
    confidence = max(prob) * 100

    st.sidebar.write(f"Sentiment: **{sentiment}**")
    st.sidebar.write(f"Confidence Score: **{confidence:.2f}%**")

# Model download
if st.checkbox("Download Model"):
    pickle.dump(rf_classifier, open("rf_model.pkl", "wb"))
    with open("rf_model.pkl", "rb") as f:
        st.download_button("Download Trained Model", f, file_name="rf_model.pkl")

# Footer
st.markdown("---")
st.markdown("Developed by [Nitheeshkumar R](https://github.com/nitheesh2509). Powered by Streamlit.")
