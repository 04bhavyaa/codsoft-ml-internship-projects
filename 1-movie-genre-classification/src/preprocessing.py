import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Download NLTK stopwords
nltk.download('stopwords')

# Function to load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path, sep="::", header=None, names=["ID", "Title", "Genre", "Description"])
    return data

# Function to clean the text (removing stopwords, lowercasing, etc.)
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_text = " ".join([word.lower() for word in words if word.lower() not in stop_words])
    return cleaned_text

# Function to preprocess the dataset
def preprocess_data(file_path):
    data = load_data(file_path)
    data['Description'] = data['Description'].apply(clean_text)
    return data

# Example usage
if __name__ == "__main__":
    data = preprocess_data('data/train_data.txt')
    print(data.head())
