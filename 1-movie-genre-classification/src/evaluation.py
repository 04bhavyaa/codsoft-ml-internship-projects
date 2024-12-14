import joblib
from sklearn.metrics import accuracy_score
from feature_extraction import extract_tfidf_features
import pandas as pd

# Function to evaluate the model
def evaluate_model(file_path, model_path='models/trained_model.pkl'):
    # Load the model
    model = joblib.load(model_path)
    
    # Load and preprocess the test data
    data = pd.read_csv(file_path, sep="::", header=None, names=["ID", "Title", "Genre", "Description"])
    descriptions = data['Description']
    genres = data['Genre']
    
    # Extract TF-IDF features for test data
    tfidf_matrix = extract_tfidf_features(descriptions)
    
    # Make predictions
    y_pred = model.predict(tfidf_matrix)
    
    # Evaluate the model
    print(f'Accuracy: {accuracy_score(genres, y_pred):.4f}')

# Example usage
if __name__ == "__main__":
    evaluate_model('data/test_data.txt')
