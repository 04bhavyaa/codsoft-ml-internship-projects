from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from feature_extraction import extract_tfidf_features
import joblib
import pandas as pd

# Function to train the model
def train_model(file_path):
    data = pd.read_csv(file_path, sep="::", header=None, names=["ID", "Title", "Genre", "Description"])
    descriptions = data['Description']
    genres = data['Genre']
    
    # Extract TF-IDF features
    tfidf_matrix = extract_tfidf_features(descriptions)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, genres, test_size=0.2, random_state=42)
    
    # Train a Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save the trained model
    joblib.dump(model, 'models/trained_model.pkl')

# Example usage
if __name__ == "__main__":
    train_model('data/train_data.txt')
