from sklearn.feature_extraction.text import TfidfVectorizer

# Function to extract TF-IDF features
def extract_tfidf_features(descriptions):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return tfidf_matrix

# Example usage
if __name__ == "__main__":
    from preprocessing import preprocess_data
    data = preprocess_data('data/train_data.txt')
    tfidf_matrix = extract_tfidf_features(data['Description'])
    print(tfidf_matrix.shape)
