import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from typing import List
from qdrant_client import QdrantClient

# Define stopwords as a list instead of a set
VIETNAMESE_STOPWORDS = [
    'và', 'của', 'cho', 'là', 'để', 'trong', 'với', 'các', 'có', 'được', 
    'tại', 'về', 'từ', 'theo', 'đến', 'không', 'những', 'này', 'đó', 'khi',
    'gi', 'gì', 'la', 'là', 'ma', 'mà', 'the', 'thế', 'như', 'nhu'
]

def preprocess_text(text: str) -> str:
    """Preprocess text for TF-IDF vectorization"""
    # Normalize Vietnamese text
    text = text.lower()
    
    # Special handling for legal document references
    text = re.sub(r'tt(\d+)', r'thông tư \1', text)
    text = re.sub(r'đ(\d+)', r'điều \1', text)
    text = re.sub(r'điều(\d+)', r'điều \1', text)
    text = re.sub(r'thông tư(\d+)', r'thông tư \1', text)
    
    # Handle number sequences
    text = re.sub(r'(\d+)', r' \1 ', text)
    
    return text

def train_vectorizer(texts: List[str], output_path: str = 'tfidf_vectorizer.pkl'):
    """Train and save the TF-IDF vectorizer"""
    print(f"Training on {len(texts)} documents...")
    
    # Create vectorizer with improved settings
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        stop_words=VIETNAMESE_STOPWORDS,  # Now using a list
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95,  # Maximum document frequency
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    # Fit vectorizer
    print("Fitting vectorizer...")
    vectorizer.fit(texts)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Adjust weights for important terms
    feature_names = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_
    
    # Terms to boost
    boost_terms = {
        'điều': 2.0,
        'thông tư': 2.0,
        'nghị định': 2.0,
        'luật': 2.0,
        'pháp lệnh': 2.0
    }
    
    print("Adjusting weights for important terms...")
    # Apply boosting
    for term, boost in boost_terms.items():
        if term in feature_names:
            idx = list(feature_names).index(term)
            idf[idx] *= boost
            print(f"Boosted '{term}' by {boost}x")
    
    # Boost numeric terms
    num_boosted = 0
    for i, term in enumerate(feature_names):
        if re.match(r'^\d+$', term):
            idf[i] *= 1.5
            num_boosted += 1
    print(f"Boosted {num_boosted} numeric terms by 1.5x")
    
    vectorizer.idf_ = idf
    
    # Save vectorizer
    joblib.dump(vectorizer, output_path)
    print(f"Vectorizer saved to {output_path}")
    
    return vectorizer

def get_training_texts() -> List[str]:
    """Get texts from your Qdrant collection for training"""
    print("Connecting to Qdrant...")
    # Initialize Qdrant client
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    
    # Get all documents from collection
    collection_name = os.getenv('COLLECTION_NAME')
    print(f"Fetching documents from collection: {collection_name}")
    
    response = client.scroll(
        collection_name=collection_name,
        limit=10000,  # Adjust based on your collection size
        with_payload=True
    )
    
    # Extract texts from documents
    texts = []
    for point in response[0]:  # response[0] contains the points
        # Combine relevant fields for vectorization
        content = point.payload.get('content', '')
        headline = point.payload.get('article_headline', '')
        doc_name = point.payload.get('document_name', '')
        
        combined_text = f"{doc_name} {headline} {content}"
        texts.append(combined_text)
    
    print(f"Retrieved {len(texts)} documents")
    return texts

if __name__ == "__main__":
    # Load environment variables
    import dotenv
    dotenv.load_dotenv(".env.local")
    
    try:
        # Get training texts
        texts = get_training_texts()
        
        # Train and save vectorizer
        vectorizer = train_vectorizer(texts)
        print("Training completed successfully!")
        
        # Optional: Test the vectorizer
        test_query = "điều 2 trong thông tư 67"
        print("\nTesting vectorizer with query:", test_query)
        vector = vectorizer.transform([test_query])
        feature_names = vectorizer.get_feature_names_out()
        print("Top terms and weights:")
        for idx, weight in zip(vector.indices, vector.data):
            print(f"  {feature_names[idx]}: {weight:.4f}")
            
    except Exception as e:
        print(f"Error during training: {e}")