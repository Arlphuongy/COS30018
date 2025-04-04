import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from typing import List
from qdrant_client import QdrantClient
import sys
import importlib.util

# Define the module where preprocess_text will live
module_name = "retrieval_utils"

# Create a new module for the preprocess_text function
spec = importlib.util.spec_from_loader(module_name, loader=None)
retrieval_utils = importlib.util.module_from_spec(spec)
sys.modules[module_name] = retrieval_utils

# Define stopwords as a list instead of a set
VIETNAMESE_STOPWORDS = [
    'và', 'của', 'cho', 'là', 'để', 'trong', 'với', 'các', 'có', 'được', 
    'tại', 'về', 'từ', 'theo', 'đến', 'không', 'những', 'này', 'đó', 'khi',
    'gi', 'gì', 'la', 'là', 'ma', 'mà', 'the', 'thế', 'như', 'nhu'
]

# Add the preprocess_text function to the dedicated module
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

# Add the function to our dedicated module
retrieval_utils.preprocess_text = preprocess_text

def train_vectorizer(texts: List[str], output_path: str = 'tfidf_vectorizer.pkl'):
    """Train and save the TF-IDF vectorizer"""
    print(f"Training on {len(texts)} documents...")
    
    # Create vectorizer with improved settings, using the function from our module
    vectorizer = TfidfVectorizer(
        preprocessor=retrieval_utils.preprocess_text,  # Use from module
        stop_words=VIETNAMESE_STOPWORDS,
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
    
    # Make a backup of existing vectorizer if it exists
    if os.path.exists(output_path):
        backup_path = output_path + ".backup"
        print(f"Creating backup of existing vectorizer at {backup_path}")
        import shutil
        shutil.copy2(output_path, backup_path)
    
    # Save vectorizer
    joblib.dump(vectorizer, output_path)
    print(f"Vectorizer saved to {output_path}")
    
    # Also save a small Python file with the preprocess_text function for reference
    with open("retrieval_utils.py", "w", encoding="utf-8") as f:
        f.write("""import re

def preprocess_text(text: str) -> str:
    \"\"\"Preprocess text for TF-IDF vectorization\"\"\"
    # Normalize Vietnamese text
    text = text.lower()
    
    # Special handling for legal document references
    text = re.sub(r'tt(\d+)', r'thông tư \\1', text)
    text = re.sub(r'đ(\d+)', r'điều \\1', text)
    text = re.sub(r'điều(\d+)', r'điều \\1', text)
    text = re.sub(r'thông tư(\d+)', r'thông tư \\1', text)
    
    # Handle number sequences
    text = re.sub(r'(\d+)', r' \\1 ', text)
    
    return text
""")
    print("Also saved retrieval_utils.py with preprocess_text function")
    
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

def update_retrieval_file():
    """Update retrieval.py to use preprocess_text from the module"""
    try:
        # Check if retrieval.py exists
        if not os.path.exists("retrieval.py"):
            print("Warning: retrieval.py not found, skipping update")
            return
            
        # Read the current file
        with open("retrieval.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Only modify if it's not already using retrieval_utils
        if "from retrieval_utils import preprocess_text" not in content:
            # Add the import at the top after existing imports
            import_line = "from retrieval_utils import preprocess_text\n"
            
            # Find a good place to insert the import - after other imports
            import_section_end = content.find("# Load environment variables")
            if import_section_end == -1:
                import_section_end = content.find("# Define if reranking should be enabled")
                
            if import_section_end > 0:
                modified_content = content[:import_section_end] + import_line + content[import_section_end:]
                
                # Create backup
                import shutil
                shutil.copy2("retrieval.py", "retrieval.py.backup")
                
                # Write modified file
                with open("retrieval.py", "w", encoding="utf-8") as f:
                    f.write(modified_content)
                    
                print("Updated retrieval.py to import preprocess_text from retrieval_utils")
            else:
                print("Couldn't find a good place to insert import in retrieval.py")
    except Exception as e:
        print(f"Error updating retrieval.py: {e}")

if __name__ == "__main__":
    # Load environment variables
    import dotenv
    dotenv.load_dotenv(".env.local")
    
    try:
        # Get training texts
        texts = get_training_texts()
        
        # Train and save vectorizer
        vectorizer = train_vectorizer(texts)
        print("\nTraining completed successfully!")
        
        # Update retrieval.py
        update_retrieval_file()
        
        # Optional: Test the vectorizer
        test_query = "điều 2 trong thông tư 67"
        print("\nTesting vectorizer with query:", test_query)
        vector = vectorizer.transform([test_query])
        feature_names = vectorizer.get_feature_names_out()
        print("Top terms and weights:")
        for idx, weight in zip(vector.indices, vector.data):
            print(f"  {feature_names[idx]}: {weight:.4f}")
            
        print("\n======== NEXT STEPS ========")
        print("1. Run your application: gradio app.py")
        print("If you encounter any issues, try these troubleshooting steps:")
        print("2. Make sure retrieval.py is importing preprocess_text from retrieval_utils")
        print("3. Verify that retrieval_utils.py exists in your project directory")
        
    except Exception as e:
        print(f"Error during training: {e}")