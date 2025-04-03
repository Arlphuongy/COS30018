import os
import joblib
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Prefetch, FusionQuery, Fusion, SparseVector
from sentence_transformers import SentenceTransformer
import torch
import re
import dotenv
import unicodedata
from typing import List, Any

from retrieval_utils import preprocess_text

# For safer pickle loading, make this function available at module level
def dummy_preprocess_text(text):
    """Dummy function with same name as what's in pickle file"""
    return preprocess_text(text)

# Make the function available under the expected name for the pickle
globals()['preprocess_text'] = preprocess_text

def query_embedding_huggingface(sentences, model) -> np.ndarray:
    """
    Create embeddings using Hugging Face models
    
    Args:
        sentences: List of texts to embed
        model: Model path or instance to use for embedding
        
    Returns:
        numpy.ndarray: Embeddings matrix
    """
    # If model is already a SentenceTransformer instance, use it directly
    if not isinstance(model, str) and hasattr(model, 'encode'):
        embeddings = model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    # Otherwise load the model from the path
    else:
        from sentence_transformers import SentenceTransformer
        model_instance = SentenceTransformer(model, trust_remote_code=True)
        embeddings = model_instance.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    return embeddings

# Load environment variables from .env file
dotenv.load_dotenv(".env.local")

# Qdrant configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')  # Using COLLECTION_NAME for full_rag_legal_docs
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
VECTORIZER_PATH = os.getenv('VECTORIZER_PATH', 'tfidf_vectorizer.pkl')

# Define if reranking should be enabled by default
ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'True').lower() in ('true', '1', 't')

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

# Load model
model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=os.getenv("CACHE_FOLDER", "./cache"), trust_remote_code=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define a set of common Vietnamese stopwords
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'cho', 'là', 'để', 'trong', 'với', 'các', 'có', 'được', 
    'tại', 'về', 'từ', 'theo', 'đến', 'không', 'những', 'này', 'đó', 'khi'
}

# Try to load the TF-IDF vectorizer with error handling
try:
    # Load the TF-IDF vectorizer
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loaded TF-IDF vectorizer from {VECTORIZER_PATH}")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    print("Creating a dummy vectorizer as fallback")
    
    # Create a simple fallback vectorizer if loading fails
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True
    )
    # Fit on a minimal corpus just to initialize
    vectorizer.fit(["dummy text for initialization"])

# Initialize reranker (lazy loading)
_gemini_reranker = None

def get_lazy_reranker():
    """Lazy loading of Gemini reranker to avoid unnecessary initialization"""
    global _gemini_reranker
    if _gemini_reranker is None and os.getenv('ENABLE_GEMINI_RERANKING', 'True').lower() in ('true', '1', 't'):
        # Import here to avoid circular imports
        try:
            from llm_reranker import get_reranker
            _gemini_reranker = get_reranker()
        except ImportError:
            print("Warning: llm_reranker module not found, reranking disabled")
            return None
    return _gemini_reranker

# Define a set of common Vietnamese stopwords
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'cho', 'là', 'để', 'trong', 'với', 'các', 'có', 'được', 
    'tại', 'về', 'từ', 'theo', 'đến', 'không', 'những', 'này', 'đó', 'khi'
}


# Extract query entities function
def extract_query_entities(query):
    """
    Extract various entity types from queries with improved handling for Vietnamese text.
    Handles both with and without diacritics, different formats, and abbreviations.
    """
    # Ensure we're working with normalized query text
    query = query.lower()
    
    entities = {}
    
    # Find Điều references with enhanced patterns
    dieu_patterns = [
        r'điều\s+(\d+)',  # Standard format eg. Điều 4
        r'dieu\s+(\d+)',  # Without diacritics
        r'điều\s+(\d+)\s*[,\.]\s*điều\s+(\d+)',  # Multiple, eg. Điều 2, Điều 3
        r'dieu\s+(\d+)\s*[,\.]\s*dieu\s+(\d+)',  # Multiple without diacritics
        r'điều\s*(\d+)[-–]\s*(\d+)',  # Range, eg. Điều 4-6
        r'dieu\s*(\d+)[-–]\s*(\d+)',  # Range without diacritics
    ]
    
    dieu_matches = []
    for pattern in dieu_patterns:
        matches = re.findall(pattern, query)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    dieu_matches.extend(list(match))
                else:
                    dieu_matches.append(match)
    
    if dieu_matches:
        entities['dieu'] = list(set(dieu_matches))  # Remove duplicates
    
    # Find Thông tư references with enhanced patterns
    tt_patterns = [
        r'thông\s*tư\s+(?:số\s+)?(\d+)',  # Standard, eg. Thông tư 67
        r'thong\s*tu\s+(?:số\s+)?(\d+)',  # Without diacritics
        r'thông\s*tư\s+(?:so\s+)?(\d+)',  # Mixed diacritics
        r'tt\s*(\d+)',  # Abbreviated, eg. TT67
        r'thông\s*tư\s+(?:số\s+)?(\d+)/\d+/[\w-]+',  # Full ref with year/code
        r'thong\s*tu\s+(?:số\s+)?(\d+)/\d+/[\w-]+',  # Full ref without diacritics
    ]
    
    tt_matches = []
    for pattern in tt_patterns:
        matches = re.findall(pattern, query)
        for match in matches:
            if isinstance(match, tuple):
                tt_matches.append(match[0])  # Just take the number part
            else:
                tt_matches.append(match)
    
    if tt_matches:
        entities['thong_tu'] = [f"tt{num}" for num in set(tt_matches)]  # Remove duplicates
    
    # Find Nghị định references with enhanced patterns
    nd_patterns = [
        r'nghị\s*định\s+(?:số\s+)?(\d+)',  # Standard, eg. Nghị định 46
        r'nghi\s*dinh\s+(?:số\s+)?(\d+)',  # Without diacritics
        r'nghị\s*định\s+(?:so\s+)?(\d+)',  # Mixed diacritics
        r'nd\s*(\d+)',  # Abbreviated, eg. ND46
        r'nghị\s*định\s+(?:số\s+)?(\d+)/\d+/[\w-]+',  # Full ref with year/code, eg. Nghị định 46/2023/NĐ-CP
        r'nghi\s*dinh\s+(?:số\s+)?(\d+)/\d+/[\w-]+',  # Full ref without diacritics
    ]
    
    nd_matches = []
    for pattern in nd_patterns:
        matches = re.findall(pattern, query)
        for match in matches:
            if isinstance(match, tuple):
                nd_matches.append(match[0])
            else:
                nd_matches.append(match)
    
    if nd_matches:
        entities['nghi_dinh'] = [f"nd{num}" for num in set(nd_matches)]
    
    return entities

# def basic_entity_filter(results, entities):
#     """Filter results to match entity requirements."""
#     if not entities:
#         return results
    
#     filtered = []
    
#     for result in results:
#         try:
#             payload = result.payload
            
#             # Safely handle potential None or missing values
#             headline = str(payload.get('article_headline', '')).lower()
#             doc_name = str(payload.get('document_name', '')).lower()
#             reference_id = str(payload.get('reference_id', '')).lower()
            
#             # Extract "điều" number
#             dieu_match = re.search(r'điều\s+(\d+)', headline)
#             dieu_num = dieu_match.group(1) if dieu_match else None
            
#             # Match criteria
#             is_match = False
            
#             # Match article numbers
#             if 'dieu' in entities and dieu_num and dieu_num in entities['dieu']:
#                 is_match = True
                
#             # Match document types
#             if not is_match:
#                 if 'thong_tu' in entities and any(tt[2:] in doc_name for tt in entities['thong_tu']):
#                     is_match = True
            
#             if is_match:
#                 filtered.append(result)
        
#         except Exception as e:
#             print(f"Error in filtering: {e}")
#             filtered.append(result)  # Append original result if filtering fails
    
#     return filtered if filtered else results

def basic_entity_filter(results, entities):
    """Filter results to match entity requirements."""
    if not entities:
        return results
    
    filtered = []
    
    for result in results:
        try:
            payload = result.payload
            
            # Safely handle potential None or missing values by ensuring they're strings
            headline = str(payload.get('article_headline', '') or '') 
            doc_name = str(payload.get('document_name', '') or '')
            reference_id = str(payload.get('reference_id', '') or '')
            
            # Now we can safely call lower() on them
            headline = headline.lower()
            doc_name = doc_name.lower()
            reference_id = reference_id.lower()
            
            # Extract "điều" number
            dieu_match = re.search(r'điều\s+(\d+)', headline)
            dieu_num = dieu_match.group(1) if dieu_match else None
            
            # Match criteria
            is_match = False
            
            # Match article numbers
            if 'dieu' in entities and dieu_num and dieu_num in entities['dieu']:
                is_match = True
                
            # Match document types
            if not is_match:
                if 'thong_tu' in entities and any(tt[2:] in doc_name for tt in entities['thong_tu']):
                    is_match = True
            
            if is_match:
                filtered.append(result)
        
        except Exception as e:
            print(f"Error in filtering: {e}")
            filtered.append(result)  # Append original result if filtering fails
    
    return filtered if filtered else results

# def post_process_results(results, query, limit=5):
#     """
#     Process results based on the query.
#     """
#     if not results:
#         return []
    
#     entities = extract_query_entities(query)
    
#     # Apply entity filtering first
#     filtered_results = basic_entity_filter(results, entities)
#     if not filtered_results:
#         filtered_results = results
    
#     processed_results = []
#     seen_content = set()
    
#     for result in filtered_results:
#         try:
#             payload = result.payload
#             content = payload.get('content', '') or ''
            
#             # Skip duplicate content
#             content_sig = content[:100].strip() if isinstance(content, str) else ''
#             if content_sig in seen_content:
#                 continue
#             seen_content.add(content_sig)
            
#             headline = payload.get('article_headline', '') or ''
#             doc_name = payload.get('document_name', '') or ''
            
#             # Safe conversion to lowercase
#             headline = headline.lower() if isinstance(headline, str) else ''
#             doc_name = doc_name.lower() if isinstance(doc_name, str) else ''
            
#             # Entity-based boosting
#             boost_factor = 1.0
            
#             # Match article number
#             if 'dieu' in entities:
#                 dieu_match = re.search(r'điều\s+(\d+)(?!\d)', headline)
#                 if dieu_match and dieu_match.group(1) in entities['dieu']:
#                     boost_factor *= 2.0
            
#             # Match document types
#             if 'thong_tu' in entities and any(tt[2:] in doc_name for tt in entities['thong_tu']):
#                 boost_factor *= 2.0
                
#             if 'nghi_dinh' in entities and any(nd[2:] in doc_name for nd in entities['nghi_dinh']):
#                 boost_factor *= 2.0
            
#             # Apply boost to score
#             result.score *= boost_factor
#             processed_results.append(result)
        
#         except Exception as e:
#             print(f"Error in post-processing: {e}")
#             continue
    
#     processed_results.sort(key=lambda x: x.score, reverse=True)
    
#     # Normalize scores if needed
#     if processed_results:
#         max_score = max(r.score for r in processed_results)
#         if max_score > 1.0:
#             for r in processed_results:
#                 r.score = 0.5 + ((r.score / max_score) * 0.5)
    
#     return processed_results[:limit]

def post_process_results(results, query, limit=5):
    """
    Process results based on the query.
    """
    if not results:
        return []
    
    entities = extract_query_entities(query)
    
    # Apply entity filtering first
    filtered_results = basic_entity_filter(results, entities)
    if not filtered_results:
        filtered_results = results
    
    processed_results = []
    seen_content = set()
    
    for result in filtered_results:
        try:
            payload = result.payload
            content = payload.get('content', '') or ''
            
            # Skip duplicate content
            content_sig = content[:100].strip() if isinstance(content, str) else ''
            if content_sig in seen_content:
                continue
            seen_content.add(content_sig)
            
            # Safely get and convert text fields to strings
            headline = str(payload.get('article_headline', '') or '')
            doc_name = str(payload.get('document_name', '') or '')
            
            # Entity-based boosting
            boost_factor = 1.0
            
            # Match article number
            if 'dieu' in entities:
                dieu_match = re.search(r'điều\s+(\d+)(?!\d)', headline.lower())
                if dieu_match and dieu_match.group(1) in entities['dieu']:
                    boost_factor *= 2.0
            
            # Match document types
            if 'thong_tu' in entities and any(tt[2:] in doc_name.lower() for tt in entities['thong_tu']):
                boost_factor *= 2.0
                
            if 'nghi_dinh' in entities and any(nd[2:] in doc_name.lower() for nd in entities['nghi_dinh']):
                boost_factor *= 2.0
            
            # Apply boost to score
            result.score *= boost_factor
            processed_results.append(result)
        
        except Exception as e:
            print(f"Error in post-processing: {e}")
            processed_results.append(result)  # Preserve the result even if boost fails
    
    processed_results.sort(key=lambda x: x.score, reverse=True)
    
    # Normalize scores if needed
    if processed_results:
        max_score = max(r.score for r in processed_results)
        if max_score > 1.0:
            for r in processed_results:
                r.score = 0.5 + ((r.score / max_score) * 0.5)
    
    return processed_results[:limit]

def normalize_query(query):
    """
    Normalize Vietnamese queries to handle diacritics and different notation formats
    for legal document references.
    """
    # First, handle case sensitivity
    query = query.lower()
    
    # Standardize common abbreviations and notations
    # Convert TT format to Thông tư
    query = re.sub(r'\btt(\d+)\b', r'thông tư \1', query)
    # Convert TT with full reference format
    query = re.sub(r'\btt(\d+)/(\d+)/([a-z0-9-]+)\b', r'thông tư \1/\2/\3', query, flags=re.IGNORECASE)
    
    # Convert ND format to Nghị định
    query = re.sub(r'\bnd(\d+)\b', r'nghị định \1', query)
    # Convert ND with full reference format
    query = re.sub(r'\bnd(\d+)/(\d+)/([a-z0-9-]+)\b', r'nghị định \1/\2/\3', query, flags=re.IGNORECASE)
    
    # Handle diacritics - normalize only if input has no diacritics
    # This helps when someone types without diacritics but means same thing
    if not any(c for c in query if unicodedata.combining(c)):
        # Map common non-diacritic to diacritic terms
        replacements = {
            r'\bdieu\b': 'điều',
            r'\bthong tu\b': 'thông tư',
            r'\bnghi dinh\b': 'nghị định',
            r'\bphap lenh\b': 'pháp lệnh',
            r'\bluat\b': 'luật'
        }
        
        for pattern, replacement in replacements.items():
            query = re.sub(pattern, replacement, query)
    
    # Make sure numeric references have proper spacing
    query = re.sub(r'điều(\d+)', r'điều \1', query)
    query = re.sub(r'thông tư(\d+)', r'thông tư \1', query)
    query = re.sub(r'nghị định(\d+)', r'nghị định \1', query)
    
    # Remove extra whitespace
    query = ' '.join(query.split())
    
    return query

def search_hybrid_similar(text_query, collection_name, top_k=10, model=None):
    """Improved hybrid search with balanced sparse-dense combination"""
    entities = extract_query_entities(text_query)
    
    # Create sparse vector with enhanced preprocessing
    preprocessed_query = preprocess_text(text_query)
    sparse_matrix = vectorizer.transform([preprocessed_query])
    coo = sparse_matrix.tocoo()
    
    # Create sparse vector only with significant terms
    significance_threshold = 0.1  # Adjust this threshold as needed
    significant_mask = coo.data > significance_threshold
    
    sparse_vector = SparseVector(
        indices=[int(i) for i in coo.col[significant_mask].tolist()],
        values=[float(v) for v in coo.data[significant_mask].tolist()]
    )
    
    # Generate dense vector
    dense_vector = model.encode(
        text_query,
        convert_to_numpy=True,
        show_progress_bar=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).tolist()
    
    # Adjust search strategy based on query type
    if entities and any('dieu' in k or 'thong_tu' in k for k in entities.keys()):
        # For specific legal reference queries, give more weight to sparse search
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2),
                Prefetch(query=dense_vector, using="dense", limit=top_k)
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k
        )
    else:
        # For general queries, balance between sparse and dense
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=top_k),
                Prefetch(query=sparse_vector, using="sparse", limit=top_k)
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k
        )

    # Convert results to DataFrame
    data = []
    if hasattr(results, 'points'):
        points = results.points
    else:
        points = results
        
    for point in points:
        item = {
            "id": point.id,
            "score": point.score
        }
        if hasattr(point, 'payload'):
            for key, value in point.payload.items():
                item[key] = value
        data.append(item)
    
    return pd.DataFrame(data)


def search_legal_documents(query, limit=3, use_reranking=None):
    """
    Main search function for legal documents using vector search.
    
    Args:
        query: User query
        limit: Maximum number of results to return
        use_reranking: Whether to apply reranking (None = use environment setting)
    
    Returns:
        List of most relevant documents
    """
    try:
        # First, normalize the query to handle different formats
        original_query = query
        query = normalize_query(query)
        
        print(f"Searching for: '{original_query}'")
        if original_query != query:
            print(f"Normalized query: '{query}'")
        
        # Extract entities and keywords from normalized query
        entities = extract_query_entities(query)
        
        # Check if we should apply reranking
        should_rerank = use_reranking
        if use_reranking is None:  # If not explicitly specified, use env setting
            should_rerank = ENABLE_RERANKING
            
        if entities:
            entity_str = ", ".join([f"{k}: {v}" for k, v in entities.items()])
            print(f"Extracted entities: {entity_str}")
        
        # For all queries, get enough initial results
        search_limit = max(30, limit * 10)
        
        # Use vector search
        retrieved_df = search_hybrid_similar(
            query, 
            collection_name=COLLECTION_NAME,  # Make sure this is using the correct collection
            top_k=search_limit,
            model=model
        )

        # After getting initial results
        if 'thong_tu' in entities and 'dieu' in entities:
            # Sort results to specifically prioritize matches for "Điều X in Thông tư Y"
            retrieved_df = retrieved_df.sort_values(by=['score'], ascending=False)
            
            # Move exact matches to the top
            for idx, row in retrieved_df.iterrows():
                doc_name = str(row.get('document_name', '')).lower()
                headline = str(row.get('article_headline', '')).lower()
                is_exact_match = False
                
                # Check if this is the exact article we're looking for
                if any(f"thông tư {tt[2:]}" in doc_name for tt in entities['thong_tu']):
                    dieu_match = re.search(r'điều\s+(\d+)', headline)
                    if dieu_match and dieu_match.group(1) in entities['dieu']:
                        is_exact_match = True
                
                if is_exact_match:
                    # Move this row to the top
                    retrieved_df = pd.concat([retrieved_df.iloc[[idx]], retrieved_df.drop(idx)]).reset_index(drop=True)
                    break

        print("Top 10 initial results:")
        for i, row in retrieved_df.head(10).iterrows():
            print(f"{i+1}. {row.get('document_name', '')} - {row.get('article_headline', '')} (Score: {row.get('score', 0)})")

        if retrieved_df.empty:
            print("No results found.")
            return []
            
        print(f"Initial search returned {len(retrieved_df)} results")
        
        # Convert DataFrame to list of objects for post-processing
        from types import SimpleNamespace
        
        results = []
        for _, row in retrieved_df.iterrows():
            obj = SimpleNamespace()
            obj.id = row.get('id')
            obj.score = row.get('score', 0)
            obj.payload = {k: v for k, v in row.items() if k not in ['id', 'score']}
            results.append(obj)
        
        # Apply post-processing 
        filtered_results = post_process_results(results, query, limit * 3)
        
        # Apply reranking if enabled and we have multiple results
        if should_rerank and len(filtered_results) > 1:
            print("Applying LLM reranking...")
            try:
                reranker = get_lazy_reranker()
                if reranker:
                    filtered_results = reranker.rerank(query, filtered_results, limit)
            except Exception as e:
                print(f"Reranking failed: {str(e)}")
        
        # Trim to requested limit
        filtered_results = filtered_results[:limit]
        
        # Add detailed debug information about document metadata
        print("=== DEBUG: Final Results ===")
        for i, result in enumerate(filtered_results):
            print(f"Result #{i+1}:")
            document_name = result.payload.get('document_name', '')
            article_headline = result.payload.get('article_headline', '')
            content_preview = result.payload.get('content', '')[:50] if result.payload.get('content') else ''
            
            print(f"  From: {document_name}")
            print(f"  Title: {article_headline}")
            print(f"  Score: {result.score:.4f}")
            print(f"  Content starts: {content_preview}...")
            print("-" * 40)
        
        print(f"Returning {len(filtered_results)} most relevant results")
        
        return filtered_results
    
    except Exception as e:
        print(f"Error during search: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Test query
    #query = "Nếu muốn mở một đại lý bảo hiểm thì vốn điều lệ là bao nhiêu?"
    #query = "dieu 2 trong thong tu 67 la gi"
    query = "dieu 81 trong nghi dinh 46 la gi"
    
    # Print environment configuration
    print(f"Using collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    
    # Run search
    results = search_legal_documents(query, limit=5)
