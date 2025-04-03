import re

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
