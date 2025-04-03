# llm_reranker.py
import os
import dotenv
import google.generativeai as genai
import json
import time
import re
import copy
from typing import List, Dict, Any, Optional

# Load environment variables
dotenv.load_dotenv(".env.local")
API_KEY = os.getenv('GEMINI_API_KEY')

class LLMReranker:
    """Reranker for Vietnamese legal documents using Large Language Models (Gemini)"""
    
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.0):
        """
        Initialize the LLM-based reranker
        
        Args:
            model_name: LLM model to use ("gemini-1.5-flash" recommended for reranking)
            temperature: Temperature for generation (lower is more deterministic)
        """
        print(f"Initializing LLMReranker with model: {model_name}")
        
        # Configure Gemini
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature}
        )
        self.model_name = model_name
        
        # Determine if we're using Gemini 1.5 or 2.0
        self.is_gemini_2 = "2.0" in model_name
        
        print(f"LLM Reranker initialized successfully using {model_name}")

    def rerank(self, query: str, documents: List[Any], top_k: int = 5) -> List[Any]:
        """
        Rerank documents using LLM
        
        Args:
            query: The search query
            documents: List of document objects with payload attributes
            top_k: Number of top results to return
            
        Returns:
            List of reranked document objects
        """
        if not documents:
            return []
            
        if len(documents) == 1:
            return documents  # No need to rerank a single document
        
        start_time = time.time()
        
        # Extract contents from documents
        document_contents = []
        for doc in documents:
            content = doc.payload.get('content', '')
            headline = doc.payload.get('article_headline', '')
            if headline:
                content = headline + "\n" + content
            document_contents.append(content)
        
        # Calculate relevance scores
        scores = self._calculate_relevance_scores(query, document_contents, documents)
        
        # Store original scores and ranks for reference
        original_scores = {doc.id: doc.score for doc in documents}
        original_ranks = {doc.id: i for i, doc in enumerate(documents)}
        
        # Create copies of documents with new scores
        scored_docs = []
        for doc, score in zip(documents, scores):
            # Create a shallow copy of the document (if possible)
            try:
                # Try to use copy method if available
                doc_copy = copy.copy(doc)
            except:
                # Fallback to the original document
                doc_copy = doc
                
            # Set the new score
            try:
                doc_copy.score = float(score)
            except ValueError:
                # If we can't modify the score, create a new object or use a wrapper
                print(f"Warning: Could not modify score for document {doc.id}")
                
            scored_docs.append(doc_copy)
        
        # Sort by new scores
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        
        # Calculate time taken
        elapsed = time.time() - start_time
        
        # Debug information
        if len(scored_docs) > 0:
            print(f"\nReranking results with {self.model_name} (took {elapsed:.2f}s):")
            for i, doc in enumerate(scored_docs[:top_k]):
                print(f"  Rank {i+1}: Score {doc.score:.4f} (was {original_scores[doc.id]:.4f} at position {original_ranks[doc.id]+1})")
                    
        return scored_docs[:top_k]
        
    def _calculate_relevance_scores(self, query: str, document_contents: List[str], documents: List[Any]) -> List[float]:
        """
        Calculate relevance scores between query and documents using LLM
        
        Args:
            query: The search query
            document_contents: List of document content strings
            documents: Original document objects (for metadata)
            
        Returns:
            List of relevance scores (0-10 scale, converted to 0-1 range)
        """
        scores = []
        
        # Determine batch size based on document count and length
        # Smaller batch for longer documents to avoid context limits
        avg_doc_length = sum(len(doc) for doc in document_contents) / len(document_contents)
        batch_size = 10 if avg_doc_length < 1000 else (5 if avg_doc_length < 5000 else 3)
        
        # Process in batches 
        for i in range(0, len(document_contents), batch_size):
            batch_docs = document_contents[i:i+batch_size]
            batch_metadata = documents[i:i+batch_size]
            batch_scores = self._score_batch(query, batch_docs, batch_metadata)
            scores.extend(batch_scores)
            
            # Throttle API calls if not on the last batch
            if i + batch_size < len(document_contents):
                time.sleep(0.5)
        
        # Normalize scores to 0-1 range
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                scores = [(score - min_score) / (max_score - min_score) for score in scores]
            else:
                # If all scores are the same, normalize to 0.5
                scores = [0.5 for _ in scores]
                
        return scores
    
    def _score_batch(self, query: str, documents: List[str], document_metadata: List[Any]) -> List[float]:
        """
        Score a batch of documents using LLM
        
        Args:
            query: The search query
            documents: List of document content strings
            document_metadata: Original document objects (for metadata)
            
        Returns:
            List of relevance scores
        """
        # Create prompt for scoring
        prompt = self._create_scoring_prompt(query, documents, document_metadata)
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Parse scores from the response
            scores = self._parse_scores(response.text, len(documents))
            
            # Log sample of response for debugging
            print(f"Sample LLM response: {response.text[:100]}...")
            
            return scores
        except Exception as e:
            print(f"Error during LLM scoring: {str(e)}")
            # Fallback: return neutral scores
            return [0.5 for _ in documents]
    
    def _create_scoring_prompt(self, query: str, documents: List[str], document_metadata: List[Any]) -> str:
        """
        Create a prompt for LLM to score documents
        
        Args:
            query: The search query
            documents: List of document content strings
            document_metadata: Original document objects (for metadata)
            
        Returns:
            Formatted prompt string
        """
        # Format the prompt with query and documents
        prompt = f"""You are an expert legal document search engine specialized in Vietnamese law.

Query: {query}

I need you to score the relevance of the following {len(documents)} documents to the query on a scale of 0 to 10 (where 10 is most relevant).

For each document, consider:
1. Direct answer relevance (0-5): Does it directly address what the query is asking about?
2. Information completeness (0-3): How complete is the information provided?
3. Context and specificity (0-2): Is the context appropriate and specific enough?

For Vietnamese legal documents, pay special attention to:
- Điều (Article) number references - if the query mentions Điều X, documents about that specific article should score higher
- Thông tư (Circular) references - if the query mentions a specific Thông tư, matching documents should score higher
- Legal terminology and concepts that directly match the query intent

Return ONLY a JSON array of scores like [8.5, 6.2, 9.0, ...] with no additional text or explanation.

Documents to score:
"""

        # Add each document with its metadata
        for i, (doc, metadata) in enumerate(zip(documents, document_metadata)):
            # Truncate content to avoid exceeding context limits
            # Gemini can handle much more context than previous models
            max_chars = 5000 if self.is_gemini_2 else 2500
            doc_preview = doc[:max_chars] + "..." if len(doc) > max_chars else doc
            
            document_name = metadata.payload.get('document_name', 'Unknown')
            article_headline = metadata.payload.get('article_headline', 'Unknown')
            
            prompt += f"\n--- Document {i+1} ---\n"
            prompt += f"Source: {document_name}\n"
            prompt += f"Title: {article_headline}\n"
            prompt += f"Content: {doc_preview}\n"
        
        return prompt
    
    def _parse_scores(self, response_text: str, expected_count: int) -> List[float]:
        """
        Parse scores from LLM's response
        
        Args:
            response_text: Text response from LLM
            expected_count: Expected number of scores
            
        Returns:
            List of parsed scores
        """
        try:
            # Try to find and parse JSON array
            response_text = response_text.strip()
            
            # Find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                scores = json.loads(json_str)
                
                # Validate scores
                if isinstance(scores, list) and len(scores) == expected_count:
                    # Convert all elements to float
                    scores = [float(score) for score in scores]
                    return scores
            
            # If we couldn't parse the JSON array properly, try extracting numbers
            numbers = []
            for line in response_text.split('\n'):
                if len(numbers) >= expected_count:
                    break
                # Try to find numbers in the line
                matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', line)
                for match in matches:
                    if len(numbers) < expected_count:
                        numbers.append(float(match))
            
            # If we found enough numbers, use them as scores
            if len(numbers) == expected_count:
                return numbers
                
            # If we still couldn't parse properly
            print(f"Failed to parse scores from response: {response_text}")
            return [5.0 for _ in range(expected_count)]  # Default neutral score
            
        except Exception as e:
            print(f"Error parsing scores: {str(e)}")
            print(f"Response was: {response_text}")
            return [5.0 for _ in range(expected_count)]  # Default neutral score


# Factory function to create reranker instance
def get_reranker(model_name=None):
    """
    Get a reranker instance with optional model override
    
    Args:
        model_name: Override the default model
        
    Returns:
        LLMReranker instance
    """
    # Get model name from environment or use default
    default_model = os.getenv('LLM_MODEL')
    model_name = model_name or default_model
    
    # Create and return reranker
    return LLMReranker(model_name=model_name)