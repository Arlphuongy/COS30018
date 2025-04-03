import json
import time
import numpy as np
from tqdm import tqdm
import traceback
import re
import os
import dotenv
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
from retrieval_utils import preprocess_text

dotenv.load_dotenv(".env.local")

class RAGEvaluator:
    """Enhanced evaluator for testing RAG implementation against hallucinations"""
    
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
    
    def __init__(self, 
                 embedding_model_name=EMBEDDING_MODEL,
                 hallucination_weights=None,
                 use_cache=True):
        """
        Initialize evaluation models and parameters
        
        Args:
            embedding_model_name: Model for measuring semantic similarity 
            hallucination_weights: Dict of weights for hallucination score calculation
            use_cache: If True, cache embeddings to avoid recomputation
        """
        # Set up embedding model
        try:
            self.embedding_model = SentenceTransformer(
                embedding_model_name, 
                cache_folder=os.getenv("CACHE_FOLDER", "./cache"), 
                trust_remote_code=True
            )
            print(f"Successfully loaded embedding model: {embedding_model_name}")
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            print("Using fallback model")
            # Fallback to basic model if the specified one fails
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # Default weights for hallucination score components
        self.hallucination_weights = hallucination_weights or {
            "semantic_similarity": 0.7,
            "citation_score": 0.3
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.hallucination_weights.values())
        for k in self.hallucination_weights:
            self.hallucination_weights[k] /= weight_sum
        
        self.use_cache = use_cache
        self.embedding_cache = {}
    
    @lru_cache(maxsize=1024)
    def _cached_encode(self, text):
        """Cache embeddings to avoid recomputation"""
        if not text:
            return np.zeros((384,))  # Default embedding dimension
        
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            # Return zero vector on error
            return np.zeros((384,))
    
    def load_test_data(self, file_path):
        """Load test questions and ground truths from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            print(f"Successfully loaded {len(test_data)} test questions from {file_path}")
            return test_data
        except Exception as e:
            print(f"Error loading test data from {file_path}: {str(e)}")
            traceback.print_exc()
            return []
    
    def save_results(self, results, file_path):
        """Save evaluation results to a JSON file with error handling"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved results to {file_path}")
        except Exception as e:
            print(f"Error saving results to {file_path}: {str(e)}")
            
            # Save to a backup file in case of error
            backup_path = f"{file_path}.backup"
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Saved backup results to {backup_path}")
            except Exception as backup_error:
                print(f"Failed to save backup results: {str(backup_error)}")
    
    def compute_semantic_similarity(self, response, ground_truth):
        """
        Compute semantic similarity between response and ground truth using embeddings
        
        Args:
            response: Generated response text
            ground_truth: Ground truth answer text
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        try:
            if not response or not ground_truth:
                return 0.0
            
            # Clean and truncate texts to prevent memory issues
            response = response[:10000].strip() if isinstance(response, str) else ""
            ground_truth = ground_truth[:10000].strip() if isinstance(ground_truth, str) else ""
            
            if not response or not ground_truth:
                return 0.0
                
            # Use cached encodings if available
            if self.use_cache:
                response_embedding = self._cached_encode(response)
                ground_truth_embedding = self._cached_encode(ground_truth)
            else:
                # Batch encode for efficiency
                texts = [response, ground_truth]
                embeddings = self.embedding_model.encode(
                    texts, 
                    convert_to_numpy=True, 
                    show_progress_bar=False
                )
                response_embedding = embeddings[0]
                ground_truth_embedding = embeddings[1]
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                [response_embedding], 
                [ground_truth_embedding]
            )[0][0]
            
            # Ensure result is between 0 and 1
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            print(f"Error computing semantic similarity: {str(e)}")
            return 0.0
    
    def extract_citations(self, response):
        """
        Extract citations from response text using regex patterns
        
        Args:
            response: Generated response text
            
        Returns:
            list: Extracted citations
        """
        if not response or not isinstance(response, str):
            return []
        
        # Enhanced pattern set for Vietnamese legal document citations
        citation_patterns = [
            # Thông tư (Circular) patterns
            r'[Tt]hông\s*tư\s+(?:số\s+)?(\d+)(?:/\d+/[A-Za-z0-9-]+)?',
            r'\bTT\s*(\d+)\b',
            
            # Nghị định (Decree) patterns
            r'[Nn]ghị\s*định\s+(?:số\s+)?(\d+)(?:/\d+/[A-Za-z0-9-]+)?',
            r'\bND\s*(\d+)\b',
            
            # Điều (Article) patterns
            r'[Đ|đ]iều\s+(\d+)',
            
            # Combined patterns
            r'[Tt]hông\s*tư\s+(?:số\s+)?(\d+).*?[Đ|đ]iều\s+(\d+)',
            r'[Nn]ghị\s*định\s+(?:số\s+)?(\d+).*?[Đ|đ]iều\s+(\d+)'
        ]
        
        citations = set()
        
        # Extract combined citations (document + article)
        try:
            tt_dieu_matches = re.findall(r'[Tt]hông\s*tư\s+(?:số\s+)?(\d+).*?[Đ|đ]iều\s+(\d+)', response)
            for match in tt_dieu_matches:
                if len(match) >= 2:
                    citations.add(f"Thông tư {match[0]}, Điều {match[1]}")
        except Exception as e:
            print(f"Error extracting Thông tư + Điều citations: {str(e)}")
        
        try:
            nd_dieu_matches = re.findall(r'[Nn]ghị\s*định\s+(?:số\s+)?(\d+).*?[Đ|đ]iều\s+(\d+)', response)
            for match in nd_dieu_matches:
                if len(match) >= 2:
                    citations.add(f"Nghị định {match[0]}, Điều {match[1]}")
        except Exception as e:
            print(f"Error extracting Nghị định + Điều citations: {str(e)}")
        
        # Extract individual document citations
        try:
            for pattern in citation_patterns:
                for match in re.findall(pattern, response):
                    match_value = match
                    if isinstance(match, tuple):
                        # Skip combined patterns already processed
                        if len(match) >= 2 and ('thông tư' in pattern.lower() or 'nghị định' in pattern.lower()):
                            continue
                        match_value = match[0]  # Get first capture group
                    
                    # Clean and normalize match
                    match_str = str(match_value).strip()
                    
                    # Create appropriate citation based on pattern
                    if 'thông tư' in pattern.lower() or pattern.lower().startswith(r'\btt'):
                        citations.add(f"Thông tư {match_str}")
                    elif 'nghị định' in pattern.lower() or pattern.lower().startswith(r'\bnd'):
                        citations.add(f"Nghị định {match_str}")
                    elif 'điều' in pattern.lower():
                        citations.add(f"Điều {match_str}")
        except Exception as e:
            print(f"Error extracting individual citations: {str(e)}")
        
        return list(citations)
    
# In rag_evaluation.py, modify the calculate_citation_score method

    def calculate_citation_score(self, extracted_citations, expected_citations):
        """
        Calculate F1 score for citation accuracy with improved partial matching
        
        Args:
            extracted_citations: List of citations found in response
            expected_citations: List of expected citations from ground truth
            
        Returns:
            float: F1 score between 0 and 1
        """
        try:
            # Handle empty cases
            if not expected_citations or any("Không có" in str(citation) for citation in expected_citations):
                # If no citations expected, perfect score for no citations found
                return 1.0 if not extracted_citations else 0.5
            
            if not extracted_citations:
                return 0.0  # No citations found when expected
            
            # Normalize citations for better matching
            normalized_expected = [self._normalize_citation(citation) for citation in expected_citations]
            normalized_extracted = [self._normalize_citation(citation) for citation in extracted_citations]
            
            # Calculate precision with partial matching
            matched_extracted = 0
            for extracted in normalized_extracted:
                # Try exact matches first
                if any(extracted == expected for expected in normalized_expected):
                    matched_extracted += 1
                    continue
                    
                # Try partial matches using similarity
                best_similarity = 0
                for expected in normalized_expected:
                    similarity = self._citation_similarity(extracted, expected)
                    best_similarity = max(best_similarity, similarity)
                
                # Count as partial match if similarity above threshold
                if best_similarity > 0.6:  # Threshold for considering a good match
                    matched_extracted += best_similarity
            
            # Calculate recall with partial matching
            matched_expected = 0
            for expected in normalized_expected:
                # Try exact matches first
                if any(expected == extracted for extracted in normalized_extracted):
                    matched_expected += 1
                    continue
                    
                # Try partial matches using similarity
                best_similarity = 0
                for extracted in normalized_extracted:
                    similarity = self._citation_similarity(expected, extracted)
                    best_similarity = max(best_similarity, similarity)
                
                # Count as partial match if similarity above threshold
                if best_similarity > 0.6:  # Threshold for considering a good match
                    matched_expected += best_similarity
            
            # Calculate precision and recall
            precision = matched_extracted / len(normalized_extracted) if normalized_extracted else 0
            recall = matched_expected / len(normalized_expected) if normalized_expected else 0
            
            # Handle zero division
            if precision + recall == 0:
                return 0.0
                
            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall)
            return min(1.0, f1_score)
        except Exception as e:
            print(f"Error calculating citation score: {str(e)}")
            return 0.0

    def _normalize_citation(self, citation):
        """Normalize citation string for better matching"""
        if not citation:
            return ""
            
        citation = str(citation).lower().strip()
        
        # Standardize formatting
        citation = re.sub(r'\s+', ' ', citation)
        citation = re.sub(r'[,;:.]', ' ', citation)
        
        # Standardize document types
        citation = re.sub(r'thong\s*tu|tt', 'thong tu', citation)
        citation = re.sub(r'nghi\s*dinh|nd', 'nghi dinh', citation)
        citation = re.sub(r'dieu', 'dieu', citation)
        
        return citation.strip()

    def _citation_similarity(self, citation1, citation2):
        """
        Calculate similarity between two citations
        
        Args:
            citation1: First citation string
            citation2: Second citation string
            
        Returns:
            float: Similarity score from 0 to 1
        """
        # Extract document types and numbers
        type_num1 = re.findall(r'(thong tu|nghi dinh|dieu)\s*(\d+)', citation1)
        type_num2 = re.findall(r'(thong tu|nghi dinh|dieu)\s*(\d+)', citation2)
        
        if not type_num1 or not type_num2:
            # If no structured info, fall back to string overlap
            return self._string_overlap(citation1, citation2)
        
        # Calculate matches
        matches = 0
        total = max(len(type_num1), len(type_num2))
        
        for doc_type1, num1 in type_num1:
            for doc_type2, num2 in type_num2:
                # Same document type and number
                if doc_type1 == doc_type2 and num1 == num2:
                    matches += 1
                    break
                # Same document type but different number
                elif doc_type1 == doc_type2:
                    matches += 0.5
                    break
                # Different document type but same number
                elif num1 == num2:
                    matches += 0.3
                    break
        
        return matches / total if total > 0 else 0

    def _string_overlap(self, str1, str2):
        """Calculate simple string overlap ratio"""
        if not str1 or not str2:
            return 0
            
        # Calculate token overlap
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())
        
        common = tokens1.intersection(tokens2)
        return len(common) / max(len(tokens1), len(tokens2))
    
    def _citation_match(self, citation1, citation2):
        """
        Check if two citations match using fuzzy matching
        
        Args:
            citation1: First citation string
            citation2: Second citation string
            
        Returns:
            bool: True if citations match
        """
        try:
            # Direct substring match
            if citation1 in citation2 or citation2 in citation1:
                return True
            
            # Extract numbers for comparison
            tt_match1 = re.search(r'thông\s*tư\s+(\d+)', citation1)
            tt_match2 = re.search(r'thông\s*tư\s+(\d+)', citation2)
            
            if tt_match1 and tt_match2 and tt_match1.group(1) == tt_match2.group(1):
                return True
            
            nd_match1 = re.search(r'nghị\s*định\s+(\d+)', citation1)
            nd_match2 = re.search(r'nghị\s*định\s+(\d+)', citation2)
            
            if nd_match1 and nd_match2 and nd_match1.group(1) == nd_match2.group(1):
                return True
            
            dieu_match1 = re.search(r'điều\s+(\d+)', citation1)
            dieu_match2 = re.search(r'điều\s+(\d+)', citation2)
            
            if dieu_match1 and dieu_match2 and dieu_match1.group(1) == dieu_match2.group(1):
                return True
                
            return False
        except Exception as e:
            print(f"Error in citation matching: {str(e)}")
            return False
    
    # Improved document ID extraction function for rag_evaluation.py
    # In rag_evaluation.py, modify the extract_document_ids method

    def extract_document_ids(self, sources_text):
        """
        Extract document IDs from sources text with enhanced pattern recognition
        
        Args:
            sources_text: Text containing document references
            
        Returns:
            list: Extracted document IDs
        """
        if not sources_text or not isinstance(sources_text, str):
            return []
        
        try:
            # Enhanced patterns for Vietnamese legal document references
            doc_patterns = [
                # Source ID patterns
                r'(?:Nguồn|Source)[^\d]*#?(\d+)',
                r'(?:Tài liệu|Document)[^\d]*#?(\d+)',
                
                # Legal document patterns (Thông tư)
                r'[Tt]h[ôo]ng\s*t[ưu]\s*(?:số\s+)?(\d+)',
                r'(?<=[^a-zA-Z0-9])TT[_\s]*(\d+)(?=[^a-zA-Z0-9]|$)',
                
                # Legal document patterns (Nghị định)
                r'[Nn]gh[iị]\s*[đd][iị]nh\s*(?:số\s+)?(\d+)',
                r'(?<=[^a-zA-Z0-9])ND[_\s]*(\d+)(?=[^a-zA-Z0-9]|$)',
                
                # Article patterns
                r'[ĐđDd]i[êeề]u\s*(\d+)'
            ]
            
            doc_ids = []
            
            for pattern in doc_patterns:
                matches = re.findall(pattern, sources_text)
                for match in matches:
                    if match and match.strip():
                        # Detect pattern type and format properly
                        match = match.strip()
                        
                        if 'thông tư' in pattern.lower() or pattern.lower().find('tt') >= 0:
                            doc_ids.append(f"TT_{match}")
                        elif 'nghị định' in pattern.lower() or pattern.lower().find('nd') >= 0:
                            doc_ids.append(f"ND_{match}")
                        elif 'điều' in pattern.lower():
                            doc_ids.append(f"Dieu_{match}")
                        else:
                            # For source numbers, check if they refer to documents in the text
                            source_num = int(match)
                            source_context = self._get_context_for_source(sources_text, source_num)
                            
                            if 'thông tư' in source_context.lower() or 'tt' in source_context.lower():
                                tt_match = re.search(r'[Tt][Tt][_\s]*(\d+)|[Tt]h[ôo]ng\s*t[ưu]\s*(?:số\s+)?(\d+)', source_context)
                                if tt_match:
                                    doc_num = tt_match.group(1) or tt_match.group(2)
                                    doc_ids.append(f"TT_{doc_num}")
                            elif 'nghị định' in source_context.lower() or 'nd' in source_context.lower():
                                nd_match = re.search(r'[Nn][Dd][_\s]*(\d+)|[Nn]gh[iị]\s*[đd][iị]nh\s*(?:số\s+)?(\d+)', source_context)
                                if nd_match:
                                    doc_num = nd_match.group(1) or nd_match.group(2)
                                    doc_ids.append(f"ND_{doc_num}")
                            elif 'điều' in source_context.lower():
                                dieu_match = re.search(r'[ĐđDd]i[êeề]u\s*(\d+)', source_context)
                                if dieu_match:
                                    doc_ids.append(f"Dieu_{dieu_match.group(1)}")
            
            # Normalize and remove duplicates
            normalized_ids = []
            seen = set()
            for doc_id in doc_ids:
                if doc_id and doc_id not in seen:
                    normalized_ids.append(doc_id)
                    seen.add(doc_id)
            
            return normalized_ids
        except Exception as e:
            print(f"Error extracting document IDs: {str(e)}")
            return []

    def _get_context_for_source(self, text, source_num):
        """Extract context around a source number mention"""
        pattern = rf'(?:[Nn]gu[ồô]n|[Ss]ource|[Tt][àa]i li[ệê]u|[Dd]ocument)[^\d]*#?{source_num}'
        match = re.search(pattern, text)
        if match:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            return text[start:end]
        return ""
    
    def calculate_retrieval_metrics(self, retrieved_docs, relevant_docs):
        """
        Calculate retrieval quality metrics: precision, recall, F1, and MRR
        
        Args:
            retrieved_docs: List of document IDs retrieved by the system
            relevant_docs: List of relevant document IDs from ground truth
            
        Returns:
            dict: Retrieval metrics
        """
        try:
            # Ensure safe inputs
            retrieved_docs = retrieved_docs or []
            relevant_docs = relevant_docs or []
            
            # Handle empty cases
            if not relevant_docs:
                return {
                    "precision": 1.0 if not retrieved_docs else 0.0,
                    "recall": 1.0 if not retrieved_docs else 0.0,
                    "f1": 1.0 if not retrieved_docs else 0.0,
                    "mrr": 0.0
                }
            
            if not retrieved_docs:
                return {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "mrr": 0.0
                }
            
            # Calculate relevant documents that were retrieved
            relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
            
            # Calculate precision and recall
            precision = len(relevant_retrieved) / len(retrieved_docs)
            recall = min(1.0, len(relevant_retrieved) / len(relevant_docs))
            
            # Calculate F1 score
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            
            # Calculate MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, doc in enumerate(retrieved_docs):
                if doc in relevant_docs:
                    # Reciprocal of the rank (1-based indexing)
                    mrr = 1.0 / (i + 1)
                    break
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mrr": mrr
            }
        except Exception as e:
            print(f"Error calculating retrieval metrics: {str(e)}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mrr": 0.0
            }
    
    def calculate_additional_metrics(self, rag_answer, ground_truth, retrieved_docs, relevant_docs):
        """
        Calculate additional RAG-specific evaluation metrics
        
        Args:
            rag_answer: Generated response text
            ground_truth: Ground truth answer text
            retrieved_docs: List of document IDs retrieved by the system
            relevant_docs: List of relevant document IDs from ground truth
            
        Returns:
            dict: Additional evaluation metrics
        """
        try:
            metrics = {}
            
            # Ensure we have valid input values
            retrieved_docs = retrieved_docs or []
            relevant_docs = relevant_docs or []
            rag_answer = rag_answer or ""
            ground_truth = ground_truth or ""
            
            # 1. Faithfulness - how well the generated content aligns with the retrieved documents
            if retrieved_docs and relevant_docs:
                relevant_retrieved = len(set(retrieved_docs).intersection(set(relevant_docs)))
                metrics["faithfulness"] = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 1.0
            else:
                metrics["faithfulness"] = 0.0 if retrieved_docs else 1.0
            
            # 2. Completeness - how comprehensively the response covers the ground truth
            metrics["completeness"] = self.compute_semantic_similarity(rag_answer, ground_truth)
            
            # 3. Precision - lexical precision of the response compared to ground truth
            if rag_answer and ground_truth:
                # Use tokenization to count overlap in words
                rag_tokens = set(preprocess_text(rag_answer).split())
                truth_tokens = set(preprocess_text(ground_truth).split())
                
                if rag_tokens:
                    metrics["precision"] = len(rag_tokens.intersection(truth_tokens)) / len(rag_tokens)
                else:
                    metrics["precision"] = 0.0
            else:
                metrics["precision"] = 0.0
            
            # 4. Source Utilization - how well the system used available sources
            # Assuming limit=5 for retrieved documents
            metrics["source_utilization"] = min(1.0, len(retrieved_docs) / 5.0) if retrieved_docs else 0.0
            
            # 5. Calculate overall RAG Quality with weighted metrics
            weights = {
                "faithfulness": 0.35,  # Prioritize factual correctness
                "completeness": 0.35,  # Ensure comprehensive answers
                "precision": 0.15,     # Reward precise responses
                "source_utilization": 0.15  # Consider document usage
            }
            
            # Calculate weighted average
            metrics["rag_quality"] = sum(metrics.get(k, 0.0) * weights[k] for k in weights)
            
            return metrics
        except Exception as e:
            print(f"Error calculating additional metrics: {str(e)}")
            return {
                "faithfulness": 0.0,
                "completeness": 0.0,
                "precision": 0.0,
                "source_utilization": 0.0,
                "rag_quality": 0.0
            }
    
    def calculate_hallucination_score(self, response, ground_truth, citation_score):
        """
        Calculate hallucination score using weighted combination of metrics
        
        Args:
            response: Generated response text
            ground_truth: Ground truth answer text
            citation_score: Citation accuracy score
            
        Returns:
            float: Hallucination score between 0 and 1 (lower is better)
        """
        try:
            # Compute semantic similarity
            semantic_sim = self.compute_semantic_similarity(response, ground_truth)
            
            # Calculate weighted score
            hallucination_score = 1.0 - (
                (self.hallucination_weights["semantic_similarity"] * semantic_sim) + 
                (self.hallucination_weights["citation_score"] * citation_score)
            )
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, hallucination_score))
        except Exception as e:
            print(f"Error calculating hallucination score: {str(e)}")
            return 1.0
    
    def evaluate_with_rag(self, test_data, output_file, limit=5, max_retries=2):
        """
        Run test questions through RAG system and evaluate results
        
        Args:
            test_data: List of test questions and ground truths
            output_file: Path to save evaluation results
            limit: Number of documents to retrieve per query
            max_retries: Maximum number of retries for failed queries
            
        Returns:
            list: Evaluation results
        """
        # Import here to avoid circular imports
        try:
            from generate import process_query
        except ImportError as e:
            print(f"Error importing process_query: {str(e)}")
            print("Please ensure generate.py is in the current directory")
            return []
        
        results = []
        total_questions = len(test_data)
        
        for idx, item in enumerate(tqdm(test_data, desc="Evaluating with RAG")):
            question = item["question"]
            ground_truth = item["ground_truth"]
            expected_citations = item.get("citations", [])
            question_type = item.get("question_type", "Unknown")
            relevant_docs = item.get("relevant_docs", [])
            
            print(f"\nProcessing question {idx+1}/{total_questions}: {question[:50]}...")
            
            # Try with retries
            for retry in range(max_retries):
                start_time = time.time()
                try:
                    # Call process_query function with the test question
                    rag_answer, sources_text = process_query(question, num_docs=limit)
                    
                    # Make sure we have valid values
                    rag_answer = rag_answer or "Không thể tạo câu trả lời."
                    sources_text = sources_text or ""
                    
                    # Extract document IDs from sources if available
                    retrieved_docs = self.extract_document_ids(sources_text)
                    
                    # Calculate retrieval metrics
                    retrieval_metrics = self.calculate_retrieval_metrics(retrieved_docs, relevant_docs)
                    
                    # Calculate response metrics
                    processing_time = time.time() - start_time
                    semantic_similarity = self.compute_semantic_similarity(rag_answer, ground_truth)
                    
                    # Extract citations
                    combined_text = f"{rag_answer}\n{sources_text}"
                    extracted_citations = self.extract_citations(combined_text)
                    
                    citation_score = self.calculate_citation_score(extracted_citations, expected_citations)
                    
                    # Calculate hallucination score
                    hallucination_score = self.calculate_hallucination_score(
                        rag_answer, 
                        ground_truth, 
                        citation_score
                    )
                    
                    # Calculate additional RAG metrics
                    additional_metrics = self.calculate_additional_metrics(
                        rag_answer,
                        ground_truth,
                        retrieved_docs,
                        relevant_docs
                    )
                    
                    # Add to results
                    results.append({
                        "question": question,
                        "question_type": question_type,
                        "ground_truth": ground_truth,
                        "rag_answer": rag_answer,
                        "sources": sources_text,
                        "retrieved_docs": retrieved_docs,
                        "relevant_docs": relevant_docs,
                        "retrieval_metrics": retrieval_metrics,
                        "expected_citations": expected_citations,
                        "extracted_citations": extracted_citations,
                        "processing_time": processing_time,
                        "semantic_similarity": float(semantic_similarity),
                        "citation_score": float(citation_score),
                        "hallucination_score": float(hallucination_score),
                        **{f"rag_{k}": float(v) for k, v in additional_metrics.items()}
                    })

        # Success - break retry loop
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error processing question (attempt {retry+1}/{max_retries}): {error_msg}")
                    
                    if retry == max_retries - 1:
                        # Add failed result on last retry
                        results.append({
                            "question": question,
                            "question_type": question_type,
                            "ground_truth": ground_truth,
                            "rag_answer": f"ERROR: {error_msg}",
                            "sources": "",
                            "retrieved_docs": [],
                            "relevant_docs": relevant_docs,
                            "retrieval_metrics": {
                                "precision": 0.0,
                                "recall": 0.0,
                                "f1": 0.0,
                                "mrr": 0.0
                            },
                            "expected_citations": expected_citations,
                            "extracted_citations": [],
                            "processing_time": time.time() - start_time,
                            "semantic_similarity": 0.0,
                            "citation_score": 0.0,
                            "hallucination_score": 1.0,  # Max hallucination for errors
                            "rag_faithfulness": 0.0,
                            "rag_completeness": 0.0,
                            "rag_precision": 0.0,
                            "rag_source_utilization": 0.0,
                            "rag_quality": 0.0
                        })
                    else:
                        # Wait before retry
                        time.sleep(10)
            
            # Save intermediate results every 5 questions
            if (idx + 1) % 5 == 0 or idx == total_questions - 1:
                self.save_results(results, output_file)
                print(f"Saved intermediate results after processing {idx+1}/{total_questions} questions")
            
            print(f"Waiting 8 seconds before next question...")
            time.sleep(8)

        
        # Final save of results
        self.save_results(results, output_file)
        return results
        
    
    def align_results_for_comparison(self, rag_results, baseline_results):
        """
        Align RAG and baseline results for fair comparison
        
        Args:
            rag_results: List of RAG evaluation results
            baseline_results: List of baseline evaluation results
            
        Returns:
            tuple: (aligned_rag, aligned_baseline)
        """
        if not rag_results or not baseline_results:
            return [], []
            
        aligned_rag = []
        aligned_baseline = []
        
        # Create dictionaries with question as key for easier lookup
        rag_dict = {r["question"]: r for r in rag_results if "question" in r}
        baseline_dict = {r["question"]: r for r in baseline_results if "question" in r}
        
        # Get all questions that appear in both datasets
        common_questions = set(rag_dict.keys()) & set(baseline_dict.keys())
        
        # For each common question, add corresponding results to aligned lists
        for question in common_questions:
            rag_result = rag_dict[question]
            baseline_result = baseline_dict[question]
            
            # Skip error results
            rag_has_error = "rag_answer" in rag_result and "ERROR:" in rag_result["rag_answer"]
            baseline_has_error = "baseline_answer" in baseline_result and "ERROR:" in baseline_result["baseline_answer"]
            
            if rag_has_error or baseline_has_error:
                continue
                
            aligned_rag.append(rag_result)
            aligned_baseline.append(baseline_result)
        
        print(f"Aligned {len(aligned_rag)} questions for comparison (out of {len(rag_results)} RAG and {len(baseline_results)} baseline)")
        return aligned_rag, aligned_baseline
    
    def calculate_overall_metrics(self, results):
        """
        Calculate overall metrics from evaluation results
        
        Args:
            results: List of evaluation results
            
        Returns:
            dict: Aggregated metrics
        """
        if not results:
            return {}
        
        metrics = {
            "total_questions": len(results),
            "errors": 0,
            "avg_semantic_similarity": 0.0,
            "avg_citation_score": 0.0,
            "avg_hallucination_score": 0.0,
            "avg_processing_time": 0.0,
            "retrieval_metrics": {
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
                "avg_mrr": 0.0
            },
            "rag_specific_metrics": {
                "avg_faithfulness": 0.0,
                "avg_completeness": 0.0,
                "avg_precision": 0.0,
                "avg_source_utilization": 0.0,
                "avg_quality": 0.0
            },
            "by_question_type": {}
        }
        
        # Filter out errors for metric calculation
        has_rag_answer = any("rag_answer" in r for r in results)
        error_key = "rag_answer" if has_rag_answer else "baseline_answer"
        
        # Count errors
        for r in results:
            if error_key in r and isinstance(r[error_key], str) and "ERROR:" in r[error_key]:
                metrics["errors"] += 1
        
        # Filter out error results
        valid_results = [r for r in results if error_key not in r or "ERROR:" not in r.get(error_key, "")]
        valid_count = len(valid_results)
        
        if valid_count > 0:
            # Calculate overall averages
            metrics["avg_semantic_similarity"] = sum(r.get("semantic_similarity", 0.0) for r in valid_results) / valid_count
            metrics["avg_citation_score"] = sum(r.get("citation_score", 0.0) for r in valid_results) / valid_count
            metrics["avg_hallucination_score"] = sum(r.get("hallucination_score", 1.0) for r in valid_results) / valid_count
            
            # Add processing time if available
            if any("processing_time" in r for r in valid_results):
                metrics["avg_processing_time"] = sum(r.get("processing_time", 0.0) for r in valid_results) / valid_count
            
            # Calculate average retrieval metrics if available
            if any("retrieval_metrics" in r for r in valid_results):
                retrieval_sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}
                retrieval_count = 0
                
                for r in valid_results:
                    if "retrieval_metrics" in r:
                        retrieval_count += 1
                        for metric, value in r["retrieval_metrics"].items():
                            retrieval_sums[metric] += value
                
                if retrieval_count > 0:
                    for metric in retrieval_sums:
                        metrics["retrieval_metrics"][f"avg_{metric}"] = retrieval_sums[metric] / retrieval_count
            
            # Calculate average RAG-specific metrics if available
            rag_metrics = [
                "rag_faithfulness", "rag_completeness", "rag_precision", 
                "rag_source_utilization", "rag_quality"
            ]
            
            if any(any(metric in r for metric in rag_metrics) for r in valid_results):
                for metric in rag_metrics:
                    base_metric = metric[4:]  # Remove 'rag_' prefix
                    metric_values = [r.get(metric, 0.0) for r in valid_results if metric in r]
                    if metric_values:
                        metrics["rag_specific_metrics"][f"avg_{base_metric}"] = sum(metric_values) / len(metric_values)
        
        # Group by question type
        question_types = {}
        for r in results:
            q_type = r.get("question_type", "Unknown")
            if q_type not in question_types:
                question_types[q_type] = []
            question_types[q_type].append(r)
        
        # Calculate metrics by question type
        for q_type, type_results in question_types.items():
            valid_type_results = [r for r in type_results if error_key not in r or "ERROR:" not in r.get(error_key, "")]
            valid_type_count = len(valid_type_results)
            
            type_metrics = {
                "count": len(type_results),
                "errors": len(type_results) - valid_type_count,
            }
            
            if valid_type_count > 0:
                # Add general metrics
                type_metrics["avg_semantic_similarity"] = sum(r.get("semantic_similarity", 0.0) for r in valid_type_results) / valid_type_count
                type_metrics["avg_citation_score"] = sum(r.get("citation_score", 0.0) for r in valid_type_results) / valid_type_count
                type_metrics["avg_hallucination_score"] = sum(r.get("hallucination_score", 1.0) for r in valid_type_results) / valid_type_count
                
                # Add RAG-specific metrics by question type if available
                for metric in rag_metrics:
                    if any(metric in r for r in valid_type_results):
                        base_metric = metric[4:]  # Remove 'rag_' prefix
                        metric_values = [r.get(metric, 0.0) for r in valid_type_results if metric in r]
                        if metric_values:
                            type_metrics[f"avg_{base_metric}"] = sum(metric_values) / len(metric_values)
            
            metrics["by_question_type"][q_type] = type_metrics
                
        return metrics
    
    def perform_statistical_tests(self, rag_results, baseline_results):
        """
        Perform statistical tests to compare RAG and baseline performance
        
        Args:
            rag_results: List of RAG evaluation results
            baseline_results: List of baseline evaluation results
            
        Returns:
            dict: Statistical test results
        """
        if not rag_results or not baseline_results:
            return {
                "valid_comparison": False,
                "reason": "Missing results for comparison"
            }
                
        # Align results to ensure fair comparison
        aligned_rag, aligned_baseline = self.align_results_for_comparison(rag_results, baseline_results)
        
        if len(aligned_rag) < 5:  # Need minimum sample size for meaningful tests
            return {
                "valid_comparison": False,
                "reason": f"Sample size too small ({len(aligned_rag)})"
            }
                
        # Define metrics to test with their properties
        metrics_to_test = {
            "semantic_similarity": {
                "display_name": "Semantic Similarity",
                "higher_better": True
            },
            "citation_score": {
                "display_name": "Citation Score",
                "higher_better": True
            },
            "hallucination_score": {
                "display_name": "Hallucination Score",
                "higher_better": False  # Lower hallucination is better
            }
        }
        
        # Add RAG-specific metrics if they exist in the results
        rag_specific_metrics = {
            "rag_faithfulness": {
                "display_name": "Faithfulness",
                "higher_better": True
            },
            "rag_completeness": {
                "display_name": "Completeness",
                "higher_better": True
            },
            "rag_precision": {
                "display_name": "Precision",
                "higher_better": True
            },
            "rag_source_utilization": {
                "display_name": "Source Utilization",
                "higher_better": True
            },
            "rag_quality": {
                "display_name": "RAG Quality",
                "higher_better": True
            }
        }
        
        # Add RAG-specific metrics if they exist in results
        for metric in rag_specific_metrics:
            if any(metric in r for r in aligned_rag):
                metrics_to_test[metric] = rag_specific_metrics[metric]
        
        # Initialize test results
        test_results = {
            "valid_comparison": True,
            "sample_size": len(aligned_rag),
            "tests": {}
        }
        
        # Perform paired t-tests for each metric
        for metric, config in metrics_to_test.items():
            try:
                # Check if metric exists in results
                if not all(metric in r for r in aligned_rag) or not all(metric in r for r in aligned_baseline):
                    continue
                
                # Extract metric values from both sets
                rag_values = [float(r.get(metric, 0.0)) for r in aligned_rag]
                baseline_values = [float(r.get(metric, 0.0)) for r in aligned_baseline]
                
                # Validate data
                if len(rag_values) <= 1 or len(baseline_values) <= 1:
                    test_results["tests"][metric] = {
                        "display_name": config["display_name"],
                        "error": "Insufficient data points",
                        "rag_mean": float(np.mean(rag_values)) if rag_values else 0.0,
                        "baseline_mean": float(np.mean(baseline_values)) if baseline_values else 0.0,
                        "higher_better": config["higher_better"]
                    }
                    continue
                
                # Calculate means
                rag_mean = float(np.mean(rag_values))
                baseline_mean = float(np.mean(baseline_values))
                
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(rag_values, baseline_values)
                t_stat = float(t_stat)
                p_value = float(p_value)
                
                # Calculate effect size (Cohen's d for paired samples)
                diff = np.array(rag_values) - np.array(baseline_values)
                diff_std = np.std(diff, ddof=1)
                effect_size = float(np.mean(diff) / diff_std) if diff_std > 0 else 0.0
                
                # Determine if RAG is better
                higher_better = config["higher_better"]
                
                if higher_better:
                    rag_better = rag_mean > baseline_mean
                else:
                    rag_better = rag_mean < baseline_mean
                
                # Store results
                test_results["tests"][metric] = {
                    "display_name": config["display_name"],
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "effect_size": effect_size,
                    "effect_magnitude": self._interpret_effect_size(effect_size),
                    "rag_mean": rag_mean,
                    "baseline_mean": baseline_mean,
                    "rag_better": rag_better,
                    "higher_better": higher_better
                }
            except Exception as e:
                print(f"Error performing statistical test for {metric}: {str(e)}")
                test_results["tests"][metric] = {
                    "display_name": config.get("display_name", metric),
                    "error": str(e)
                }
                
        # Add correlation analysis between metrics
        try:
            correlations = {}
            
            # Select metrics available in RAG results
            rag_metrics_available = {}
            for k in metrics_to_test:
                if all(k in r for r in aligned_rag):
                    rag_metrics_available[k] = [r.get(k, 0.0) for r in aligned_rag]
            
            # Calculate correlations between pairs of metrics
            for metric1 in rag_metrics_available:
                for metric2 in rag_metrics_available:
                    if metric1 < metric2:  # Avoid duplicates and self-correlations
                        values1 = rag_metrics_available[metric1]
                        values2 = rag_metrics_available[metric2]
                        
                        # Convert to numpy arrays and ensure floating point
                        values1 = np.array(values1, dtype=float)
                        values2 = np.array(values2, dtype=float)
                        
                        # Skip if either array has constant values
                        if np.std(values1) == 0 or np.std(values2) == 0:
                            correlations[f"{metric1}_vs_{metric2}"] = {
                                "correlation": float('nan'),
                                "p_value": float('nan'),
                                "significant": False,
                                "note": "Skipped due to constant input"
                            }
                            continue
                            
                        # Calculate Pearson correlation
                        corr, p_value = stats.pearsonr(values1, values2)
                        
                        correlations[f"{metric1}_vs_{metric2}"] = {
                            "correlation": float(corr),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05
                        }
                                
            test_results["correlations"] = correlations
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            test_results["correlations"] = {}
        
        return test_results
    
    def _interpret_effect_size(self, effect_size):
        """
        Interpret Cohen's d effect size magnitude
        
        Args:
            effect_size: Cohen's d effect size
            
        Returns:
            str: Effect size interpretation
        """
        effect_size = abs(effect_size)
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
                    
    
    def evaluate_without_rag(self, test_data, llm_responses_file, output_file):
        """
        Evaluate baseline responses without RAG
        
        Args:
            test_data: List of test questions and ground truths
            llm_responses_file: Path to file with baseline responses
            output_file: Path to save evaluation results
            
        Returns:
            list: Evaluation results
        """
        # Load baseline responses
        try:
            with open(llm_responses_file, 'r', encoding='utf-8') as f:
                baseline_responses = json.load(f)
            print(f"Successfully loaded {len(baseline_responses)} baseline responses from {llm_responses_file}")
        except Exception as e:
            print(f"Error loading baseline responses from {llm_responses_file}: {str(e)}")
            return []
        
        results = []
        total_questions = min(len(test_data), len(baseline_responses))
        
        for i in tqdm(range(total_questions), desc="Evaluating baseline"):
            try:
                item = test_data[i]
                question = item["question"]
                ground_truth = item["ground_truth"]
                expected_citations = item.get("citations", [])
                question_type = item.get("question_type", "Unknown")
                relevant_docs = item.get("relevant_docs", [])
                
                baseline_answer = baseline_responses[i]
                
                # Calculate metrics
                semantic_similarity = self.compute_semantic_similarity(baseline_answer, ground_truth)
                extracted_citations = self.extract_citations(baseline_answer)
                citation_score = self.calculate_citation_score(extracted_citations, expected_citations)
                
                hallucination_score = self.calculate_hallucination_score(
                    baseline_answer, 
                    ground_truth, 
                    citation_score
                )
                
                # Add to results
                results.append({
                    "question": question,
                    "question_type": question_type,
                    "ground_truth": ground_truth,
                    "baseline_answer": baseline_answer,
                    "expected_citations": expected_citations,
                    "extracted_citations": extracted_citations,
                    "relevant_docs": relevant_docs,
                    "semantic_similarity": float(semantic_similarity),
                    "citation_score": float(citation_score),
                    "hallucination_score": float(hallucination_score),
                    "retrieval_metrics": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "mrr": 0.0
                    }
                })
                
                # Save intermediate results every 5 questions
                if (i + 1) % 5 == 0 or i == total_questions - 1:
                    self.save_results(results, output_file)
                    print(f"Saved intermediate results after processing {i+1}/{total_questions} baseline responses")
                    
            except Exception as e:
                print(f"Error processing baseline question {i}: {str(e)}")
                # Add error entry to maintain alignment with test data
                results.append({
                    "question": test_data[i]["question"] if i < len(test_data) else f"Unknown question {i}",
                    "question_type": "Unknown",
                    "ground_truth": test_data[i]["ground_truth"] if i < len(test_data) else "",
                    "baseline_answer": f"ERROR: {str(e)}",
                    "expected_citations": [],
                    "extracted_citations": [],
                    "semantic_similarity": 0.0,
                    "citation_score": 0.0,
                    "hallucination_score": 1.0,  # Max hallucination for errors
                    "retrieval_metrics": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "mrr": 0.0
                    }
                })