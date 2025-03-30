import json
import time
import numpy as np #used for numerical operations
from tqdm import tqdm #used for progress bars
import traceback #used for error handling
import re
from functools import lru_cache #used for caching results
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats #used for statistical tests
from generate import process_query 

class RAGEvaluator:
# Simplified evaluator for testing RAG implementation against hallucinations

    
    def __init__(self, 
                 embedding_model_name="keepitreal/vietnamese-sbert",
                 hallucination_weights=None,
                 use_cache=True):
        """
        Initialize evaluation models and parameters
        
        Args:
            embedding_model_name: Model for measuring semantic similarity 
            hallucination_weights: Dict of weights used to later define different weight factors in order to calculate hallucination score
            use_cache: True in order to avoid recomputing embeddings for the same text so performance is improved
        """
        # Set up embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
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
        
    #cahces the embeddings to avoid recomputation
    # this is a decorator that caches the results of the function for a given input
    # it uses the lru_cache from functools to limit the cache size to 1024 items    
    @lru_cache(maxsize=1024)
    def _cached_encode(self, text):
        if not text:
            return np.zeros((384,))  # Most sentence-transformers use 384 dim embeddings
        return self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    #load the test data and ground truths from the test_questions.json file    
    def load_test_data(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            print(f"Successfully loaded {len(test_data)} test questions from {file_path}")
            return test_data
        except Exception as e:
            print(f"Error loading test data from {file_path}: {str(e)}")
            traceback.print_exc()
            return []
    
    #save evaluation results to a json fiel
    def save_results(self, results, file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved results to {file_path}")
        except Exception as e:
            print(f"Error saving results to {file_path}: {str(e)}")
            traceback.print_exc()
            # Save to a backup file in case of error
            backup_path = f"{file_path}.backup"
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Saved backup results to {backup_path}")
            except:
                print("Failed to save backup results")
    
    # This function takes the RAG generated response and the ground truth answer
    # Converts both to vector embeddings using the model
    # Calculates the cosine similarity between the embeddings
    # Returns a score between 0 and 1 where 1 means perfect sematic match
    def compute_semantic_similarity(self, response, ground_truth):
        try:
            if not response or not ground_truth:
                return 0.0
                
            # Clean and truncate texts if they're too long to prevent memory issues
            response = response[:10000].strip()
            ground_truth = ground_truth[:10000].strip()
            
            # Use cached encodings if available
            if self.use_cache:
                response_embedding = self._cached_encode(response)
                ground_truth_embedding = self._cached_encode(ground_truth)
            else:
                response_embedding = self.embedding_model.encode(response, convert_to_numpy=True, show_progress_bar=False)
                ground_truth_embedding = self.embedding_model.encode(ground_truth, convert_to_numpy=True, show_progress_bar=False)
            
            similarity = cosine_similarity([response_embedding], [ground_truth_embedding])[0][0]
            return float(similarity)  # Ensure it's a native Python float, not numpy float
        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
            traceback.print_exc()
            return 0.0
    
    #this function uses regular expressions to identify citations from the response text
    # It looks for patterns that match the format of citations in the text
    def extract_citations(self, response):
        if not response:
            return []
        
        # Simple pattern for extracting citations
        citation_patterns = [
            r'[Tt]hông tư (\d+).*?[Đ|đ]iều (\d+)',
            r'TT(\d+).*?[Đ|đ]iều (\d+)',
            r'[Tt]hông tư.*?(\d+)',
            r'[Đ|đ]iều.*?(\d+)',
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    tt = match[0].strip()
                    dieu = match[1].strip()
                    citation = f"Thông tư {tt}, Điều {dieu}"
                    if citation not in citations:
                        citations.append(citation)
                elif isinstance(match, tuple) and len(match) == 1 or isinstance(match, str):
                    match_str = match[0] if isinstance(match, tuple) else match
                    if 'thông tư' in pattern.lower() or 'tt' in pattern.lower():
                        citation = f"Thông tư {match_str.strip()}"
                        if citation not in citations:
                            citations.append(citation)
                    elif 'điều' in pattern.lower():
                        citation = f"Điều {match_str.strip()}"
                        if citation not in citations:
                            citations.append(citation)
        
        return citations
    
    # this function will compare the citations found in the response against the expected citations from the test data (ground truth)
    # it calculates an F1 score that balances precision (are all citations correct?) and recall (are all expected citations found?)
    # the score is between 0 and 1, where 1 means perfect match 
    # 0.5 means that it has given more citations than expected to/gave citations in cases where none were expected
    def calculate_citation_score(self, extracted_citations, expected_citations):
        try:
            # Handle empty cases
            if not expected_citations or any(("Không có" in str(citation)) for citation in expected_citations):
                return 1.0 if not extracted_citations else 0.5
            
            if not extracted_citations:
                return 0.0
            
            # Normalize expected citations
            normalized_expected = []
            for citation in expected_citations:
                citation = str(citation)
                normalized_expected.append(citation)
                
            # Simple matching logic
            matched = 0
            for ext_citation in extracted_citations:
                ext_citation_lower = ext_citation.lower()
                
                for exp_citation in normalized_expected:
                    exp_citation_lower = exp_citation.lower()
                    
                    # Basic matching - check if one contains the other
                    if ext_citation_lower in exp_citation_lower or exp_citation_lower in ext_citation_lower:
                        matched += 1
                        break
            
            precision = matched / len(extracted_citations) if extracted_citations else 0
            recall = matched / len(normalized_expected) if normalized_expected else 0
            
            if precision + recall == 0:
                return 0.0
                    
            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall)
            return min(1.0, f1_score)
        except Exception as e:
            print(f"Error calculating citation score: {str(e)}")
            traceback.print_exc()
            return 0.0
    
    #this function evaluates document retrieval quality
    # precision : what percentage of retrieved documents are relevant?
    # recall : what percentage of relevant documents were retrieved?
    # f1 : balanced mean of precision and recall
    # MRR (mean reciprocal rank) : measures how early the first relevant document appears in the retrieved list
    def calculate_retrieval_metrics(self, retrieved_docs, relevant_docs):
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
            
        # Calculate precision and recall
        relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
        precision = len(relevant_retrieved) / len(retrieved_docs)
        recall = len(relevant_retrieved) / len(relevant_docs)
        
        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                mrr = 1.0 / (i + 1)
                break
                
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr
        }
    
    # calculates a composite hallucination score  
    # combines semantic similarity and citation accuracy using the configurable weights
    # inverts the score so that higher values will indicate more hallucination
    # clamping to ensure the score remains between 0-1
    def calculate_hallucination_score(self, response, ground_truth, citation_score):
        """Calculate hallucination score using weights"""
        try:
            semantic_sim = self.compute_semantic_similarity(response, ground_truth)
            
            # Calculate weighted score
            hallucination_score = 1.0 - (
                (self.hallucination_weights["semantic_similarity"] * semantic_sim) + 
                (self.hallucination_weights["citation_score"] * citation_score)
            )
            
            return max(0.0, min(1.0, hallucination_score))
        except Exception as e:
            print(f"Error calculating hallucination score: {str(e)}")
            return 1.0
    
    #this function extracts document IDs from the sources text
    # it uses regex to find patterns that match the expected format of document IDs
    # it returns a list of unique document IDs found in the sources text
    # it will return an empty list if no document IDs are found or if the sources text is empty
    def extract_document_ids(self, sources_text):
        """Extract document IDs from sources text"""
        if not sources_text:
            return []
            
        doc_matches = re.findall(r'Document ID: ([^\s,]+)', sources_text)
        return [match for match in doc_matches if match]
    
    # the complete evaluation pipeline
    # processes each test question thru the RAG system
    # calculates all the metrics (semantic similarity, citation score, hallucination score)
    # hadnles errors and retries if necessary
    # saves intermediate results to a file every 5 questions
    def evaluate_with_rag(self, test_data, output_file, limit=5, max_retries=2):
        """Run test questions through RAG system and evaluate results"""
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
                    })
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error processing question (attempt {retry+1}/{max_retries}): {error_msg}")
                    traceback.print_exc()
                    
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
                        })
                    else:
                        # Wait before retry
                        time.sleep(2)
            
            # Save intermediate results every 5 questions
            if (idx + 1) % 5 == 0 or idx == total_questions - 1:
                self.save_results(results, output_file)
                print(f"Saved intermediate results after processing {idx+1}/{total_questions} questions")
        
        # Final save of results
        self.save_results(results, output_file)
        return results
    
    # this function evaluates the baseline responses without using the RAG system
    # it loads the baseline responses from a file and compares them against the test data
    # it calculates the same metrics as the RAG evaluation (semantic similarity, citation score, hallucination score)
    def evaluate_without_rag(self, test_data, llm_responses_file, output_file):
        # Load baseline responses
        try:
            with open(llm_responses_file, 'r', encoding='utf-8') as f:
                baseline_responses = json.load(f)
            print(f"Successfully loaded {len(baseline_responses)} baseline responses from {llm_responses_file}")
        except Exception as e:
            print(f"Error loading baseline responses from {llm_responses_file}: {str(e)}")
            traceback.print_exc()
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
                })
                
                # Save intermediate results every 5 questions
                if (i + 1) % 5 == 0 or i == total_questions - 1:
                    self.save_results(results, output_file)
                    print(f"Saved intermediate results after processing {i+1}/{total_questions} baseline responses")
                    
            except Exception as e:
                print(f"Error processing baseline question {i}: {str(e)}")
                traceback.print_exc()
                # Add error entry to maintain alignment with test data
                results.append({
                    "question": test_data[i]["question"] if i < len(test_data) else f"Unknown question {i}",
                    "question_type": "Unknown",
                    "ground_truth": test_data[i]["ground_truth"] if i < len(test_data) else "",
                    "baseline_answer": "ERROR: Failed to process baseline answer",
                    "expected_citations": [],
                    "extracted_citations": [],
                    "semantic_similarity": 0.0,
                    "citation_score": 0.0,
                    "hallucination_score": 1.0  # Max hallucination for errors
                })
        
        # Final save of results
        self.save_results(results, output_file)
        return results

    # Aligns RAG and baseline results for comparison
    # it ensures that the results are matched based on the question text
    # it returns two lists of aligned results (RAG and baseline) for further analysis
    # it will return empty lists if either of the results is empty or if there are no common questions
    # it will also print the number of aligned questions for debugging purposes
    def align_results_for_comparison(self, rag_results, baseline_results):
        if not rag_results or not baseline_results:
            return [], []
            
        aligned_rag = []
        aligned_baseline = []
        
        # Create dictionaries with question as key for easier lookup
        rag_dict = {r["question"]: r for r in rag_results}
        baseline_dict = {r["question"]: r for r in baseline_results}
        
        # Get all unique questions
        all_questions = set(rag_dict.keys()) & set(baseline_dict.keys())  # Only questions in both sets
        
        # For each question, add corresponding results to aligned lists
        for question in all_questions:
            aligned_rag.append(rag_dict[question])
            aligned_baseline.append(baseline_dict[question])
        
        print(f"Aligned {len(aligned_rag)} questions for comparison (out of {len(rag_results)} RAG and {len(baseline_results)} baseline)")
        return aligned_rag, aligned_baseline
    
    def calculate_overall_metrics(self, results):
        """Calculate overall metrics from evaluation results"""
        if not results:
            return {}
            
        metrics = {
            "total_questions": len(results),
            "errors": sum(1 for r in results if "ERROR:" in r.get("rag_answer", "")),
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
            "by_question_type": {}
        }
        
        # Count valid results (non-error)
        valid_results = [r for r in results if "ERROR:" not in r.get("rag_answer", "")]
        valid_count = len(valid_results)
        
        if valid_count > 0:
            # Calculate overall averages from valid results
            metrics["avg_semantic_similarity"] = sum(r.get("semantic_similarity", 0.0) for r in valid_results) / valid_count
            metrics["avg_citation_score"] = sum(r.get("citation_score", 0.0) for r in valid_results) / valid_count
            metrics["avg_hallucination_score"] = sum(r.get("hallucination_score", 1.0) for r in valid_results) / valid_count
            metrics["avg_processing_time"] = sum(r.get("processing_time", 0.0) for r in results) / len(results)
            
            # Calculate average retrieval metrics
            retrieval_sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}
            retrieval_counts = {"precision": 0, "recall": 0, "f1": 0, "mrr": 0}
            
            for r in valid_results:
                if "retrieval_metrics" in r:
                    for metric, value in r["retrieval_metrics"].items():
                        retrieval_sums[metric] += value
                        retrieval_counts[metric] += 1
            
            for metric in retrieval_sums:
                if retrieval_counts[metric] > 0:
                    metrics["retrieval_metrics"][f"avg_{metric}"] = retrieval_sums[metric] / retrieval_counts[metric]
        
        # Group by question type
        question_types = {}
        for r in results:
            q_type = r.get("question_type", "Unknown")
            if q_type not in question_types:
                question_types[q_type] = []
            question_types[q_type].append(r)
        
        # Calculate metrics by question type
        for q_type, type_results in question_types.items():
            valid_type_results = [r for r in type_results if "ERROR:" not in r.get("rag_answer", "")]
            valid_type_count = len(valid_type_results)
            
            type_metrics = {
                "count": len(type_results),
                "errors": sum(1 for r in type_results if "ERROR:" in r.get("rag_answer", "")),
            }
            
            if valid_type_count > 0:
                type_metrics["avg_semantic_similarity"] = sum(r.get("semantic_similarity", 0.0) for r in valid_type_results) / valid_type_count
                type_metrics["avg_citation_score"] = sum(r.get("citation_score", 0.0) for r in valid_type_results) / valid_type_count
                type_metrics["avg_hallucination_score"] = sum(r.get("hallucination_score", 1.0) for r in valid_type_results) / valid_type_count
            
            metrics["by_question_type"][q_type] = type_metrics
            
        return metrics
    
    # this function performs statistical tests to compare the RAG results with the baseline results
    # runs paired t-tests to check if the differences in metrics are statistically significant
    # calculates effect sizes (cohen's d) to quantify the magnitude of differences
    # determines if improvements are statistically significant
    # returns a dictionary with test results for each metric
    def perform_statistical_tests(self, rag_results, baseline_results):
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
            
        # Extract metrics for comparison
        rag_metrics = {
            "semantic_similarity": [r.get("semantic_similarity", 0.0) for r in aligned_rag],
            "citation_score": [r.get("citation_score", 0.0) for r in aligned_rag],
            "hallucination_score": [r.get("hallucination_score", 1.0) for r in aligned_rag]
        }
        
        baseline_metrics = {
            "semantic_similarity": [r.get("semantic_similarity", 0.0) for r in aligned_baseline],
            "citation_score": [r.get("citation_score", 0.0) for r in aligned_baseline],
            "hallucination_score": [r.get("hallucination_score", 1.0) for r in aligned_baseline]
        }
        
        # Perform paired t-tests
        test_results = {
            "valid_comparison": True,
            "sample_size": len(aligned_rag),
            "tests": {}
        }
        
        for metric in rag_metrics:
            try:
                t_stat, p_value = stats.ttest_rel(rag_metrics[metric], baseline_metrics[metric])
                
                # Calculate effect size (Cohen's d for paired samples)
                diff = np.array(rag_metrics[metric]) - np.array(baseline_metrics[metric])
                effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                
                # Is RAG better? (direction depends on metric)
                if metric == "hallucination_score":
                    # For hallucination, lower is better
                    rag_better = np.mean(rag_metrics[metric]) < np.mean(baseline_metrics[metric])
                else:
                    # For others, higher is better
                    rag_better = np.mean(rag_metrics[metric]) > np.mean(baseline_metrics[metric])
                
                test_results["tests"][metric] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "effect_size": float(effect_size),
                    "effect_magnitude": self._interpret_effect_size(effect_size),
                    "rag_mean": float(np.mean(rag_metrics[metric])),
                    "baseline_mean": float(np.mean(baseline_metrics[metric])),
                    "rag_better": rag_better
                }
            except Exception as e:
                print(f"Error performing statistical test for {metric}: {str(e)}")
                test_results["tests"][metric] = {
                    "error": str(e)
                }
        
        return test_results
    
    def _interpret_effect_size(self, effect_size):
        """Interpret Cohen's d effect size magnitude"""
        effect_size = abs(effect_size)
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
        
