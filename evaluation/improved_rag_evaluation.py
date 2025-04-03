import re
import numpy as np
from collections import defaultdict

class DocumentMatcher:
    """Improved document matching for RAG evaluation"""
    
    @staticmethod
    def normalize_doc_id(doc_id):
        """Normalize document IDs for better matching"""
        if not doc_id:
            return ""
            
        doc_id = str(doc_id).lower().strip()
        
        # Handle special cases with asterisks or other non-document IDs
        if doc_id in ["**", "*"]:
            return ""
            
        # Extract document type and number
        doc_type = None
        doc_num = None
        
        # Check for TT/ND/Dieu pattern
        tt_match = re.search(r'(?:tt|thong\s*tu)[_\s]*(\d+)', doc_id)
        if tt_match:
            doc_type = "tt"
            doc_num = tt_match.group(1)
        
        nd_match = re.search(r'(?:nd|nghi\s*dinh)[_\s]*(\d+)', doc_id)
        if nd_match:
            doc_type = "nd"
            doc_num = nd_match.group(1)
            
        dieu_match = re.search(r'(?:dieu|điều|d)[_\s]*(\d+)', doc_id)
        if dieu_match:
            doc_type = "dieu"
            doc_num = dieu_match.group(1)
            
        # If we have both type and number, return normalized format
        if doc_type and doc_num:
            return f"{doc_type}_{doc_num}"
            
        # Otherwise, just clean up the string
        doc_id = re.sub(r'[_\-\s]', '_', doc_id)
        return doc_id
    
    @staticmethod
    def get_document_components(doc_id):
        """Extract document type and number components"""
        doc_id = str(doc_id).lower().strip()
        
        components = {
            "original": doc_id,
            "type": None,
            "number": None
        }
        
        # Extract document type
        if "tt" in doc_id or "thong" in doc_id or "thông" in doc_id:
            components["type"] = "tt"
        elif "nd" in doc_id or "nghi" in doc_id or "nghị" in doc_id:
            components["type"] = "nd"
        elif "dieu" in doc_id or "điều" in doc_id or "d_" in doc_id:
            components["type"] = "dieu"
            
        # Extract number
        num_match = re.search(r'(\d+)', doc_id)
        if num_match:
            components["number"] = num_match.group(1)
            
        return components
    
    @staticmethod
    def calculate_faithfulness(retrieved_docs, relevant_docs):
        """
        Calculate faithfulness with improved matching
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            
        Returns:
            float: Faithfulness score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
        
        if not relevant_docs:
            return 1.0  # If no relevant docs expected, perfect score
            
        # Clean and normalize document IDs
        normalized_retrieved = [DocumentMatcher.normalize_doc_id(doc) for doc in retrieved_docs]
        normalized_relevant = [DocumentMatcher.normalize_doc_id(doc) for doc in relevant_docs]
        
        # Remove empty strings
        normalized_retrieved = [doc for doc in normalized_retrieved if doc]
        normalized_relevant = [doc for doc in normalized_relevant if doc]
        
        if not normalized_retrieved:
            return 0.0
        
        # Get components for partial matching
        retrieved_components = [DocumentMatcher.get_document_components(doc) for doc in normalized_retrieved]
        relevant_components = [DocumentMatcher.get_document_components(doc) for doc in normalized_relevant]
        
        # Calculate exact matches
        exact_matches = len(set(normalized_retrieved).intersection(set(normalized_relevant)))
        
        # Calculate partial matches (when document type or number matches)
        partial_matches = 0
        
        for r_comp in retrieved_components:
            for rel_comp in relevant_components:
                # Skip if either component is missing type or number
                if not r_comp["type"] or not r_comp["number"] or not rel_comp["type"] or not rel_comp["number"]:
                    continue
                
                # Check for partial matches (same type or same number)
                if r_comp["type"] == rel_comp["type"] and r_comp["number"] != rel_comp["number"]:
                    partial_matches += 0.5  # Same document type, different number
                elif r_comp["type"] != rel_comp["type"] and r_comp["number"] == rel_comp["number"]:
                    partial_matches += 0.3  # Different document type, same number
        
        # Calculate combined score
        match_score = exact_matches + 0.5 * partial_matches
        
        # Normalize by number of retrieved docs
        faithfulness = min(1.0, match_score / len(normalized_retrieved))
        
        return faithfulness

class CitationMatcher:
    """Improved citation matching for RAG evaluation"""
    
    @staticmethod
    def normalize_citation(citation):
        """Normalize citation string for better matching"""
        if not citation:
            return ""
            
        citation = str(citation).lower().strip()
        
        # Standardize spacing
        citation = re.sub(r'\s+', ' ', citation)
        citation = re.sub(r'[,;:\.]', ' ', citation)
        
        # Standardize document types
        citation = re.sub(r'th[oô]ng\s*t[uư]|tt', 'thong tu', citation)
        citation = re.sub(r'ngh[iị]\s*[dđ][iị]nh|nd', 'nghi dinh', citation)
        citation = re.sub(r'[dđ]i[eêề]u', 'dieu', citation)
        
        return citation.strip()
    
    @staticmethod
    def extract_components(citation):
        """Extract document type and number components from citation"""
        components = []
        
        # Find all document type + number pairs
        patterns = [
            (r'(thong tu|tt)[^\d]*(\d+)', 'thong tu'),
            (r'(nghi dinh|nd)[^\d]*(\d+)', 'nghi dinh'),
            (r'(dieu|điều)[^\d]*(\d+)', 'dieu')
        ]
        
        for pattern, doc_type in patterns:
            matches = re.findall(pattern, citation.lower())
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    number = match[1]
                    components.append({
                        "type": doc_type,
                        "number": number
                    })
        
        return components
    
    @staticmethod
    def calculate_citation_score(extracted_citations, expected_citations):
        """
        Calculate citation score with improved matching
        
        Args:
            extracted_citations: List of citations found in response
            expected_citations: List of expected citations
            
        Returns:
            float: Citation score between 0 and 1
        """
        # Handle empty cases
        if not expected_citations:
            return 1.0 if not extracted_citations else 0.5
        
        if not extracted_citations:
            return 0.0
        
        # Normalize citations
        norm_extracted = [CitationMatcher.normalize_citation(c) for c in extracted_citations]
        norm_expected = [CitationMatcher.normalize_citation(c) for c in expected_citations]
        
        # Extract components for matching
        extracted_components = []
        for citation in norm_extracted:
            extracted_components.extend(CitationMatcher.extract_components(citation))
            
        expected_components = []
        for citation in norm_expected:
            expected_components.extend(CitationMatcher.extract_components(citation))
        
        # No components found
        if not extracted_components or not expected_components:
            # Fall back to string matching
            exact_matches = sum(1 for e in norm_extracted if any(e in ex for ex in norm_expected))
            return min(1.0, exact_matches / len(norm_extracted))
        
        # Calculate matches with components
        matched_extracted = 0
        
        for ex_comp in extracted_components:
            # Look for exact matches
            exact_match = any(
                ex_comp["type"] == exp_comp["type"] and ex_comp["number"] == exp_comp["number"]
                for exp_comp in expected_components
            )
            
            if exact_match:
                matched_extracted += 1
                continue
                
            # Look for partial matches
            best_match = 0
            for exp_comp in expected_components:
                if ex_comp["type"] == exp_comp["type"]:
                    best_match = max(best_match, 0.5)  # Same type
                if ex_comp["number"] == exp_comp["number"]:
                    best_match = max(best_match, 0.3)  # Same number
            
            matched_extracted += best_match
        
        # Calculate precision
        precision = matched_extracted / len(extracted_components) if extracted_components else 0
        
        # Calculate recall
        matched_expected = 0
        for exp_comp in expected_components:
            # Look for exact matches
            exact_match = any(
                exp_comp["type"] == ex_comp["type"] and exp_comp["number"] == ex_comp["number"]
                for ex_comp in extracted_components
            )
            
            if exact_match:
                matched_expected += 1
                continue
                
            # Look for partial matches
            best_match = 0
            for ex_comp in extracted_components:
                if exp_comp["type"] == ex_comp["type"]:
                    best_match = max(best_match, 0.5)  # Same type
                if exp_comp["number"] == ex_comp["number"]:
                    best_match = max(best_match, 0.3)  # Same number
            
            matched_expected += best_match
        
        recall = matched_expected / len(expected_components) if expected_components else 0
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
            
        f1_score = 2 * (precision * recall) / (precision + recall)
        return min(1.0, f1_score)

class RAGMetricsCalculator:
    """Enhanced calculator for RAG evaluation metrics"""
    
    def __init__(self):
        """Initialize the metrics calculator"""
        self.document_matcher = DocumentMatcher()
        self.citation_matcher = CitationMatcher()
    
    def recalculate_faithfulness(self, result):
        """Recalculate faithfulness metric for a result"""
        retrieved_docs = result.get("retrieved_docs", [])
        relevant_docs = result.get("relevant_docs", [])
        
        faithfulness = DocumentMatcher.calculate_faithfulness(retrieved_docs, relevant_docs)
        return faithfulness
    
    def recalculate_citation_score(self, result):
        """Recalculate citation score for a result"""
        extracted_citations = result.get("extracted_citations", [])
        expected_citations = result.get("expected_citations", [])
        
        citation_score = CitationMatcher.calculate_citation_score(extracted_citations, expected_citations)
        return citation_score
    
    def recalculate_result_metrics(self, result):
        """Recalculate all metrics for a result"""
        # Deep copy to avoid modifying original
        updated_result = result.copy()
        
        # Recalculate faithfulness
        faithfulness = self.recalculate_faithfulness(result)
        updated_result["rag_faithfulness"] = faithfulness
        
        # Recalculate citation score (if needed)
        recalculated_citation = self.recalculate_citation_score(result)
        if abs(recalculated_citation - result.get("citation_score", 0)) > 0.1:
            updated_result["citation_score"] = recalculated_citation
        
        # Recalculate hallucination score
        semantic_similarity = result.get("semantic_similarity", 0.0)
        citation_score = updated_result.get("citation_score", 0.0)
        
        # Use weighted combination (70% semantic, 30% citation)
        hallucination_score = 1.0 - (0.7 * semantic_similarity + 0.3 * citation_score)
        updated_result["hallucination_score"] = max(0.0, min(1.0, hallucination_score))
        
        # Recalculate RAG quality
        completeness = result.get("rag_completeness", semantic_similarity)
        precision = result.get("rag_precision", 0.0)
        source_utilization = result.get("rag_source_utilization", 0.0)
        
        # Use weighted combination for RAG quality
        rag_quality = (
            0.35 * faithfulness +
            0.35 * completeness +
            0.15 * precision +
            0.15 * source_utilization
        )
        updated_result["rag_quality"] = max(0.0, min(1.0, rag_quality))
        
        return updated_result
    
    def batch_update_results(self, results):
        """Update metrics for a batch of results"""
        if results is None:
            return []
            
        updated_results = []
        
        for result in results:
            updated_result = self.recalculate_result_metrics(result)
            updated_results.append(updated_result)
        
        return updated_results
    
    def calculate_aggregate_metrics(self, results):
        """Calculate aggregate metrics for all results"""
        if not results:
            return {}
            
        metrics = {}
        
        # Group results by question type
        question_types = defaultdict(list)
        for result in results:
            q_type = result.get("question_type", "Unknown")
            question_types[q_type].append(result)
        
        # Calculate overall metrics
        metrics["overall"] = self._calculate_type_metrics(results)
        
        # Calculate metrics by question type
        metrics["by_question_type"] = {}
        for q_type, type_results in question_types.items():
            metrics["by_question_type"][q_type] = self._calculate_type_metrics(type_results)
        
        return metrics
    
    def _calculate_type_metrics(self, results):
        """Calculate metrics for a group of results"""
        if not results:
            return {}
            
        # Extract metrics
        faithfulness = [r.get("rag_faithfulness", 0.0) for r in results]
        completeness = [r.get("rag_completeness", 0.0) for r in results]
        precision = [r.get("rag_precision", 0.0) for r in results]
        source_utilization = [r.get("rag_source_utilization", 0.0) for r in results]
        rag_quality = [r.get("rag_quality", 0.0) for r in results]
        
        semantic_similarity = [r.get("semantic_similarity", 0.0) for r in results]
        citation_score = [r.get("citation_score", 0.0) for r in results]
        hallucination_score = [r.get("hallucination_score", 1.0) for r in results]
        
        # Calculate averages
        type_metrics = {
            "count": len(results),
            "avg_faithfulness": np.mean(faithfulness),
            "avg_completeness": np.mean(completeness), 
            "avg_precision": np.mean(precision),
            "avg_source_utilization": np.mean(source_utilization),
            "avg_rag_quality": np.mean(rag_quality),
            "avg_semantic_similarity": np.mean(semantic_similarity),
            "avg_citation_score": np.mean(citation_score),
            "avg_hallucination_score": np.mean(hallucination_score)
        }
        
        return type_metrics

# Demo usage
def demo_faithfulness_calculation():
    """Demonstrate faithfulness calculation with examples"""
    examples = [
        {
            "retrieved_docs": ["**", "67", "46", "Dieu_38", "Dieu_41", "Dieu_40", "Dieu_43", "Dieu_35"],
            "relevant_docs": ["ND_46", "Dieu_41", "TT_67", "Dieu_38"],
            "expected": "Very low faithfulness - Different formats!"
        },
        {
            "retrieved_docs": ["TT_67", "ND_46", "Dieu_38", "Dieu_41"],
            "relevant_docs": ["TT_67", "ND_46", "Dieu_38", "Dieu_41"],
            "expected": "Perfect faithfulness - Exact match!"
        },
        {
            "retrieved_docs": ["TT_67", "ND_46", "Dieu_38", "Dieu_41", "Dieu_42"],
            "relevant_docs": ["TT_67", "ND_46", "Dieu_38", "Dieu_41"],
            "expected": "High faithfulness - Most docs match!"
        },
        {
            "retrieved_docs": ["Thông tư 67", "Nghị định 46", "Điều 38", "Điều 41"],
            "relevant_docs": ["TT_67", "ND_46", "Dieu_38", "Dieu_41"],
            "expected": "Perfect faithfulness - Different format but same content!"
        }
    ]
    
    print("Document Matcher Demo:")
    for i, example in enumerate(examples, 1):
        faith_score = DocumentMatcher.calculate_faithfulness(
            example["retrieved_docs"],
            example["relevant_docs"]
        )
        print(f"Example {i}: Score = {faith_score:.4f} - {example['expected']}")
        print(f"  Retrieved: {example['retrieved_docs']}")
        print(f"  Relevant:  {example['relevant_docs']}")
        
    print("\nNormalization examples:")
    test_docs = ["TT_67", "Thông tư 67", "tt67", "Thông tư số 67", "THONG TU 67", "67", "**"]
    for doc in test_docs:
        print(f"  Original: '{doc}' → Normalized: '{DocumentMatcher.normalize_doc_id(doc)}'")

if __name__ == "__main__":
    demo_faithfulness_calculation()