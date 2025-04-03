import os
import argparse
import json
import time
import logging
import numpy as np
from tqdm import tqdm
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_evaluation")

# Define delay between API requests
REQUEST_DELAY = 5.0  # seconds

def setup_argparse():
    """Set up command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Enhanced evaluation of RAG implementation against baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--test_file", default="data/test_questions.json", 
                        help="Path to test questions file")
    parser.add_argument("--baseline_file", default="data/baseline_responses.json", 
                        help="Path to baseline responses file (if available)")
    parser.add_argument("--rag_results_file", default=None,
                        help="Path to pre-existing RAG results file for report generation")
    parser.add_argument("--baseline_results_file", default=None,
                        help="Path to pre-existing baseline results file for report generation")
    
    # Output files
    parser.add_argument("--rag_output", default="eval_results/rag_results.json", 
                        help="Path to save RAG evaluation results")
    parser.add_argument("--baseline_output", default="eval_results/baseline_results.json", 
                        help="Path to save baseline evaluation results")
    parser.add_argument("--report_output", default="eval_results/evaluation_report.html", 
                        help="Path to save HTML evaluation report")
    
    # RAG parameters
    parser.add_argument("--num_docs", type=int, default=5, 
                        help="Number of documents to retrieve per query")
    parser.add_argument("--embedding_model", default="dangvantuan/vietnamese-document-embedding", 
                        help="Embedding model for semantic similarity")
    
    # Evaluation control
    parser.add_argument("--only_rag", action="store_true", 
                        help="Only evaluate RAG, skip baseline comparison")
    parser.add_argument("--only_baseline", action="store_true", 
                        help="Only evaluate baseline, skip RAG")
    parser.add_argument("--report_only", action="store_true",
                        help="Generate report from existing results only")
    parser.add_argument("--limit_questions", type=int, default=0, 
                        help="Limit number of questions to evaluate (0 = no limit)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable embedding caching")
    parser.add_argument("--reprocess_results", action="store_true",
                        help="Reprocess existing results with improved metrics")
    
    # Hallucination score weights
    parser.add_argument("--sem_sim_weight", type=float, default=0.7,
                        help="Weight for semantic similarity in hallucination score")
    parser.add_argument("--citation_weight", type=float, default=0.3,
                        help="Weight for citation score in hallucination score")
    
    return parser.parse_args()

def load_existing_results(file_path):
    """Load existing evaluation results from JSON file"""
    if not file_path or not os.path.exists(file_path):
        logger.warning(f"Results file not found: {file_path}")
        return None
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} results from {file_path}")
        return results
    except Exception as e:
        logger.error(f"Error loading results from {file_path}: {str(e)}")
        return None

def save_results(results, file_path):
    """Save evaluation results to a JSON file with error handling"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved results to {file_path}")
    except Exception as e:
        logger.error(f"Error saving results to {file_path}: {str(e)}")
        
        # Save to a backup file in case of error
        backup_path = f"{file_path}.backup"
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved backup results to {backup_path}")
        except Exception as backup_error:
            logger.error(f"Failed to save backup results: {str(backup_error)}")

def ensure_output_dir(file_path):
    """Ensure output directory exists"""
    directory = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created output directory: {directory}")

def main():
    """Run the RAG evaluation process"""
    # Parse command-line arguments
    args = setup_argparse()
    
    # Import libraries here to avoid circular imports
    try:
        from evaluation.rag_evaluation import RAGEvaluator
        from evaluation.improved_rag_evaluation import DocumentMatcher, CitationMatcher, RAGMetricsCalculator
        from evaluation.improved_charts import EnhancedReportGenerator
    except ImportError as e:
        logger.error(f"Error importing required modules: {str(e)}")
        logger.error("Please ensure all required modules are installed.")
        return
    
    # Record start time
    start_time = time.time()
    logger.info(f"Starting evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure output directories exist
    ensure_output_dir(args.rag_output)
    ensure_output_dir(args.baseline_output)
    ensure_output_dir(args.report_output)
    
    # Configure hallucination weights
    hallucination_weights = {
        "semantic_similarity": args.sem_sim_weight,
        "citation_score": args.citation_weight,
    }
    
    # Create metrics calculator for improved metrics
    metrics_calculator = RAGMetricsCalculator()
    
    rag_results = None
    baseline_results = None
    
    # Load existing results if specified
    if args.report_only or args.reprocess_results:
        rag_file = args.rag_results_file or args.rag_output
        baseline_file = args.baseline_results_file or args.baseline_output
        
        rag_results = load_existing_results(rag_file)
        baseline_results = load_existing_results(baseline_file)
        
        if not rag_results and not baseline_results:
            logger.error("No valid results loaded for report generation")
            return
    
    # Reprocess existing results with improved metrics
    if args.reprocess_results:
        if rag_results:
            logger.info("Reprocessing RAG results with improved metrics...")
            
            # Process RAG results in batches to show progress
            updated_rag_results = []
            batch_size = 10
            
            for i in tqdm(range(0, len(rag_results), batch_size), desc="Reprocessing RAG results"):
                batch = rag_results[i:i+batch_size]
                updated_batch = metrics_calculator.batch_update_results(batch)
                updated_rag_results.extend(updated_batch)
            
            # Save updated results
            save_results(updated_rag_results, args.rag_output)
            rag_results = updated_rag_results
            logger.info(f"Saved {len(updated_rag_results)} reprocessed RAG results")
        
        if baseline_results:
            logger.info("Reprocessing baseline results with improved metrics...")
            
            # Process baseline results in batches to show progress
            updated_baseline_results = []
            
            for i in tqdm(range(0, len(baseline_results), batch_size), desc="Reprocessing baseline results"):
                batch = baseline_results[i:i+batch_size]
                updated_batch = metrics_calculator.batch_update_results(batch)
                updated_baseline_results.extend(updated_batch)
            
            # Save updated results
            save_results(updated_baseline_results, args.baseline_output)
            baseline_results = updated_baseline_results
            logger.info(f"Saved {len(updated_baseline_results)} reprocessed baseline results")
    
    # Run regular evaluation if not in report_only or reprocess_only mode
    if not args.report_only and not args.reprocess_results:
        # Set up evaluator
        evaluator = RAGEvaluator(
            embedding_model_name=args.embedding_model,
            hallucination_weights=hallucination_weights,
            use_cache=not args.no_cache
        )
        
        # Load test data
        test_data = evaluator.load_test_data(args.test_file)
        
        if not test_data:
            logger.error("No test data loaded. Exiting.")
            return
            
        # Apply limit if specified
        if args.limit_questions > 0 and args.limit_questions < len(test_data):
            logger.info(f"Limiting evaluation to first {args.limit_questions} questions")
            test_data = test_data[:args.limit_questions]
        
        # Evaluate with RAG
        if not args.only_baseline:
            logger.info("=== Starting RAG evaluation ===")
            rag_results = evaluator.evaluate_with_rag(
                test_data, 
                args.rag_output, 
                limit=args.num_docs
            )
            
            # Update RAG results with improved metrics
            logger.info("Updating RAG results with improved metrics...")
            rag_results = metrics_calculator.batch_update_results(rag_results)
            save_results(rag_results, args.rag_output)
            
            time.sleep(REQUEST_DELAY)
        
        # Evaluate baseline
        if not args.only_rag and os.path.exists(args.baseline_file):
            logger.info("=== Starting baseline evaluation ===")
            baseline_results = evaluator.evaluate_without_rag(
                test_data, 
                args.baseline_file, 
                args.baseline_output
            )
            
            # Update baseline results with improved metrics
            logger.info("Updating baseline results with improved metrics...")
        if baseline_results is not None:
            logger.info("Reprocessing baseline results with improved metrics...")
            
            # Process baseline results in batches to show progress
            updated_baseline_results = []
            batch_size = 10  # Make sure this is defined earlier
            
            for i in tqdm(range(0, len(baseline_results), batch_size), desc="Reprocessing baseline results"):
                batch = baseline_results[i:i+batch_size]
                updated_batch = metrics_calculator.batch_update_results(batch)
                updated_baseline_results.extend(updated_batch)
            
            # Save updated results
            save_results(updated_baseline_results, args.baseline_output)
            baseline_results = updated_baseline_results
            logger.info(f"Saved {len(updated_baseline_results)} reprocessed baseline results")
        else:
            logger.info("No baseline results to process")
            save_results(baseline_results, args.baseline_output)
            
            time.sleep(REQUEST_DELAY)
    
    # Generate report
    if rag_results or baseline_results:
        # Calculate metrics
        logger.info("Calculating aggregate metrics...")
        rag_metrics = metrics_calculator.calculate_aggregate_metrics(rag_results) if rag_results else None
        baseline_metrics = metrics_calculator.calculate_aggregate_metrics(baseline_results) if baseline_results else None
        
        # Print summary of RAG metrics
        if rag_metrics and "overall" in rag_metrics:
            logger.info("=== RAG Metrics Summary ===")
            overall = rag_metrics["overall"]
            for metric, value in overall.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")
        
        # Generate report
        logger.info("=== Generating evaluation report ===")
        report_generator = EnhancedReportGenerator()
        report_generator.generate_html_report(
            rag_results=rag_results,
            rag_metrics=rag_metrics,
            baseline_results=baseline_results,
            baseline_metrics=baseline_metrics,
            output_file=args.report_output
        )
    
    total_time = time.time() - start_time
    logger.info(f"Evaluation completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()