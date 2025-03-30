import os
import argparse
import json
import time
from rag_evaluation import RAGEvaluator
from report_generation import ReportGenerator

# the command line interface and main execution flow for rag system
# parsing command line arguments that control the evaluation process
# initializing and coordinating the evalution components
# runs the evaluations for both RAG and baseline systems
# generates a report summarizing the results
# this script is basically the glue that connects the evaluation lofic and reporting system

def main():
    """Run the simplified RAG evaluation process with command line arguments"""
    parser = argparse.ArgumentParser(description="Simplified evaluation of RAG implementation against baseline")
    
    # Input files
    parser.add_argument("--test_file", default="test_questions.json", 
                        help="Path to test questions file")
    parser.add_argument("--baseline_file", default="baseline_responses.json", 
                        help="Path to baseline responses file (if available)")
    
    # Output files
    parser.add_argument("--rag_output", default="rag_results.json", 
                        help="Path to save RAG evaluation results")
    parser.add_argument("--baseline_output", default="baseline_results.json", 
                        help="Path to save baseline evaluation results")
    parser.add_argument("--report_output", default="evaluation_report.html", 
                        help="Path to save HTML evaluation report")
    
    # RAG parameters
    parser.add_argument("--num_docs", type=int, default=5, 
                        help="Number of documents to retrieve per query")
    parser.add_argument("--embedding_model", default="keepitreal/vietnamese-sbert", 
                        help="Embedding model for semantic similarity")
    
    # Evaluation control
    parser.add_argument("--only_rag", action="store_true", 
                        help="Only evaluate RAG, skip baseline comparison")
    parser.add_argument("--only_baseline", action="store_true", 
                        help="Only evaluate baseline, skip RAG")
    parser.add_argument("--limit_questions", type=int, default=0, 
                        help="Limit number of questions to evaluate (0 = no limit)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable embedding caching")
    
    # Hallucination score weights
    parser.add_argument("--sem_sim_weight", type=float, default=0.7,
                        help="Weight for semantic similarity in hallucination score")
    parser.add_argument("--citation_weight", type=float, default=0.3,
                        help="Weight for citation score in hallucination score")
    
    args = parser.parse_args()
    
    start_time = time.time()
    print(f"Starting evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configure hallucination weights
    hallucination_weights = {
        "semantic_similarity": args.sem_sim_weight,
        "citation_score": args.citation_weight,
    }
    
    # set up phase, parses arguments and initializes the evaluator with specified configuration
    evaluator = RAGEvaluator(
        embedding_model_name=args.embedding_model,
        hallucination_weights=hallucination_weights,
        use_cache=not args.no_cache
    )
    
    # Load test data
    test_data = evaluator.load_test_data(args.test_file)
    
    # Apply limit if specified
    if args.limit_questions > 0 and args.limit_questions < len(test_data):
        print(f"Limiting evaluation to first {args.limit_questions} questions")
        test_data = test_data[:args.limit_questions]
    
    rag_results = None
    baseline_results = None
    
    # Evaluate with RAG
    if not args.only_baseline:
        print("\n=== Starting RAG evaluation ===")
        rag_results = evaluator.evaluate_with_rag(
            test_data, 
            args.rag_output, 
            limit=args.num_docs
        )
    
    # Evaluate baseline
    if not args.only_rag and os.path.exists(args.baseline_file):
        print("\n=== Starting baseline evaluation ===")
        baseline_results = evaluator.evaluate_without_rag(
            test_data, 
            args.baseline_file, 
            args.baseline_output
        )
    
    # Calculate metrics and generate report
    if rag_results:
        # Calculate overall RAG metrics
        rag_metrics = evaluator.calculate_overall_metrics(rag_results)
        
        # Calculate baseline metrics if available
        baseline_metrics = None
        if baseline_results:
            baseline_metrics = evaluator.calculate_overall_metrics(baseline_results)
        
        # Perform statistical tests if both RAG and baseline results are available
        comparison_stats = None
        if baseline_results:
            comparison_stats = evaluator.perform_statistical_tests(rag_results, baseline_results)
        
        # Initialize report generator
        report_gen = ReportGenerator()
        
        # Generate charts
        charts = report_gen.generate_charts(rag_results, baseline_results)
        
        # Generate HTML report
        print("\n=== Generating evaluation report ===")
        report_gen.generate_html_report(
            rag_results=rag_results,
            rag_metrics=rag_metrics,
            baseline_results=baseline_results,
            baseline_metrics=baseline_metrics,
            comparison_stats=comparison_stats,
            charts=charts,
            output_file=args.report_output
        )
        
        # Print summary metrics
        print("\n=== Evaluation Summary ===")
        print("RAG Performance:")
        for metric, value in rag_metrics.items():
            if metric not in ("by_question_type", "retrieval_metrics") and not isinstance(value, dict):
                if isinstance(value, float):
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {value}")
        
        # Print retrieval metrics
        if "retrieval_metrics" in rag_metrics:
            print("\nRetrieval Metrics:")
            for metric, value in rag_metrics["retrieval_metrics"].items():
                if isinstance(value, float):
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {value}")
        
        # Print comparison if available
        if comparison_stats and comparison_stats.get("valid_comparison", False):
            print("\n=== Comparison with Baseline ===")
            print(f"Sample size: {comparison_stats.get('sample_size', 0)} paired samples")
            
            for metric, test_results in comparison_stats.get("tests", {}).items():
                if "error" in test_results:
                    continue
                
                rag_value = test_results.get("rag_mean", 0.0)
                baseline_value = test_results.get("baseline_mean", 0.0)
                
                # Calculate improvement
                if metric == "hallucination_score":
                    # For hallucination, lower is better
                    improvement = ((baseline_value - rag_value) / max(0.0001, baseline_value)) * 100
                    better_text = "lower (better)" if rag_value < baseline_value else "higher (worse)"
                else:
                    # For others, higher is better
                    improvement = ((rag_value - baseline_value) / max(0.0001, baseline_value)) * 100
                    better_text = "higher (better)" if rag_value > baseline_value else "lower (worse)"
                
                p_value = test_results.get("p_value", 1.0)
                is_significant = test_results.get("significant", False)
                significance_text = "Significant" if is_significant else "Not significant"
                
                print(f"  {metric.replace('_', ' ').title()}: RAG: {rag_value:.4f}, Baseline: {baseline_value:.4f}")
                print(f"    {improvement:.2f}% {better_text} - p={p_value:.4f} ({significance_text})")
    
    total_time = time.time() - start_time
    print(f"\nEvaluation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Results saved to {args.report_output}")

if __name__ == "__main__":
    main()