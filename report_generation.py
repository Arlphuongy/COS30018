import os
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# this file basically implements a comprehensive report for visualizing and presenting the results  
# creates visual charts showing different performance metrics
# sligtly interactive HTML report that presents the findings in a digestible format
# statistical comparisons between rag and baseline system 

# this standalone function creates low level charts using plotly's graph objects api
# it creates a scatter plot showing the correlation between semantic similarity and hallucination score
# builds a histogram of processing times grouped by question type
# uses explicit data to avoid issues with the pandas data frame manipulation
def create_direct_charts(rag_results):
    charts = {}
    
    # 1. Create direct correlation chart
    try:
        # Extract data directly
        x_values = [r.get("semantic_similarity", 0.0) for r in rag_results]
        y_values = [r.get("hallucination_score", 1.0) for r in rag_results]
        question_types = [r.get("question_type", "Unknown") for r in rag_results]
        
        # Create figure directly with go.Scatter
        fig1 = go.Figure()
        
        # Group by question type for different colors
        question_type_set = set(question_types)
        for q_type in question_type_set:
            indices = [i for i, qt in enumerate(question_types) if qt == q_type]
            fig1.add_trace(go.Scatter(
                x=[x_values[i] for i in indices],
                y=[y_values[i] for i in indices],
                mode='markers',
                marker=dict(size=10, opacity=0.7),
                name=q_type
            ))
        
        fig1.update_layout(
            title="Correlation: Semantic Similarity vs Hallucination",
            xaxis=dict(title="Semantic Similarity", range=[0, 1]),
            yaxis=dict(title="Hallucination Score", range=[0, 1])
        )
        
        charts["metric_correlation"] = fig1
    except Exception as e:
        print(f"Error creating direct correlation chart: {str(e)}")
    
    # 2. Create direct processing time histogram
    try:
        processing_times = [r.get("processing_time", 0.0) for r in rag_results]
        question_types = [r.get("question_type", "Unknown") for r in rag_results]
        
        fig2 = go.Figure()
        
        # Create a simple histogram
        for q_type in set(question_types):
            type_times = [t for i, t in enumerate(processing_times) if question_types[i] == q_type]
            fig2.add_trace(go.Histogram(
                x=type_times,
                name=q_type,
                opacity=0.7,
                xbins=dict(
                    start=0,
                    end=max(processing_times) + 1,
                    size=0.5  # bin size of 0.5 seconds
                )
            ))
        
        fig2.update_layout(
            title="Distribution of Processing Times",
            xaxis=dict(title="Processing Time (seconds)"),
            yaxis=dict(title="Count"),
            barmode='overlay'
        )
        
        charts["processing_time"] = fig2
    except Exception as e:
        print(f"Error creating direct processing time chart: {str(e)}")
    
    return charts

# the main class for generating the HTML report
# two primary methods

# 1 generate charts for visualizing the results
# box plots showing hallucination scores by question type
# scatter plots showing the correlation between semantic similarity and hallucination score
# histograms displaying the distribution of processing times
# comparison charts contrasting RAG and baseline performance 
# more box plots for retreival metrics (precision, recall, f1, mrr)
# uses pandas dataframe for data manipulation and plotly for visualization

# 2 generate the HTML report itself, including the charts and other metrics
# the report is structured with sections for performance summary, comparison with baseline, error summary, and methodology
# the report is styled with CSS for better readability and presentation
class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_charts(self, rag_results, baseline_results=None):
        #Generate Plotly charts for visualization
        charts = {}
        
        # Skip chart generation if no valid results
        if not rag_results or all("ERROR:" in r.get("rag_answer", "") for r in rag_results):
            return charts
            
        # Convert results to DataFrame - only non-error results
        valid_results = [r for r in rag_results if "ERROR:" not in r.get("rag_answer", "")]
        
        if not valid_results:
            return charts
            
        rag_df = pd.DataFrame([{
            "question_id": i+1,
            "question": r["question"],
            "question_type": r.get("question_type", "Unknown"),
            "rag_similarity": r.get("semantic_similarity", 0.0),
            "rag_citation": r.get("citation_score", 0.0),
            "rag_hallucination": r.get("hallucination_score", 1.0),
            "processing_time": r.get("processing_time", 0)
        } for i, r in enumerate(valid_results)])
        
        # Ensure question_type exists
        if "question_type" not in rag_df.columns:
            rag_df["question_type"] = "Unknown"
        
        # 1. Hallucination scores by question type
        try:
            fig1 = px.box(rag_df, x="question_type", y="rag_hallucination", 
                        title="Hallucination Scores by Question Type",
                        labels={"question_type": "Question Type", "rag_hallucination": "Hallucination Score"},
                        color="question_type")
            fig1.update_layout(showlegend=False)
            charts["hallucination_by_type"] = fig1
        except Exception as e:
            print(f"Error creating hallucination chart: {str(e)}")
        
        # 2. Correlation between metrics
        try:
            # Drop rows with missing values
            corr_df = rag_df.dropna(subset=["rag_similarity", "rag_citation", "rag_hallucination"])
            
            if len(corr_df) > 5:  # Need minimum data points
                fig2 = px.scatter(corr_df, x="rag_similarity", y="rag_hallucination", 
                                color="question_type", size="rag_citation",
                                hover_data=["question_id", "question"],
                                title="Correlation: Semantic Similarity vs Hallucination",
                                labels={
                                    "rag_similarity": "Semantic Similarity", 
                                    "rag_hallucination": "Hallucination Score",
                                    "rag_citation": "Citation Score"
                                })
                charts["metric_correlation"] = fig2
        except Exception as e:
            print(f"Error creating correlation chart: {str(e)}")
        
        # 3. Processing time analysis
        try:
            fig3 = px.histogram(rag_df, x="processing_time", 
                              title="Distribution of Processing Times",
                              labels={"processing_time": "Processing Time (seconds)"},
                              color="question_type")
            charts["processing_time"] = fig3
        except Exception as e:
            print(f"Error creating processing time chart: {str(e)}")
            
        # 4. Comparison charts if baseline available
        if baseline_results:
            valid_baseline = [r for r in baseline_results if "ERROR:" not in r.get("baseline_answer", "")]
            
            if valid_baseline:
                try:
                    # Create comparison dataframe
                    comp_data = []
                    for rag, baseline in zip(valid_results, valid_baseline):
                        if rag["question"] == baseline["question"]:
                            comp_data.append({
                                "question": rag["question"],
                                "question_type": rag.get("question_type", "Unknown"),
                                "metric": "Semantic Similarity",
                                "rag_value": rag.get("semantic_similarity", 0.0),
                                "baseline_value": baseline.get("semantic_similarity", 0.0)
                            })
                            comp_data.append({
                                "question": rag["question"],
                                "question_type": rag.get("question_type", "Unknown"),
                                "metric": "Citation Score",
                                "rag_value": rag.get("citation_score", 0.0),
                                "baseline_value": baseline.get("citation_score", 0.0)
                            })
                            comp_data.append({
                                "question": rag["question"],
                                "question_type": rag.get("question_type", "Unknown"),
                                "metric": "Hallucination Score",
                                "rag_value": rag.get("hallucination_score", 1.0),
                                "baseline_value": baseline.get("hallucination_score", 1.0)
                            })
                            
                    if comp_data:
                        comp_df = pd.DataFrame(comp_data)
                        
                        # Create grouped bar chart for comparison
                        metrics = ["Semantic Similarity", "Citation Score", "Hallucination Score"]
                        
                        fig4 = make_subplots(rows=1, cols=3, subplot_titles=metrics)
                        
                        for i, metric in enumerate(metrics):
                            metric_df = comp_df[comp_df["metric"] == metric]
                            
                            rag_mean = metric_df["rag_value"].mean()
                            baseline_mean = metric_df["baseline_value"].mean()
                            
                            fig4.add_trace(
                                go.Bar(
                                    x=["RAG", "Baseline"],
                                    y=[rag_mean, baseline_mean],
                                    name=metric,
                                    text=[f"{rag_mean:.3f}", f"{baseline_mean:.3f}"],
                                    textposition="auto"
                                ),
                                row=1, col=i+1
                            )
                        
                        fig4.update_layout(
                            title="RAG vs Baseline Comparison",
                            showlegend=False,
                            height=400
                        )
                        
                        charts["rag_vs_baseline"] = fig4
                except Exception as e:
                    print(f"Error creating comparison charts: {str(e)}")
        
        # 5. Retrieval quality metrics if available
        try:
            retrieval_data = []
            for i, result in enumerate(valid_results):
                if "retrieval_metrics" in result:
                    metrics = result["retrieval_metrics"]
                    question_type = result.get("question_type", "Unknown")
                    
                    for metric_name, value in metrics.items():
                        retrieval_data.append({
                            "question_id": i+1,
                            "question_type": question_type,
                            "metric": metric_name,
                            "value": value
                        })
            
            if retrieval_data:
                retrieval_df = pd.DataFrame(retrieval_data)
                
                fig6 = px.box(
                    retrieval_df, 
                    x="metric", 
                    y="value", 
                    color="question_type",
                    title="Retrieval Quality Metrics",
                    labels={"value": "Score", "metric": "Metric"}
                )
                
                charts["retrieval_metrics"] = fig6
        except Exception as e:
            print(f"Error creating retrieval metrics chart: {str(e)}")
            
        return charts
                
    def generate_html_report(self, rag_results, rag_metrics, 
                           baseline_results=None, baseline_metrics=None,
                           comparison_stats=None, charts=None, output_file="evaluation_report.html"):
        """Generate HTML report from evaluation results"""
        
        # Create direct charts that should work even when regular ones fail
        direct_charts = create_direct_charts(rag_results)
        
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Report</title>
            <meta charset="UTF-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart-container {{ margin-bottom: 30px; text-align: center; height: 500px; }}
                .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
                .improvement-positive {{ color: green; }}
                .improvement-negative {{ color: red; }}
                .dashboard-row {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
                .dashboard-card {{ flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 5px; padding: 15px; }}
                .dashboard-card h3 {{ margin-top: 0; }}
                .significant {{ font-weight: bold; color: green; }}
                .not-significant {{ color: #888; }}
                .error-summary {{ background-color: #fff3f3; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .tab-content {{ border: 1px solid #dee2e6; padding: 15px; margin-top: -1px; }}
            </style>
        </head>
        <body>
            <h1>RAG Evaluation Report</h1>
            <p>Date generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="dashboard-row">
                <div class="dashboard-card">
                    <h3>RAG Performance Summary</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add RAG metrics
        if rag_metrics:
            for metric, value in rag_metrics.items():
                if metric not in ("by_question_type", "retrieval_metrics") and not isinstance(value, dict):
                    if isinstance(value, float):
                        html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
                    else:
                        html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            
            # Add retrieval metrics
            if "retrieval_metrics" in rag_metrics:
                for metric, value in rag_metrics["retrieval_metrics"].items():
                    if isinstance(value, float):
                        html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
                    else:
                        html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html += f"""
                    </table>
                </div>
        """
        
        # Add comparison section if baseline metrics available
        if baseline_metrics and comparison_stats and comparison_stats.get("valid_comparison", False):
            html += f"""
                <div class="dashboard-card">
                    <h3>Comparison with Baseline</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>RAG</th>
                            <th>Baseline</th>
                            <th>Improvement</th>
                            <th>p-value</th>
                        </tr>
            """
            
            for metric, test_results in comparison_stats.get("tests", {}).items():
                if "error" in test_results:
                    continue
                
                rag_value = test_results.get("rag_mean", 0.0)
                baseline_value = test_results.get("baseline_mean", 0.0)
                
                # Calculate improvement
                if metric == "hallucination_score":
                    # For hallucination, lower is better
                    improvement = ((baseline_value - rag_value) / max(0.0001, baseline_value)) * 100
                else:
                    # For others, higher is better
                    improvement = ((rag_value - baseline_value) / max(0.0001, baseline_value)) * 100
                
                # Format p-value and significance indicator
                p_value = test_results.get("p_value", 1.0)
                is_significant = test_results.get("significant", False)
                significance_class = "significant" if is_significant else "not-significant"
                
                # Format improvement with color
                improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                
                html += f"""
                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td>{rag_value:.4f}</td>
                            <td>{baseline_value:.4f}</td>
                            <td class="{improvement_class}">{improvement:.2f}%</td>
                            <td class="{significance_class}">{p_value:.4f}</td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            """
        
        html += """
            </div>
        """
        
        # Add error summary if there are errors
        error_count = rag_metrics.get("errors", 0) if rag_metrics else 0
        total_count = rag_metrics.get("total_questions", 0) if rag_metrics else 0
        
        if error_count > 0 and total_count > 0:
            error_pct = (error_count / total_count) * 100
            html += f"""
                <div class="error-summary">
                    <h3>Error Summary</h3>
                    <p>Total questions: {total_count}</p>
                    <p>Processing errors: {error_count} ({error_pct:.1f}%)</p>
                </div>
            """
        
        # Add metrics by question type
        if rag_metrics and "by_question_type" in rag_metrics and rag_metrics["by_question_type"]:
            html += """
                <h2>Performance by Question Type</h2>
                <table>
                    <tr>
                        <th>Question Type</th>
                        <th>Count</th>
                        <th>Semantic Similarity</th>
                        <th>Citation Score</th>
                        <th>Hallucination Score</th>
                        <th>Errors</th>
                    </tr>
            """
            
            for q_type, type_metrics in rag_metrics["by_question_type"].items():
                html += f"""
                    <tr>
                        <td>{q_type}</td>
                        <td>{type_metrics.get("count", 0)}</td>
                        <td>{type_metrics.get("avg_semantic_similarity", 0.0):.4f}</td>
                        <td>{type_metrics.get("avg_citation_score", 0.0):.4f}</td>
                        <td>{type_metrics.get("avg_hallucination_score", 1.0):.4f}</td>
                        <td>{type_metrics.get("errors", 0)}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        # Add charts if available
        if charts or direct_charts:
            if "hallucination_by_type" in charts:
                html += """
                    <h2>Hallucination by Question Type</h2>
                    <div class="chart-container" id="hallucination_chart"></div>
                """
                
            # Use direct chart for metric correlation
            if "metric_correlation" in direct_charts:
                html += """
                    <h2>Metrics Correlation</h2>
                    <div class="chart-container" id="correlation_chart"></div>
                """
                
            # Use direct chart for processing time
            if "processing_time" in direct_charts:
                html += """
                    <h2>Processing Time Analysis</h2>
                    <div class="chart-container" id="processing_time_chart"></div>
                """
                
            if "retrieval_metrics" in charts:
                html += """
                    <h2>Retrieval Metrics</h2>
                    <div class="chart-container" id="retrieval_metrics_chart"></div>
                """
                
            if "rag_vs_baseline" in charts:
                html += """
                    <h2>RAG vs Baseline Comparison</h2>
                    <div class="chart-container" id="comparison_chart"></div>
                """
        
        # Add methodology section
        html += """
            <h2>Methodology</h2>
            <p>This evaluation compares a Retrieval-Augmented Generation (RAG) system against a baseline model using multiple metrics:</p>
            <ul>
                <li><strong>Semantic Similarity:</strong> Measures how closely the model's responses match the ground truth answers using cosine similarity of embeddings.</li>
                <li><strong>Citation Score:</strong> Evaluates the model's ability to correctly cite relevant legal documents using F1 score.</li>
                <li><strong>Hallucination Score:</strong> A combined metric where lower scores indicate less hallucination (more factual accuracy).</li>
                <li><strong>Retrieval Metrics:</strong> Includes precision, recall, F1, and MRR to evaluate document retrieval quality.</li>
            </ul>
            
            <h3>Hallucination Score Calculation</h3>
            <p>The hallucination score is a weighted combination of semantic similarity and citation accuracy:</p>
            <pre>Hallucination Score = 1.0 - (w₁ × Semantic Similarity + w₂ × Citation Score)</pre>
            <p>where w₁ and w₂ are configurable weights that sum to 1.</p>
        """
        
        # Add JavaScript for charts
        if charts or direct_charts:
            html += """
            <script>
            """
            
            chart_mappings = {
                "hallucination_by_type": "hallucination_chart",
                "metric_correlation": "correlation_chart",
                "processing_time": "processing_time_chart",
                "retrieval_metrics": "retrieval_metrics_chart",
                "rag_vs_baseline": "comparison_chart"
            }
            
            # First try to use direct charts for the problematic ones
            for chart_id, fig in direct_charts.items():
                if chart_id in chart_mappings:
                    div_id = chart_mappings[chart_id]
                    html += f"""
                    var plotlyData = {fig.to_json()};
                    Plotly.newPlot('{div_id}', plotlyData.data, plotlyData.layout);
                    """
            
            # Then use regular charts for the rest
            for chart_id, fig in charts.items():
                # Skip if we already used a direct chart
                if chart_id in direct_charts:
                    continue
                    
                if chart_id in chart_mappings:
                    div_id = chart_mappings[chart_id]
                    html += f"""
                    var plotlyData = {fig.to_json()};
                    Plotly.newPlot('{div_id}', plotlyData.data, plotlyData.layout);
                    """
            
            html += """
            </script>
            """
        
        # Close HTML tags
        html += """
        </body>
        </html>
        """
        
        # Write to file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Evaluation report saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error writing HTML report: {str(e)}")
            return False