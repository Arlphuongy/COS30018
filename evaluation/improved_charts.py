import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
import os
import json
import time

class EnhancedChartGenerator:
    """Generate enhanced charts for RAG evaluation reports"""
    
    def __init__(self, height=600, width=None):
        """
        Initialize chart generator
        
        Args:
            height: Default chart height
            width: Default chart width (None for auto)
        """
        self.default_height = height
        self.default_width = width
        
        # Set default colors
        self.colors = {
            "Factual": "#636EFA",
            "Contextual": "#EF553B",
            "Complex Reasoning": "#00CC96",
            "Edge Case": "#AB63FA",
            "Unknown": "#FFA15A"
        }
    
    def generate_radar_chart(self, data, title="RAG Quality Metrics by Question Type"):
        """
        Generate radar chart for RAG metrics by question type
        
        Args:
            data: Dictionary with metrics by question type
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Prepare data
        question_types = list(data.keys())
        metrics = ["Faithfulness", "Completeness", "Precision", "Source Utilization", "RAG Quality"]
        
        # Create figure
        fig = go.Figure()
        
        for q_type in question_types:
            type_data = data[q_type]
            
            # Extract metric values
            values = [
                type_data.get("avg_faithfulness", 0.0),
                type_data.get("avg_completeness", 0.0),
                type_data.get("avg_precision", 0.0),
                type_data.get("avg_source_utilization", 0.0),
                type_data.get("avg_rag_quality", 0.0)
            ]
            
            # Add trace
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=q_type,
                line_color=self.colors.get(q_type, None)
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=title,
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    def generate_boxplot(self, results, title="Distribution of RAG Quality Metrics"):
        """
        Generate boxplot of RAG metrics
        
        Args:
            results: List of evaluation results
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Prepare data frame
        metrics_data = []
        
        for result in results:
            q_type = result.get("question_type", "Unknown")
            
            # Add faithfulness
            metrics_data.append({
                "Metric": "Faithfulness",
                "Score": result.get("rag_faithfulness", 0.0),
                "Question Type": q_type
            })
            
            # Add completeness
            metrics_data.append({
                "Metric": "Completeness",
                "Score": result.get("rag_completeness", 0.0),
                "Question Type": q_type
            })
            
            # Add precision
            metrics_data.append({
                "Metric": "Precision",
                "Score": result.get("rag_precision", 0.0),
                "Question Type": q_type
            })
            
            # Add source utilization
            metrics_data.append({
                "Metric": "Source_utilization",
                "Score": result.get("rag_source_utilization", 0.0),
                "Question Type": q_type
            })
            
            # Add RAG quality
            metrics_data.append({
                "Metric": "Rag_quality",
                "Score": result.get("rag_quality", 0.0),
                "Question Type": q_type
            })
        
        # Create data frame
        df = pd.DataFrame(metrics_data)
        
        # Create boxplot
        fig = px.box(
            df,
            x="Metric",
            y="Score",
            color="Question Type",
            title=title,
            color_discrete_map=self.colors
        )
        
        # Update layout
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            boxmode='group',
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def generate_hallucination_chart(self, results, title="Hallucination Scores by Question Type"):
        """
        Generate boxplot of hallucination scores by question type
        
        Args:
            results: List of evaluation results
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Prepare data frame
        hall_data = []
        
        for result in results:
            hall_data.append({
                "Question Type": result.get("question_type", "Unknown"),
                "Hallucination Score": result.get("hallucination_score", 0.0)
            })
        
        # Create data frame
        df = pd.DataFrame(hall_data)
        
        # Create boxplot
        fig = px.box(
            df,
            x="Question Type",
            y="Hallucination Score",
            color="Question Type",
            title=title,
            color_discrete_map=self.colors
        )
        
        # Update layout
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=False,
            xaxis_title="Question Type",
            yaxis_title="Hallucination Score",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def generate_correlation_chart(self, results, title="Correlation: Semantic Similarity vs Hallucination"):
        """
        Generate scatter plot to show correlation between metrics
        
        Args:
            results: List of evaluation results
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Prepare data
        corr_data = []
        
        for result in results:
            corr_data.append({
                "Question Type": result.get("question_type", "Unknown"),
                "Semantic Similarity": result.get("semantic_similarity", 0.0),
                "Hallucination Score": result.get("hallucination_score", 0.0)
            })
        
        # Create data frame
        df = pd.DataFrame(corr_data)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x="Semantic Similarity",
            y="Hallucination Score",
            color="Question Type",
            title=title,
            color_discrete_map=self.colors,
            opacity=0.7
        )
        
        # Add trend line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[1, 0],
                mode='lines',
                line=dict(color='rgba(0,0,0,0.3)', dash='dash'),
                name='Expected Trend',
                showlegend=False
            )
        )
        
        # Update layout
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def generate_processing_time_chart(self, results, title="Distribution of Processing Times"):
        """
        Generate histogram of processing times
        
        Args:
            results: List of evaluation results
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Prepare data
        time_data = []
        
        for result in results:
            time_data.append({
                "Question Type": result.get("question_type", "Unknown"),
                "Processing Time": result.get("processing_time", 0.0)
            })
        
        # Create data frame
        df = pd.DataFrame(time_data)
        
        # Create histogram
        fig = px.histogram(
            df,
            x="Processing Time",
            color="Question Type",
            title=title,
            color_discrete_map=self.colors,
            opacity=0.7,
            barmode='overlay',
            histnorm='percent',
            nbins=20
        )
        
        # Update layout
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            xaxis_title="Processing Time (seconds)",
            yaxis_title="Percent"
        )
        
        return fig
    
    def generate_comparison_chart(self, rag_results, baseline_results, title="RAG vs Baseline Comparison"):
        """
        Generate comparison chart between RAG and baseline
        
        Args:
            rag_results: List of RAG evaluation results
            baseline_results: List of baseline evaluation results
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if not rag_results or not baseline_results:
            return None
            
        # Match results by question
        rag_dict = {r.get("question", ""): r for r in rag_results}
        baseline_dict = {r.get("question", ""): r for r in baseline_results}
        
        common_questions = set(rag_dict.keys()) & set(baseline_dict.keys())
        
        if not common_questions:
            return None
            
        # Prepare metrics for comparison
        rag_metrics = {
            "Semantic Similarity": [],
            "Citation Score": [],
            "Hallucination Score": []
        }
        
        baseline_metrics = {
            "Semantic Similarity": [],
            "Citation Score": [],
            "Hallucination Score": []
        }
        
        for question in common_questions:
            rag = rag_dict[question]
            baseline = baseline_dict[question]
            
            rag_metrics["Semantic Similarity"].append(rag.get("semantic_similarity", 0.0))
            rag_metrics["Citation Score"].append(rag.get("citation_score", 0.0))
            rag_metrics["Hallucination Score"].append(rag.get("hallucination_score", 0.0))
            
            baseline_metrics["Semantic Similarity"].append(baseline.get("semantic_similarity", 0.0))
            baseline_metrics["Citation Score"].append(baseline.get("citation_score", 0.0))
            baseline_metrics["Hallucination Score"].append(baseline.get("hallucination_score", 0.0))
            
        # Calculate averages
        rag_averages = {k: np.mean(v) for k, v in rag_metrics.items()}
        baseline_averages = {k: np.mean(v) for k, v in baseline_metrics.items()}
        
        # Create subplot
        fig = make_subplots(
            rows=1, 
            cols=3,
            subplot_titles=list(rag_metrics.keys())
        )
        
        # Add bars for each metric
        for i, metric in enumerate(rag_metrics.keys(), 1):
            rag_val = rag_averages[metric]
            baseline_val = baseline_averages[metric]
            
            # Text needs to show positive improvements
            if metric == "Hallucination Score":
                better = "lower" if rag_val < baseline_val else "higher"
                pct_change = abs((rag_val - baseline_val) / max(0.001, baseline_val)) * 100
                text = f"{rag_val:.3f} ({pct_change:.1f}% {better})"
            else:
                better = "higher" if rag_val > baseline_val else "lower"
                pct_change = abs((rag_val - baseline_val) / max(0.001, baseline_val)) * 100
                text = f"{rag_val:.3f} ({pct_change:.1f}% {better})"
            
            fig.add_trace(
                go.Bar(
                    x=["RAG"],
                    y=[rag_val],
                    text=[f"{rag_val:.3f}"],
                    textposition="auto",
                    name=f"RAG {metric}",
                    marker_color="#1f77b4",
                    showlegend=False
                ),
                row=1, col=i
            )
            
            fig.add_trace(
                go.Bar(
                    x=["Baseline"],
                    y=[baseline_val],
                    text=[f"{baseline_val:.3f}"],
                    textposition="auto",
                    name=f"Baseline {metric}",
                    marker_color="#ff7f0e",
                    showlegend=False
                ),
                row=1, col=i
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=self.default_height,
            width=self.default_width
        )
        
        return fig

class EnhancedReportGenerator:
    """Enhanced report generator with better chart support"""
    
    def __init__(self, chart_height=500):
        """Initialize report generator"""
        self.chart_generator = EnhancedChartGenerator(height=chart_height)
        
    def generate_html_report(self, rag_results, rag_metrics, 
                           baseline_results=None, baseline_metrics=None,
                           comparison_stats=None, output_file="evaluation_report.html"):
        """
        Generate HTML report with evaluation results and charts
        
        Args:
            rag_results: List of RAG evaluation results
            rag_metrics: Dictionary with RAG metrics
            baseline_results: List of baseline evaluation results
            baseline_metrics: Dictionary with baseline metrics
            comparison_stats: Dictionary with comparison statistics
            output_file: Path to output HTML file
        """
        # Generate charts
        charts = self._generate_charts(rag_results, baseline_results)
        
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Report</title>
            <meta charset="UTF-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart-container {{ margin-bottom: 40px; text-align: center; height: 550px; width: 100%; }}
                .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
                .improvement-positive {{ color: green; font-weight: bold; }}
                .improvement-negative {{ color: red; font-weight: bold; }}
                .dashboard-row {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
                .dashboard-card {{ flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .dashboard-card h3 {{ margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                .significant {{ font-weight: bold; color: green; }}
                .not-significant {{ color: #888; }}
                .error-summary {{ background-color: #fff3f3; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); grid-gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; background-color: #f9f9f9; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 28px; font-weight: bold; margin: 15px 0; color: #2c3e50; }}
                .metric-title {{ font-size: 16px; color: #7f8c8d; margin-bottom: 5px; text-transform: uppercase; }}
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
            # Add overall metrics
            overall = rag_metrics.get("overall", {})
            
            metrics_to_show = [
                ("count", "Total Questions"),
                ("errors", "Errors"),
                ("avg_semantic_similarity", "Avg Semantic Similarity"),
                ("avg_citation_score", "Avg Citation Score"),
                ("avg_hallucination_score", "Avg Hallucination Score"),
                ("avg_faithfulness", "Avg Faithfulness"),
                ("avg_completeness", "Avg Completeness"),
                ("avg_precision", "Avg Precision"),
                ("avg_source_utilization", "Avg Source Utilization"),
                ("avg_rag_quality", "Avg RAG Quality")
            ]
            
            for key, display_name in metrics_to_show:
                value = overall.get(key, 0)
                if isinstance(value, float):
                    html += f"<tr><td>{display_name}</td><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td>{display_name}</td><td>{value}</td></tr>"
        
        html += f"""
                    </table>
                </div>
        """
        
        # Add comparison with baseline - Enhanced version with clear improvement metrics
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
                            <th>% Change</th>
                            <th>p-value</th>
                        </tr>
            """
            
            for metric, test_results in comparison_stats.get("tests", {}).items():
                if "error" in test_results:
                    continue
                
                rag_value = test_results.get("rag_mean", 0.0)
                baseline_value = test_results.get("baseline_mean", 0.0)
                display_name = test_results.get("display_name", metric.replace("_", " ").title())
                
                # Calculate improvement percentage
                higher_better = test_results.get("higher_better", True)
                
                if higher_better:
                    # For metrics where higher is better (similarity, citation score)
                    improvement = rag_value - baseline_value
                    improvement_pct = ((rag_value - baseline_value) / max(0.0001, baseline_value)) * 100
                    direction = "better" if improvement > 0 else "worse"
                else:
                    # For metrics where lower is better (hallucination score)
                    improvement = baseline_value - rag_value
                    improvement_pct = ((baseline_value - rag_value) / max(0.0001, baseline_value)) * 100
                    direction = "better" if improvement > 0 else "worse"
                
                # Format p-value and significance indicator
                p_value = test_results.get("p_value", 1.0)
                is_significant = test_results.get("significant", False)
                significance_class = "significant" if is_significant else "not-significant"
                
                # Format improvement with color and direction
                improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                
                # Show absolute improvement and percentage
                improvement_text = f"{improvement:.4f}"
                
                html += f"""
                        <tr>
                            <td>{display_name}</td>
                            <td>{rag_value:.4f}</td>
                            <td>{baseline_value:.4f}</td>
                            <td class="{improvement_class}">{improvement_text}</td>
                            <td class="{improvement_class}">{improvement_pct:.2f}% {direction}</td>
                            <td class="{significance_class}">{p_value:.4f} {'(significant)' if is_significant else ''}</td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            """
        
        html += """
            </div>
        """
        
        # Add RAG Quality Dashboard
        overall = rag_metrics.get("overall", {}) if rag_metrics else {}
        html += """
            <h2>RAG Quality Dashboard</h2>
            <div class="metrics-grid">
        """
        
        # Key metrics to show in dashboard
        key_metrics = [
            ("avg_faithfulness", "Faithfulness"),
            ("avg_completeness", "Completeness"),
            ("avg_precision", "Precision"),
            ("avg_source_utilization", "Source Utilization"),
            ("avg_rag_quality", "Quality")
        ]
        
        for key, display_name in key_metrics:
            value = overall.get(key, 0.0)
            html += f"""
                <div class="metric-card">
                    <div class="metric-title">{display_name}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
            """
        
        html += """
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
                        <th>RAG Quality</th>
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
                        <td>{type_metrics.get("avg_rag_quality", 0.0):.4f}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        # Add charts
        if charts:
            # RAG Quality by Question Type
            if "rag_quality_radar" in charts:
                html += """
                    <h2>RAG Quality by Question Type</h2>
                    <div class="chart-container" id="rag_quality_radar"></div>
                """
            
            # Distribution of RAG Quality Metrics
            if "rag_metrics_boxplot" in charts:
                html += """
                    <h2>Distribution of RAG Quality Metrics</h2>
                    <div class="chart-container" id="rag_metrics_boxplot"></div>
                """
            
            # Hallucination by Question Type
            if "hallucination_chart" in charts:
                html += """
                    <h2>Hallucination by Question Type</h2>
                    <div class="chart-container" id="hallucination_chart"></div>
                """
            
            # Metrics Correlation
            if "correlation_chart" in charts:
                html += """
                    <h2>Metrics Correlation</h2>
                    <div class="chart-container" id="correlation_chart"></div>
                """
            
            # RAG vs Baseline Comparison
            if "comparison_chart" in charts:
                html += """
                    <h2>RAG vs Baseline Comparison</h2>
                    <div class="chart-container" id="comparison_chart"></div>
                """
            
            # Add JavaScript for charts
            html += """
            <script>
            """
            
            # Add each chart JSON
            for chart_id, chart_json in charts.items():
                html += f"""
                var {chart_id}_data = {chart_json};
                Plotly.newPlot('{chart_id}', {chart_id}_data.data, {chart_id}_data.layout, {{responsive: true}});
                """
            
            html += """
            </script>
            """
        
        # Add methodology section
        html += """
            <h2>Methodology</h2>
            <p>This evaluation compares a Retrieval-Augmented Generation (RAG) system against a baseline model using multiple metrics:</p>
            <ul>
                <li><strong>Semantic Similarity:</strong> Measures how closely the model's responses match the ground truth answers using cosine similarity of embeddings.</li>
                <li><strong>Citation Score:</strong> Evaluates the model's ability to correctly cite relevant Vietnamese legal documents (including Thông tư/Circulars, Nghị định/Decrees, and Điều/Articles).</li>
                <li><strong>Hallucination Score:</strong> A combined metric where lower scores indicate less hallucination (more factual accuracy).</li>
                <li><strong>RAG-Specific Metrics:</strong>
                    <ul>
                        <li><strong>Faithfulness:</strong> Measures how well the generated content aligns with the retrieved documents.</li>
                        <li><strong>Completeness:</strong> Evaluates how comprehensively the response covers the ground truth information.</li>
                        <li><strong>Precision:</strong> Measures the lexical precision of the response compared to ground truth.</li>
                        <li><strong>Source Utilization:</strong> Evaluates how well the system used available relevant documents.</li>
                        <li><strong>RAG Quality:</strong> A weighted composite score combining the above metrics.</li>
                    </ul>
                </li>
            </ul>
            
            <h3>Citation Detection</h3>
            <p>The system detects various citation formats for Vietnamese legal documents:</p>
            <ul>
                <li><strong>Thông tư (Circulars):</strong> Examples: "Thông tư 67", "TT67", "Thông tư 67/2023/TT-BTC"</li>
                <li><strong>Nghị định (Decrees):</strong> Examples: "Nghị định 46", "ND46", "Nghị định 46/2023/NĐ-CP"</li>
                <li><strong>Điều (Articles):</strong> Examples: "Điều 41", references to specific articles</li>
            </ul>
            
            <h3>Document Matching</h3>
            <p>The document matching algorithm uses both exact and partial matching:</p>
            <ul>
                <li>Exact matches occur when document IDs are identical after normalization.</li>
                <li>Partial matches occur when document types or numbers match but not both.</li>
                <li>Document IDs are normalized to handle different formats and conventions.</li>
            </ul>
            
            <h3>Hallucination Score Calculation</h3>
            <p>The hallucination score is a weighted combination of semantic similarity and citation accuracy:</p>
            <pre>Hallucination Score = 1.0 - (0.7 × Semantic Similarity + 0.3 × Citation Score)</pre>
            <p>Lower hallucination scores indicate more factual responses.</p>
        </body>
        </html>
        """
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write HTML content to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Successfully saved HTML report to {output_file}")
    
    def _generate_charts(self, rag_results, baseline_results=None):
        """
        Generate charts for report
        
        Args:
            rag_results: List of RAG evaluation results
            baseline_results: List of baseline evaluation results
            
        Returns:
            dict: Chart data in JSON format
        """
        if not rag_results:
            return {}
            
        # Filter out error results
        valid_results = [r for r in rag_results if "ERROR:" not in r.get("rag_answer", "")]
        
        # Generate chart data
        charts = {}
        
        # Helper function to add chart
        def add_chart(chart_id, figure):
            try:
                charts[chart_id] = figure.to_json()
            except Exception as e:
                print(f"Error converting chart {chart_id} to JSON: {e}")
        
        # Generate radar chart
        try:
            # Group results by question type
            by_type = {}
            for result in valid_results:
                q_type = result.get("question_type", "Unknown")
                if q_type not in by_type:
                    by_type[q_type] = []
                by_type[q_type].append(result)
            
            # Calculate metrics by type
            type_metrics = {}
            for q_type, type_results in by_type.items():
                type_metrics[q_type] = {
                    "avg_faithfulness": np.mean([r.get("rag_faithfulness", 0.0) for r in type_results]),
                    "avg_completeness": np.mean([r.get("rag_completeness", 0.0) for r in type_results]),
                    "avg_precision": np.mean([r.get("rag_precision", 0.0) for r in type_results]),
                    "avg_source_utilization": np.mean([r.get("rag_source_utilization", 0.0) for r in type_results]),
                    "avg_rag_quality": np.mean([r.get("rag_quality", 0.0) for r in type_results])
                }
            
            # Generate radar chart
            radar_chart = self.chart_generator.generate_radar_chart(type_metrics)
            add_chart("rag_quality_radar", radar_chart)
        except Exception as e:
            print(f"Error generating radar chart: {e}")
        
        # Generate boxplot
        try:
            boxplot = self.chart_generator.generate_boxplot(valid_results)
            add_chart("rag_metrics_boxplot", boxplot)
        except Exception as e:
            print(f"Error generating boxplot: {e}")
        
        # Generate hallucination chart
        try:
            hall_chart = self.chart_generator.generate_hallucination_chart(valid_results)
            add_chart("hallucination_chart", hall_chart)
        except Exception as e:
            print(f"Error generating hallucination chart: {e}")
        
        # Generate correlation chart
        try:
            corr_chart = self.chart_generator.generate_correlation_chart(valid_results)
            add_chart("correlation_chart", corr_chart)
        except Exception as e:
            print(f"Error generating correlation chart: {e}")
        
        # Generate comparison chart
        if baseline_results:
            try:
                valid_baseline = [r for r in baseline_results if "ERROR:" not in r.get("baseline_answer", "")]
                comp_chart = self.chart_generator.generate_comparison_chart(valid_results, valid_baseline)
                if comp_chart:
                    add_chart("comparison_chart", comp_chart)
            except Exception as e:
                print(f"Error generating comparison chart: {e}")
        
        return charts