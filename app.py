import os
import dotenv
import gradio as gr
import time
from retrieval_utils import preprocess_text

# Load environment variables
dotenv.load_dotenv(".env.local")

# Import processing functions from generate.py
from generate import (
    process_query,
    process_baseline_model,
    process_retrieval_only
)

# Get model name for display
MODEL_NAME = os.getenv('MODEL_NAME')

def create_interface():
    """Creates and returns the Gradio web interface"""
    with gr.Blocks(title="Vietnamese Legal Assistant", theme=gr.themes.Soft(), css="""
        #status-indicator {
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        """) as demo:
        # Header
        gr.Markdown(
            f"""
            # 🇻🇳 Trợ lý Pháp lý 
            
            Different Testing Options:
            
            * **RAG Mode** involves retrieving relevant documents from a knowledge base and then using those documents to enhance the model's context and generation process
            * **Retrieval Only Mode** focuses solely on finding and ranking relevant documents without actually integrating them into the generation process
            * **Baseline Model Mode** represents standard language model generation without any external knowledge retrieval, relying purely on the model's pre-trained parameters
            
            *Model: {MODEL_NAME}*
            """
        )
        
        # Input area
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Question", 
                    placeholder="Example: Điều 4 trong thông tư 67 quy định về gì?",
                    lines=2
                )
            
            with gr.Column(scale=1):
                num_docs = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    value=5, 
                    step=1, 
                    label="Max number of top relevant documents to be retrieved"
                )
        
        # Test mode selection
        test_mode = gr.Radio(
            ["Normal RAG Mode", "Retrieval Only Mode", "Baseline Model Mode"],
            label="Testing Mode",
            value="Normal RAG Mode"
        )
        
        # Submit button
        submit_btn = gr.Button("Send", variant="primary")
        
        # Output area
        with gr.Row():
            with gr.Column(scale=3):
                answer_output = gr.Markdown(label="Answer")
            with gr.Column(scale=2):
                sources_output = gr.Markdown(label="Sources")
        
        # Loading indicator
        with gr.Row():
            status_indicator = gr.Markdown("Ready", elem_id="status-indicator")
        
        # Function to route between different testing modes
        def process_with_selected_mode(query, num_docs, mode):
            """Routes the query processing based on the selected mode"""
            try:
                if mode == "Normal RAG Mode":
                    answer, sources = process_query(query, num_docs)
                    return answer, sources, "Ready"
                elif mode == "Retrieval Only Mode":
                    answer, sources = process_retrieval_only(query, num_docs)
                    return answer, sources, "Ready"
                elif mode == "Baseline Model Mode":
                    # Baseline mode just uses the model - no sources
                    answer = process_baseline_model(query)
                    return answer, "", "Ready"
                else:
                    return "Invalid mode selected", "", "Ready"
            except Exception as e:
                return f"Error: {str(e)}", "", "Ready"
        
        # Event handlers with loading states
        submit_btn.click(
            fn=lambda: ("", "", "Processing..."),  # Set loading state
            inputs=None,
            outputs=[answer_output, sources_output, status_indicator],
            queue=False
        ).then(
            fn=process_with_selected_mode, 
            inputs=[query_input, num_docs, test_mode], 
            outputs=[answer_output, sources_output, status_indicator]
        )
        
        query_input.submit(
            fn=lambda: ("", "", "Processing..."),  # Set loading state
            inputs=None,
            outputs=[answer_output, sources_output, status_indicator],
            queue=False
        ).then(
            fn=process_with_selected_mode, 
            inputs=[query_input, num_docs, test_mode], 
            outputs=[answer_output, sources_output, status_indicator]
        )
        
        # Example questions
        gr.Examples(
            examples=[
                ["Phí bảo hiểm xe cơ giới là gì?"],
                ["Điều 4 trong thông tư 67 quy định về gì?"],
                ["Điều 2 và Điều 3 trong thông tư 67 bao gồm những gì?"]
            ],
            inputs=query_input
        )
        
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)  # set share=True to create a public link