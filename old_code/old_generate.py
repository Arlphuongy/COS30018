import os
import time
import dotenv
import gradio as gr
import google.generativeai as genai
import re

#load environment variables
dotenv.load_dotenv(".env.local")

#initialize gemini api
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

#set the model to use - we know gemini-1.5-flash works
MODEL_NAME = os.getenv('MODEL_NAME')
print(f"Using model: {MODEL_NAME}")

#import the retrieval module
from old_retrieval import search_legal_documents

#format the retrieved documents into a readable string for the llm
# Updated format_documents function in generate.py
def format_documents(results):
    if not results:
        return "No documents found."
    
    formatted_docs = ""
    for i, result in enumerate(results):
        payload = result.payload
        content = payload.get('content', '')
        document_name = payload.get('document_name', 'Unknown')
        reference_id = payload.get('reference_id', 'Unknown')
        section_title = payload.get('section_title', '')
        
        # Extract Điều number with improved pattern matching
        article_headline = payload.get('article_headline', '')
        dieu_match = re.search(r'điều\s+(\d+)', article_headline.lower())
        if not dieu_match:
            dieu_match = re.search(r'điều\s+(\d+)', section_title.lower())
        if not dieu_match and reference_id:
            dieu_match = re.search(r'[dđ]\s*(\d+)', reference_id.lower())
            
        dieu_num = dieu_match.group(1) if dieu_match else None
        
        score = result.score
        
        formatted_docs += f"Tài liệu #{i+1} (Độ tin cậy: {score:.4f}):\n"
        
        # Format source info
        source_info = f"Nguồn: {document_name}"
        if dieu_num:
            source_info += f", Điều {dieu_num}"
        if reference_id != 'Unknown':
            source_info += f", Ref: {reference_id}"
        formatted_docs += source_info + "\n"
        
        # Add section title if available
        if section_title:
            formatted_docs += f"Section: {section_title}\n"
            
        formatted_docs += f"Tiêu đề: {article_headline}\n"
        formatted_docs += f"Nội dung: {content}\n\n"
    
    return formatted_docs

#create full prompt for the model
def create_prompt(query, documents):
    system_instructions = """
    Bạn là trợ lý pháp lý chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là giúp người dùng hiểu rõ các quy định pháp luật 
    dựa trên các tài liệu pháp lý được cung cấp. Hãy trả lời câu hỏi của người dùng dựa trên thông tin trong các tài liệu.
    
    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ các tài liệu được cung cấp trong phần ngữ cảnh
    2. Nếu thông tin không đủ để trả lời, hãy nêu rõ và không đưa ra phán đoán
    3. Trích dẫn cụ thể các điều khoản liên quan để hỗ trợ câu trả lời
    4. Trả lời bằng tiếng Việt với ngôn ngữ dễ hiểu, tránh thuật ngữ phức tạp khi có thể
    5. Không tạo ra thông tin không có trong tài liệu, không đưa ra tư vấn pháp lý cá nhân
    6. Luôn trích dẫn nguồn (Thông tư số mấy, Điều mấy) khi đưa ra thông tin
    7. Ưu tiên thông tin từ các tài liệu có độ tin cậy (relevance score) cao hơn khi có xung đột hoặc mâu thuẫn
    8. Khi trả lời, ưu tiên thông tin từ tài liệu có điểm số cao nhất trước
    """
    
    full_prompt = f"{system_instructions}\n\nCâu hỏi: {query}\n\nNgữ cảnh từ các tài liệu pháp lý (đã được sắp xếp theo độ tin cậy):\n{documents}\n\nDựa vào ngữ cảnh trên, hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ."
    
    return full_prompt


def generate_response(query, documents):
    try:
        #format documents and create the prompt
        formatted_docs = format_documents(documents)
        full_prompt = create_prompt(query, formatted_docs)
        
        #initialize the model
        model = genai.GenerativeModel(MODEL_NAME)
        
        #generate the response
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                    top_p=0.95,
                )
            )
            return response.text
        except Exception as config_error:
            print(f"Error with generation config, trying simpler call: {config_error}")
            response = model.generate_content(full_prompt)
            return response.text
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"Đã xảy ra lỗi khi tạo phản hồi: {str(e)}"


def format_sources_for_display(documents, processing_time):
    sources_text = f"### Nguồn tài liệu tham khảo (Thời gian xử lý: {processing_time:.2f}s)\n\n"
    
    for i, doc in enumerate(documents):
        payload = doc.payload
        document_name = payload.get('document_name', 'Unknown')
        reference_id = payload.get('reference_id', 'Unknown')
        section_title = payload.get('section_title', '')
        
        # Extract Điều number with improved pattern matching
        article_headline = payload.get('article_headline', '')
        
        # Try multiple sources to find the article number
        dieu_match = re.search(r'điều\s+(\d+)', article_headline.lower())
        if not dieu_match:
            dieu_match = re.search(r'điều\s+(\d+)', section_title.lower())
        if not dieu_match and reference_id:
            dieu_match = re.search(r'[dđ]\s*(\d+)', reference_id.lower())
            
        dieu_num = dieu_match.group(1) if dieu_match else None
        
        # Extract a short preview of the document
        content = payload.get('content', '')
        excerpt = content[:200] + "..." if len(content) > 200 else content
        
        # Format the source information with document name
        source_header = f"**Nguồn #{i+1}:** {document_name}"
        
        # Only add Điều number if known
        if dieu_num:
            source_header += f", Điều {dieu_num}"
        
        # Add score
        source_header += f" (Score: {doc.score:.4f})"
        
        sources_text += source_header + "\n\n"
        
        # Add Reference ID if available
        if reference_id != 'Unknown':
            sources_text += f"**Reference ID:** {reference_id}\n\n"
        
        # Add section title if available
        if section_title:
            sources_text += f"**Section:** {section_title}\n\n"
        
        sources_text += f"*Trích đoạn:* {excerpt}\n\n---\n\n"
    
    return sources_text

#process user query and return answer with its sources
def process_query(query, num_docs = 5):
    if not query.strip():
        return "Please enter your question.", ""
    
    try:
        #track processing time
        start_time = time.time()
        
        #step 1, retrieve documents
        print(f"Retrieving documents for query: {query}")
        retrieved_docs = search_legal_documents(query, limit=num_docs)

        debug_document_metadata(retrieved_docs)
        
        if not retrieved_docs:
            return "Không tìm thấy tài liệu liên quan đến câu hỏi của bạn.", ""
        
        #step 2, generate answer
        print(f"Generating response with {len(retrieved_docs)} documents")
        answer = generate_response(query, retrieved_docs)
        
        #step 3, format sources for display
        sources_text = format_sources_for_display(retrieved_docs, time.time() - start_time)
        
        return answer, sources_text
    
    except Exception as e:
        error_message = f"An error occurred while processing the query: {str(e)}"
        print(error_message)
        return error_message, ""
    
# Add this function to generate.py
def debug_document_metadata(documents):
    """Print all available metadata keys and values for the first document"""
    if not documents:
        print("No documents to debug")
        return
    
    print("\n=== DEBUG: Document Metadata ===")
    payload = documents[0].payload
    for key, value in payload.items():
        print(f"Key: {key}, Value type: {type(value).__name__}, Preview: {str(value)[:50]}")
    print("=== END DEBUG ===\n")


def create_interface():
    with gr.Blocks(title="Vietnamese Legal Assistant", theme=gr.themes.Soft()) as demo:
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
        
        # Function for Baseline Model Mode - no retrieval, just model
        def process_baseline_model(query):
            """Process query with just the model, no retrieval or sources"""
            try:
                # Create a simple prompt with only the query
                system_instructions = """
                Bạn là trợ lý pháp lý chuyên về luật pháp Việt Nam. Hãy trả lời câu hỏi của người dùng 
                dựa trên kiến thức chung về pháp luật Việt Nam.
                """
                
                model_only_prompt = f"{system_instructions}\n\nCâu hỏi: {query}\n\nHãy trả lời câu hỏi này một cách chính xác và đầy đủ."
                
                # Initialize model and generate response
                model = genai.GenerativeModel(MODEL_NAME)
                
                try:
                    response = model.generate_content(
                        model_only_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=1024,
                            top_p=0.95,
                        )
                    )
                    return response.text
                except Exception as config_error:
                    print(f"Error with generation config, trying simpler call: {config_error}")
                    response = model.generate_content(model_only_prompt)
                    return response.text
                
            except Exception as e:
                error_message = f"An error occurred while processing the query: {str(e)}"
                print(error_message)
                return error_message
        
        # Function for Retrieval Only mode
        def process_retrieval_only(query, num_docs):
            """Process query through retrieval only, no model generation"""
            if not query.strip():
                return "Please enter your question.", ""
            
            try:
                # Track processing time
                start_time = time.time()
                
                # Retrieve documents
                print(f"Retrieving documents for query: {query}")
                retrieved_docs = search_legal_documents(query, limit=num_docs)
                
                if not retrieved_docs:
                    return "Không tìm thấy tài liệu liên quan đến câu hỏi của bạn.", ""
                
                # Format the documents that would have been sent to the model
                formatted_docs = format_documents(retrieved_docs)
                
                # Format sources for display
                sources_text = format_sources_for_display(retrieved_docs, time.time() - start_time)
                
                # Return both the formatted documents and sources
                return f"### Retrieved Documents\n\n{formatted_docs}", sources_text
            
            except Exception as e:
                error_message = f"An error occurred while processing the query: {str(e)}"
                print(error_message)
                return error_message, ""
        
        # Function to route between different testing modes
        def process_with_selected_mode(query, num_docs, mode):
            if mode == "Normal RAG Mode":
                return process_query(query, num_docs)
            elif mode == "Retrieval Only Mode":
                return process_retrieval_only(query, num_docs)
            elif mode == "Baseline Model Mode":
                # Baseline mode just uses the model - no sources
                return process_baseline_model(query), ""
            else:
                return "Invalid mode selected", ""
        
        # Event handlers
        submit_btn.click(
            fn=process_with_selected_mode, 
            inputs=[query_input, num_docs, test_mode], 
            outputs=[answer_output, sources_output]
        )
        query_input.submit(
            fn=process_with_selected_mode, 
            inputs=[query_input, num_docs, test_mode], 
            outputs=[answer_output, sources_output]
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
    demo.launch(share=True)  #set share=True to create a public link