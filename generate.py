import os
import time
import dotenv
import gradio as gr
import google.generativeai as genai

#load environment variables
dotenv.load_dotenv(".env.local")

#initialize gemini api
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

#set the model to use - we know gemini-1.5-flash works
MODEL_NAME = "models/gemini-1.5-flash"
print(f"Using model: {MODEL_NAME}")

#import the retrieval module
from retrieval import search_legal_documents


#format the retrieved documents into a readable string for the llm
def format_documents(results):
    if not results:
        return "No documents found."
    
    formatted_docs = ""
    for i, result in enumerate(results):
        metadata = result.payload.get('metadata', {})
        text = result.payload.get('text', '')
        thong_tu = metadata.get('Thông tư', 'Unknown')
        dieu = metadata.get('Điều', 'Unknown')
        score = result.score  # Get the relevance score
        
        formatted_docs += f"Tài liệu #{i+1} (Độ tin cậy: {score:.4f}):\n"
        formatted_docs += f"Nguồn: Thông tư {thong_tu}, Điều {dieu}\n"
        formatted_docs += f"Nội dung: {text}\n\n"
    
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
        metadata = doc.payload.get('metadata', {})
        thong_tu = metadata.get('Thông tư', 'Unknown')
        dieu = metadata.get('Điều', 'Unknown')
        
        #extract a short preview of the document
        text = doc.payload.get('text', '')
        excerpt = text[:200] + "..." if len(text) > 200 else text
        
        #format the source information
        sources_text += f"**Nguồn #{i+1}:** Thông tư {thong_tu}, Điều {dieu} (Độ tin cậy: {doc.score:.4f})\n\n"
        sources_text += f"*Trích đoạn:* {excerpt}\n\n---\n\n"
    
    return sources_text


#process user query and return answer with its sources
def process_query(query, num_docs = 5):
    if not query.strip():
        return "Vui lòng nhập câu hỏi của bạn.", ""
    
    try:
        #track processing time
        start_time = time.time()
        
        #step 1, retrieve documents
        print(f"Retrieving documents for query: {query}")
        retrieved_docs = search_legal_documents(query, limit=num_docs)
        
        if not retrieved_docs:
            return "Không tìm thấy tài liệu liên quan đến câu hỏi của bạn.", ""
        
        #step 2, generate answer
        print(f"Generating response with {len(retrieved_docs)} documents")
        answer = generate_response(query, retrieved_docs)
        
        #step 3, format sources for display
        sources_text = format_sources_for_display(retrieved_docs, time.time() - start_time)
        
        return answer, sources_text
    
    except Exception as e:
        error_message = f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
        print(error_message)
        return error_message, ""

#create and configure the gradio interface
def create_interface():
    with gr.Blocks(title="Trợ lý Pháp lý Việt Nam", theme=gr.themes.Soft()) as demo:
        #header
        gr.Markdown(
            f"""
            # 🇻🇳 Trợ lý Pháp lý
            
            Hệ thống này sử dụng công nghệ RAG (Retrieval-Augmented Generation) để trả lời 
            các câu hỏi liên quan đến tài liệu pháp lý.
            
            *Model: {MODEL_NAME}*
            """
        )
        
        #input area
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Câu hỏi của bạn", 
                    placeholder="Ví dụ: Điều 4 trong thông tư 67 quy định về gì?",
                    lines=2
                )
            
            with gr.Column(scale=1):
                num_docs = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    value=5, 
                    step=1, 
                    label="Số lượng tài liệu"
                )
        
        #submit button
        submit_btn = gr.Button("Gửi", variant="primary")
        
        #output area
        with gr.Row():
            with gr.Column(scale=3):
                answer_output = gr.Markdown(label="Câu trả lời")
            with gr.Column(scale=2):
                sources_output = gr.Markdown(label="Nguồn tài liệu")
        
        #event handlers
        submit_btn.click(
            fn=process_query, 
            inputs=[query_input, num_docs], 
            outputs=[answer_output, sources_output]
        )
        query_input.submit(
            fn=process_query, 
            inputs=[query_input, num_docs], 
            outputs=[answer_output, sources_output]
        )
        
        #example questions
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