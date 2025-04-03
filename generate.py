import os
import time
import google.generativeai as genai
import re
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv(".env.local")

# Initialize gemini api
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Set the model to use
MODEL_NAME = os.getenv('MODEL_NAME')
print(f"Using model: {MODEL_NAME}")

# Import the retrieval module
from retrieval import search_legal_documents


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
        
        # Extract Điều number with improved pattern matching and None checking
        article_headline = payload.get('article_headline', '')
        dieu_match = None
        
        # Safely check article_headline
        if article_headline and isinstance(article_headline, str):
            dieu_match = re.search(r'điều\s+(\d+)', article_headline.lower())
        
        # Safely check section_title if no match found
        if not dieu_match and section_title and isinstance(section_title, str):
            dieu_match = re.search(r'điều\s+(\d+)', section_title.lower())
        
        # Safely check reference_id if still no match
        if not dieu_match and reference_id and isinstance(reference_id, str):
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

def create_prompt(query, documents):
    """
    Creates a full prompt for the model with system instructions
    
    Args:
        query (str): User query
        documents (str): Formatted document text
        
    Returns:
        str: Complete prompt for the model
    """
    system_instructions = """
    Bạn là trợ lý pháp lý chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là giúp người dùng hiểu rõ các quy định pháp luật 
    dựa trên các tài liệu pháp lý được cung cấp. Hãy trả lời câu hỏi của người dùng dựa trên thông tin trong các tài liệu.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ các tài liệu được cung cấp trong phần ngữ cảnh
    2. Nếu thông tin không đủ để trả lời hoặc tài liệu không trực tiếp liên quan đến câu hỏi, hãy nêu rõ rằng 
    "Tôi không tìm thấy thông tin cụ thể về [chủ đề câu hỏi] trong các tài liệu được cung cấp" 
    và KHÔNG trích dẫn bất kỳ tài liệu nào
    3. Trích dẫn cụ thể các điều khoản liên quan để hỗ trợ câu trả lời
    4. Trả lời bằng tiếng Việt với ngôn ngữ dễ hiểu, tránh thuật ngữ phức tạp khi có thể
    5. Không tạo ra thông tin không có trong tài liệu, không đưa ra tư vấn pháp lý cá nhân
    6. Luôn trích dẫn nguồn (Thông tư số mấy, Điều mấy) khi đưa ra thông tin
    7. Ưu tiên thông tin từ các tài liệu có độ tin cậy (relevance score) cao hơn khi có xung đột hoặc mâu thuẫn
    8. Khi trả lời, ưu tiên thông tin từ tài liệu có điểm số cao nhất trước
    """
    
    full_prompt = f"{system_instructions}\n\nCâu hỏi: {query}\n\nNgữ cảnh từ các tài liệu pháp lý (đã được sắp xếp theo độ tin cậy):\n{documents}\n\nDựa vào ngữ cảnh trên, hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ. Nếu các tài liệu không chứa thông tin liên quan trực tiếp đến câu hỏi, hãy nói rõ rằng bạn không tìm thấy thông tin cụ thể về vấn đề này và KHÔNG trích dẫn bất kỳ tài liệu nào."
    
    return full_prompt

def generate_response(query, documents):
    """
    Generates a response using the LLM based on the query and documents
    
    Args:
        query (str): User query
        documents: Retrieved documents
        
    Returns:
        tuple: (response_text, should_display_sources)
    """
    try:
        # Format documents and create the prompt
        formatted_docs = format_documents(documents)
        full_prompt = create_prompt(query, formatted_docs)
        
        # Initialize the model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Generate the response
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                    top_p=0.95,
                )
            )
            response_text = response.text
            
            # Always display sources (removed the conditional check)
            return response_text, True
            
        except Exception as config_error:
            print(f"Error with generation config, trying simpler call: {config_error}")
            response = model.generate_content(full_prompt)
            return response.text, True  # Always show sources
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"Đã xảy ra lỗi khi tạo phản hồi: {str(e)}", False
    

def format_sources_for_display(documents, processing_time):
    """
    Formats source documents for display in the UI
    
    Args:
        documents: Retrieved documents
        processing_time (float): Time taken to process the query
        
    Returns:
        str: Formatted source information in Markdown
    """
    sources_text = f"### Nguồn tài liệu tham khảo (Thời gian xử lý: {processing_time:.2f}s)\n\n"
    
    for i, doc in enumerate(documents):
        payload = doc.payload
        document_name = payload.get('document_name', 'Unknown')
        reference_id = payload.get('reference_id', 'Unknown')
        section_title = payload.get('section_title', '')
        
        # Extract Điều number with improved pattern matching with None checking
        article_headline = payload.get('article_headline', '')
        dieu_match = None
        
        # Try multiple sources to find the article number, safely checking for None
        if article_headline and isinstance(article_headline, str):
            dieu_match = re.search(r'điều\s+(\d+)', article_headline.lower())
        
        if not dieu_match and section_title and isinstance(section_title, str):
            dieu_match = re.search(r'điều\s+(\d+)', section_title.lower())
        
        if not dieu_match and reference_id and isinstance(reference_id, str):
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

def process_query(query, num_docs=5):
    """
    Process a user query through the RAG pipeline
    
    Args:
        query (str): User query
        num_docs (int): Number of documents to retrieve
        
    Returns:
        tuple: (answer, sources_text)
    """
    if not query.strip():
        return "Vui lòng nhập câu hỏi của bạn.", ""
    
    try:
        # Track processing time
        start_time = time.time()
        
        # Step 1, retrieve documents
        print(f"Retrieving documents for query: {query}")
        retrieved_docs = search_legal_documents(query, limit=num_docs)
        
        debug_document_metadata(retrieved_docs)
        
        if not retrieved_docs:
            return "Không tìm thấy tài liệu liên quan đến câu hỏi của bạn.", ""
        
        # Step 2, generate answer
        print(f"Generating response with {len(retrieved_docs)} documents")
        answer, should_display_sources = generate_response(query, retrieved_docs)
        
        # Step 3, format sources for display (always show sources)
        sources_text = format_sources_for_display(retrieved_docs, time.time() - start_time)
        
        return answer, sources_text
    
    except Exception as e:
        error_message = f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
        print(error_message)
        return error_message, ""

def process_baseline_model(query):
    """
    Process query with just the model, no retrieval or sources
    
    Args:
        query (str): User query
        
    Returns:
        str: Generated response
    """
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
        error_message = f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
        print(error_message)
        return error_message

def process_retrieval_only(query, num_docs):
    """
    Process query through retrieval only, no model generation
    
    Args:
        query (str): User query
        num_docs (int): Number of documents to retrieve
        
    Returns:
        tuple: (documents, sources_text)
    """
    if not query.strip():
        return "Vui lòng nhập câu hỏi của bạn.", ""
    
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
        return f"### Các tài liệu liên quan\n\n{formatted_docs}", sources_text
    
    except Exception as e:
        error_message = f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
        print(error_message)
        return error_message, ""

if __name__ == "__main__":
    # Test the system with a sample query
    query = "dieu 81 trong nghi dinh 46 la gi"
    answer, sources = process_query(query)
    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== SOURCES ===")
    print(sources)