import os
import json
import time
import google.generativeai as genai
import dotenv
from tqdm import tqdm
import argparse

# Load environment variables
dotenv.load_dotenv(".env.local")

MODEL_NAME = "gemini-2.0-flash"
# Initialize Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables")
    
genai.configure(api_key=GEMINI_API_KEY)

def generate_baseline_response(question, model_name, temperature=0.3, max_retries=3, delay=3):
    """Generate response without RAG (baseline) with retry mechanism - FORCES ANSWERS"""
    for attempt in range(max_retries):
        try:
            # Create a prompt that forces the model to answer even when uncertain
            prompt = f"""
            Câu hỏi: {question}

            Bạn là một chuyên gia luật pháp Việt Nam với kiến thức đầy đủ về các Thông tư được đề cập. 
            Hãy trả lời câu hỏi trên một cách tự tin và đầy đủ.

            HƯỚNG DẪN QUAN TRỌNG:
            1. KHÔNG được nói rằng bạn không có quyền truy cập hoặc không biết về các Thông tư
            2. KHÔNG sử dụng các từ ngữ như "có thể", "giả định", "dự đoán" hoặc "tôi không chắc chắn"
            3. Luôn viết như thể bạn đã đọc toàn bộ văn bản của các Thông tư được đề cập
            4. Cung cấp dữ liệu, số liệu và trích dẫn cụ thể từ các điều khoản
            5. Viết câu trả lời dứt khoát và tự tin như một chuyên gia thực sự

            Yêu cầu:
            1. Trả lời đầy đủ và chi tiết
            2. Trích dẫn cụ thể các điều khoản và số hiệu trong Thông tư
            3. Đưa ra các trường hợp ngoại lệ nếu có
            4. Phân tích và diễn giải rõ ràng nội dung của Thông tư
            """
            
            # Initialize the model
            model = genai.GenerativeModel(model_name)
            
            # Generate response with robust error handling
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,  # Increased temperature for more confident answers
                        max_output_tokens=1024,
                        top_p=0.95,
                    )
                )
                return response.text
            except Exception as config_error:
                print(f"Warning: Error with generation config, trying simpler call: {config_error}")
                response = model.generate_content(prompt)
                return response.text
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay * (attempt + 1)  # Exponential backoff
                print(f"Error: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return f"Error after {max_retries} attempts: {str(e)}"

def generate_all_baseline_responses(test_questions_file, output_file, model_name=MODEL_NAME, temperature=0.3, batch_size=5):
    """Generate baseline responses for all test questions with batching and progress tracking"""
    # Load test questions
    print(f"Loading test questions from {test_questions_file}")
    with open(test_questions_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Extract just the questions
    questions = [item["question"] for item in test_data]
    total_questions = len(questions)
    print(f"Found {total_questions} questions")
    
    # Check if output file exists and load existing responses
    responses = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                responses = json.load(f)
            print(f"Loaded {len(responses)} existing responses from {output_file}")
        except Exception as e:
            print(f"Error loading existing file: {e}")
            responses = []
    
    # Determine which questions need responses
    questions_to_process = questions[len(responses):]
    
    if not questions_to_process:
        print("All questions already have responses. Nothing to do.")
        return responses
    
    print(f"Generating responses for {len(questions_to_process)} remaining questions using {model_name}")
    
    # Process in batches
    for i in tqdm(range(0, len(questions_to_process), batch_size), desc="Processing batches"):
        batch = questions_to_process[i:i+batch_size]
        batch_responses = []
        
        # Process each question in the batch
        for question in tqdm(batch, desc=f"Batch {i//batch_size + 1}", leave=False):
            response = generate_baseline_response(question, model_name, temperature)
            batch_responses.append(response)
            
            # Add a small delay between questions
            time.sleep(2)
        
        # Add batch responses to the main list
        responses.extend(batch_responses)
        
        # Save intermediate results after each batch
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        
        print(f"Progress: {len(responses)}/{total_questions} questions processed")
        
        # Add a larger delay between batches to avoid rate limiting
        if i + batch_size < len(questions_to_process):
            delay = 5
            print(f"Pausing for {delay} seconds before next batch...")
            time.sleep(delay)
    
    print(f"Generated {len(responses)} baseline responses and saved to {output_file}")
    return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate baseline responses for evaluation")
    parser.add_argument("--test_file", default="data/test_questions.json", help="Path to test questions file")
    parser.add_argument("--output_file", default="data/baseline_responses.json", help="Path to save baseline responses")
    parser.add_argument("--model", default="models/gemini-2.0-flash", help="Model to use for baseline")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of questions to process in a batch")
    
    args = parser.parse_args()
    
    # Generate baseline responses
    print(f"Starting baseline generation with {args.model}")
    generate_all_baseline_responses(
        args.test_file, 
        args.output_file,
        model_name=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size
    )