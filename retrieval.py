# initial setup and configuration
# this block sets up the environment, imports the needed libraries 
# and configures the connection to qdrant (our vector database)
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
import re
import dotenv
import unicodedata

#load environment variables from .env file
dotenv.load_dotenv(".env.local")

#qdrant configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = 'legal_docs'

#initialize global variables
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

#load model
model = SentenceTransformer('keepitreal/vietnamese-sbert')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)



# define a set of common viet stopwords
# these will be filtered out in text processing 
# bc they don't carry much meaning 
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'cho', 'là', 'để', 'trong', 'với', 'các', 'có', 'được', 
    'tại', 'về', 'từ', 'theo', 'đến', 'không', 'những', 'này', 'đó', 'khi'
}

# normalize vietnamese text function
# removes accent marks from viet characters
# converts all the text to lowercase
def normalize_vietnamese_text(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.lower()



# extract query entities function
# 1. identifies and extracts specific entities from user queries
# 2. finds references to dieu and thong tu using regex patterns
# 3. extracts important keywords from the query
# 4. returns a dictionary of all extracted entities
def extract_query_entities(query):
    query = query.lower()
    entities = {}
     
    #find dieu references (more flexible patterns)
    dieu_patterns = [
        r'điều\s+(\d+)',  #standard format eg. Điều 4
        r'điều\s+(\d+)\s*[,\.]\s*điều\s+(\d+)',  #multiple, eg. Điều 2, Điều 3
        r'điều\s*(\d+)[-–]\s*(\d+)',  #range, eg. Điều 4-6
    ]
    
    dieu_matches = []
    for pattern in dieu_patterns:
        matches = re.findall(pattern, query)
        if matches:
            #handle tuples from multiple capturing groups
            for match in matches:
                if isinstance(match, tuple):
                    dieu_matches.extend(list(match))
                else:
                    dieu_matches.append(match)
    
    if dieu_matches:
        entities['dieu'] = list(set(dieu_matches))  #remove duplicates
    
    #find tt references (more flexible patterns)
    tt_patterns = [
        r'thông\s+tư\s+(?:số\s+)?(\d+)',  #standard, eg. Thông tư 67
        r'tt\s*(\d+)',  #abbreviated, eg. TT67
        r'thông\s+tư\s+(?:số\s+)?(\d+)/\d+/[\w-]+',  #full reference with year and code
    ]
    
    tt_matches = []
    for pattern in tt_patterns:
        matches = re.findall(pattern, query)
        tt_matches.extend(matches)
    
    if tt_matches:
        entities['thong_tu'] = [f"tt{num}" for num in set(tt_matches)]  #remove duplicates
    
    #extract keywords for better context matching
    keywords = extract_keywords(query)
    if keywords:
        entities['keywords'] = keywords
    
    return entities



# keyword extraction and matching functions
# 1. extract_keywords removes entity mentions from the query
#    filters out stopwords and returns unique keywords
# 2. calculate_keyword_match_score determines relevance
#    based on how many query keywords appear in document text
def extract_keywords(query, max_keywords=5):
    #remove entity mentions to focus on topic
    clean_query = re.sub(r'điều\s+\d+|thông\s+tư\s+\d+', '', query)
    
    #simple word tokenization
    words = re.findall(r'\b\w+\b', clean_query.lower())
    
    #filter out stopwords
    keywords = [word for word in words if len(word) > 2 and word not in VIETNAMESE_STOPWORDS]
    
    #return unique keywords
    return list(set(keywords))[:max_keywords]

def calculate_keyword_match_score(keywords, text):
    if not keywords:
        return 1.0  #no keywords to match
    
    text_lower = text.lower()
    matched_keywords = sum(keyword in text_lower for keyword in keywords)
    
    #calculate score based on proportion of matched keywords
    if matched_keywords == 0:
        return 0.6  #base score if no matches
    
    #higher score for more keyword matches
    return 0.7 + (0.3 * matched_keywords / len(keywords))



# strict entity filter function
# 1. filters results to strictly match entity requirements
# 2. only returns results matching all specified entities
# 3. for example, with query "dieu 4 trong tt67"
#    it filters out everything that doesn't match both criteria
def strict_entity_filter(results, entities):
    if not entities or not results:
        return results
    
    filtered_results = []
    
    for result in results:
        metadata = result.payload.get('metadata', {})
        
        #check for exact matches of dieu if specified
        if 'dieu' in entities:
            if metadata.get('Điều') not in entities['dieu']:
                continue  #skip this result if dieu doesn't match
        
        #check for exact matches of tt
        if 'thong_tu' in entities:
            if metadata.get('Thông tư') not in entities['thong_tu']:
                continue  #skip
        
        #if we got here, all entity conditions are satisfied
        filtered_results.append(result)
    
    return filtered_results



# entity-aware search function
# 1. encodes the query using the viet language model
# 2. searches qdrant db with the encoded vector
# 3. applies strict filtering based on extracted entities
# 4. implements fallback strategy if too few results found
# 5. ensures search prioritizes entity matches
#    while still providing relevant results
def search_with_entities(query, entities, limit):
    #vector encoding
    query_vector = model.encode(
        query,
        convert_to_numpy=True,
        show_progress_bar=False,
        device=device
    ).tolist()
    
    #first approach: get a larger set of results and filter afterward
    #semantic searching with payload (metadata)
    all_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit * 3,  #get more to allow for filtering
        with_payload=True
    )
    
    #apply strict filtering based on entities
    if 'dieu' in entities or 'thong_tu' in entities:
        filtered_results = strict_entity_filter(all_results, entities)
        
        #if we have enough results after filtering, return them
        if len(filtered_results) >= 3:
            return filtered_results
        
        #or else we'll need to search again with the first entity of each type
        #to get more specific results (this is a fallback to ensure we get results)
        # ANY not ALL 
        fallback_entities = {} 
        if 'dieu' in entities and entities['dieu']:
            fallback_entities['dieu'] = [entities['dieu'][0]]
        if 'thong_tu' in entities and entities['thong_tu']:
            fallback_entities['thong_tu'] = [entities['thong_tu'][0]]
        
        #if we have specific entities from the query, use them for focused search
        if fallback_entities:
            #get more focused results with the first entity
            focused_results = []
            for result in all_results:
                metadata = result.payload.get('metadata', {})
                
                #check if this result matches any of our fallback entities
                if ('dieu' in fallback_entities and 
                    metadata.get('Điều') == fallback_entities['dieu'][0]):
                    focused_results.append(result)
                elif ('thong_tu' in fallback_entities and 
                     metadata.get('Thông tư') == fallback_entities['thong_tu'][0]):
                    focused_results.append(result)
            
            #if we found focused results ten return those
            if focused_results:
                return focused_results
    
    #return all results if no entity filtering is required or if filtering produced too few results
    return all_results



# post-processing function
# 1. removes duplicate content from results
# 2. boosts scores for exact entity matches
# 3. increases scores for content matching more keywords
# 4. combines vector similarity and entity matching scores
# 5. sorts results by final relevance score
def post_process_results(results, entities, query, limit=5):
    if not results:
        return []
    
    processed_results = []
    seen_content = set()  #track unique content to remove duplicates
    
    for result in results:
        metadata = result.payload.get('metadata', {})
        text = result.payload.get('text', '')
        
        #skip if we've seen this exact text already (deduplicate)
        #use first 100 chars as a content signature
        text_signature = text[:100].strip()
        if text_signature in seen_content:
            continue
        seen_content.add(text_signature)
        
        #base score from vector similarity
        base_score = result.score
        match_score = 1.0
        
        #entity matching boost to give high priority to exact matches
        if 'dieu' in entities and metadata.get('Điều'):
            if metadata.get('Điều') in entities['dieu']:
                match_score *= 2.0  #strong boost for exact dieu match
            else:
                match_score *= 0.3  #heavy penalty for non matching dieu
        
        if 'thong_tu' in entities and metadata.get('Thông tư'):
            if metadata.get('Thông tư') in entities['thong_tu']:
                match_score *= 2.0  #strong boost
            else:
                match_score *= 0.3  #heavy penalty
        
        #content relevance based on keywords
        if 'keywords' in entities and entities['keywords']:
            keyword_score = calculate_keyword_match_score(entities['keywords'], text)
            match_score *= keyword_score
        
        #calculate the final score as a weighted combination
        final_score = (base_score * 0.4) + (match_score * 0.6)  #greater weight to matches
        
        #create processed result with updated score
        processed_result = result
        processed_result.score = final_score
        processed_results.append(processed_result)
    
    #sort by final score
    processed_results.sort(key=lambda x: x.score, reverse=True)
    
    return processed_results[:limit]   



# result formatting function
# 1. makes text more readable and focused
# 2. attempts to start excerpt near keyword matches
# 3. tries to end at sentence boundaries when possible
# 4. adds ellipsis when truncating longer texts
def format_result_text(text, query, entities, max_length=400):
    #if text is shorter than max_length then return it entirely
    if len(text) <= max_length:
        return text
    
    #find a good starting point that includes keywords if possible
    keywords = entities.get('keywords', [])
    start_pos = 0
    best_keyword_pos = None
    
    for keyword in keywords:
        keyword_pos = text.lower().find(keyword.lower())
        if keyword_pos != -1:
            #found a keyword and check if it's a good starting point
            if best_keyword_pos is None or (keyword_pos < best_keyword_pos and keyword_pos > 10):
                best_keyword_pos = keyword_pos
    
    #if found a good keyword position then start a bit before it
    if best_keyword_pos is not None:
        start_pos = max(0, best_keyword_pos - 50)
    
    #find a good breakpoint (end of sentence or paragraph)
    end_pos = min(start_pos + max_length, len(text))
    
    #try to end at a sentence boundary if possible
    sentence_end = text.rfind('. ', start_pos, end_pos)
    if sentence_end != -1:
        end_pos = sentence_end + 2  #include the period and space
    
    #add ellipsis if we're truncating
    suffix = "..." if end_pos < len(text) else ""
    prefix = "..." if start_pos > 0 else ""
    
    return prefix + text[start_pos:end_pos] + suffix



# the main function
# 1. extracts the entities and keywords from the query
# 2. performs a search with entity awareness
# 3. applies strict entity filtering
# 4. post processes the results to improve relevance
# 5. formats and displays the results with metadata
def search_legal_documents(query, limit = 3):
    try:
        print(f"Searching for: '{query}'")
        
        #extract entities and keywords from query
        entities = extract_query_entities(query)
        if entities:
            entity_str = ", ".join([f"{k}: {v}" for k, v in entities.items()]) #key - value pairs
            print(f"Extracted entities: {entity_str}")
        
        #do search with entity aware approach
        results = search_with_entities(query, entities, 30)
        
        print(f"Found {len(results)} initial results") 
        
        #apply strict entity filtering for dieu and thong tu
        if ('dieu' in entities or 'thong_tu' in entities) and results:
            results = strict_entity_filter(results, entities)
            print(f"After strict filtering: {len(results)} results")
        
        #enhanced post processing with context matching
        filtered_results = post_process_results(results, entities, query, limit)
        
        print(f"Filtered to {len(filtered_results)} most relevant results")
        
        #display results with better formatting
        for i, result in enumerate(filtered_results):
            print(f"\nResult {i+1} [Score: {result.score:.4f}]")
            
            metadata = result.payload.get('metadata', {})
            thong_tu = metadata.get('Thông tư', 'Unknown')
            dieu = metadata.get('Điều', 'Unknown')
            
            print(f"From: Thông tư {thong_tu}, Điều {dieu}")
            
            #show better formatted text excerpt
            text = result.payload.get('text', '')
            text_preview = format_result_text(text, query, entities)
            print(f"Text excerpt: {text_preview}")
            print("-" * 80)
        
        return filtered_results
    
    except Exception as e:
        print(f"Error during search: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
    


# example queries and interactive search
# 1. runs example queries to demonstrate search results 
# 2. allows user to input queries interactively
# 3. displays top 3 results for each query
def run_example_queries():
    example_queries = [
        "Bảo hiểm xe cơ giới",
        "Điều 4 quy định về gì",
        "Điều 2 và Điều 3 trong thông tư 67 bao gồm những gì",
    ]
    
    for query in example_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: '{query}'")
        print("=" * 80)
        
        search_legal_documents(query, limit=3)  #show top 3 results

def interactive_search():
    print("\n=== Interactive Legal Document Search ===")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your search query: ")
        if query.lower() in ('exit', 'quit'):
            break
        
        print("\n" + "=" * 80)
        search_legal_documents(query, limit=3)


if __name__ == "__main__":
    print(f"Connected to Qdrant at: {QDRANT_URL}")

    run_example_queries()  

    #interactive_search()