# test_reranker_basic.py
import os
import dotenv
from core.llm_reranker import get_reranker
from core.retrieval import search_legal_documents

# Load environment variables
dotenv.load_dotenv(".env.local")

def test_basic_functionality():
    """Test that the reranker is working at a basic level"""
    
    # Simple test query
    query = "Các trường hợp loại trừ bồi thường bảo hiểm con người?"
    
    # Run search with reranking disabled
    print("SEARCHING WITHOUT RERANKING...")
    results_no_rerank = search_legal_documents(query, limit=3, use_reranking=False)
    
    # Store original order and scores
    original_order = [(doc.id, doc.score) for doc in results_no_rerank]
    
    # Run with reranking enabled
    print("\nSEARCHING WITH RERANKING...")
    results_reranked = search_legal_documents(query, limit=3, use_reranking=True)
    
    # Store new order and scores
    reranked_order = [(doc.id, doc.score) for doc in results_reranked]
    
    # Check if reranking changed anything
    order_changed = original_order != reranked_order
    print(f"\nDid reranking change the order or scores? {'Yes' if order_changed else 'No'}")
    
    # Print detailed comparison if order changed
    if order_changed:
        print("\nOriginal order:")
        for i, (doc_id, score) in enumerate(original_order):
            print(f"  {i+1}. Document {doc_id}: Score {score:.4f}")
            
        print("\nReranked order:")
        for i, (doc_id, score) in enumerate(reranked_order):
            print(f"  {i+1}. Document {doc_id}: Score {score:.4f}")
    
    # Direct reranker test
    print("\nTESTING RERANKER DIRECTLY...")
    reranker = get_reranker()
    
    # Check if we can manually rerank the results
    if results_no_rerank:
        direct_reranked = reranker.rerank(query, results_no_rerank, top_k=3)
        print(f"Direct reranker returned {len(direct_reranked)} results")
        
        # Print what the reranker did
        print("\nDirect reranking results:")
        for i, doc in enumerate(direct_reranked):
            print(f"  {i+1}. Document {doc.id}: Score {doc.score:.4f}")

if __name__ == "__main__":
    test_basic_functionality()