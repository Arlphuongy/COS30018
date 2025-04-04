# enhanced_qdrant_inspector.py
# Improved script to thoroughly inspect Qdrant collection structure

import os
import dotenv
import json
from qdrant_client import QdrantClient
from pprint import pprint
import inspect

# Load environment variables from .env file
dotenv.load_dotenv(".env.local")

# Qdrant configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_FULL')

# Initialize client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

def inspect_collection():
    """Thoroughly examine collection structure and vector configuration"""
    print(f"Connecting to Qdrant at: {QDRANT_URL}")
    print(f"Inspecting collection: {COLLECTION_NAME}")
    
    # 1. List all collections to confirm connection
    try:
        collections = client.get_collections()
        print("\n=== Available Collections ===")
        for collection in collections.collections:
            print(f"- {collection.name}")
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
    
    # 2. Get detailed collection info with internal structure examination
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print("\n=== Collection Information ===")
        print(f"Points count: {collection_info.points_count}")
        
        # Introspect collection_info object to understand its structure
        print("\n=== Collection Info Object Structure ===")
        print(f"Type: {type(collection_info)}")
        print("Attributes:", dir(collection_info))
        
        # Print config structure
        if hasattr(collection_info, 'config'):
            print("\nConfig attributes:", dir(collection_info.config))
            
            if hasattr(collection_info.config, 'params'):
                print("\nConfig params attributes:", dir(collection_info.config.params))
        
        # Check vector configuration - critical for your search issue
        vector_info = None
        vector_fields = None
        
        # Try different potential paths to get vector info depending on Qdrant version
        if hasattr(collection_info.config.params, 'vectors'):
            print("\n=== Vector Configuration (Multiple Vectors) ===")
            vectors_config = collection_info.config.params.vectors
            
            # Try to access as dictionary or object
            if hasattr(vectors_config, 'model_dump'):
                # Newer Pydantic v2
                vector_info = vectors_config.model_dump()
                print("Vector info from model_dump():")
                pprint(vector_info)
                vector_fields = list(vector_info.keys())
            elif hasattr(vectors_config, 'dict') and callable(vectors_config.dict):
                # Older Pydantic v1 
                vector_info = vectors_config.dict()
                print("Vector info from dict():")
                pprint(vector_info)
                vector_fields = list(vector_info.keys())
            else:
                # Try direct attribute access
                print("Vector info from direct attribute access:")
                print(f"Vectors object: {vectors_config}")
                print(f"Vectors dir: {dir(vectors_config)}")
                
                # Try to convert to dictionary using __dict__
                if hasattr(vectors_config, '__dict__'):
                    vector_info = vectors_config.__dict__
                    pprint(vector_info)
                    vector_fields = list(vector_info.keys())
                
        # For older Qdrant where vector config is directly on params
        elif hasattr(collection_info.config.params, 'vector_size'):
            print("\n=== Vector Configuration (Single Vector) ===")
            print(f"Vector size: {collection_info.config.params.vector_size}")
            print(f"Distance: {collection_info.config.params.distance}")
            vector_fields = ['vector']  # Default name for older Qdrant
        elif hasattr(collection_info.config.params, 'size'):
            print("\n=== Vector Configuration (Single Vector) ===")
            print(f"Vector size: {collection_info.config.params.size}")
            print(f"Distance: {collection_info.config.params.distance}")
            vector_fields = ['vector']  # Default name for older Qdrant
            
        # Output vector field names (critical for search)
        if vector_fields:
            print("\n=== Vector Field Names for Search ===")
            print(f"Vector fields to use in search: {vector_fields}")
            print("Note: For query_points(), use these field names with the 'using' parameter")
        else:
            print("\nCould not determine vector field names")
            
        # Dump full collection info as JSON for examination
        print("\n=== Full Collection Info (JSON) ===")
        try:
            # Try to convert to dict with Pydantic model_dump
            if hasattr(collection_info, 'model_dump'):
                collection_dict = collection_info.model_dump()
            # Try with older Pydantic dict() method
            elif hasattr(collection_info, 'dict') and callable(collection_info.dict):
                collection_dict = collection_info.dict()
            # Direct conversion might fail with recursion
            else:
                collection_dict = {"info": "Could not convert to dict"}
                
            print(json.dumps(collection_dict, indent=2, default=str))
        except Exception as e:
            print(f"Error converting collection info to JSON: {str(e)}")
            
    except Exception as e:
        print(f"Error getting collection info: {str(e)}")
    
    # 3. Test search API calls to determine which format works
    try:
        print("\n=== Testing Search API Calls ===")
        # Create a fake query vector of the right dimension
        vector_dim = 768  # Default for typical embedding models, adjust if needed
        try:
            if hasattr(collection_info.config.params, 'vectors'):
                # Try to get dimension from the first vector field
                vector_config = next(iter(vector_info.values())) if vector_info else None
                if vector_config and 'size' in vector_config:
                    vector_dim = vector_config['size']
            elif hasattr(collection_info.config.params, 'size'):
                vector_dim = collection_info.config.params.size
        except:
            pass  # Use default dimension if we can't determine it
            
        print(f"Using test vector with dimension: {vector_dim}")
        test_vector = [0.1] * vector_dim  # Simple test vector
        
        # Try different search API calls to see which succeeds
        # 1. Modern query_points with default vector field
        try:
            print("\nTesting: client.query_points(query_vector=vector)")
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query_vector=test_vector,
                limit=1
            )
            print("✓ Success! Modern API call works with unnamed vector")
            print(f"Result count: {len(results.points) if hasattr(results, 'points') else 0}")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
        
        # 2. Modern query_points with named vector field
        if vector_fields:
            for field in vector_fields:
                try:
                    print(f"\nTesting: client.query_points(query_vector=({field}, vector))")
                    results = client.query_points(
                        collection_name=COLLECTION_NAME,
                        query_vector=(field, test_vector),
                        limit=1
                    )
                    print(f"✓ Success! Modern API call works with named vector field '{field}'")
                    print(f"Result count: {len(results.points) if hasattr(results, 'points') else 0}")
                except Exception as e:
                    print(f"✗ Failed with field '{field}': {str(e)}")
        
        # 3. Legacy search method
        try:
            print("\nTesting: client.search(query_vector=vector)")
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=test_vector,
                limit=1
            )
            print("✓ Success! Legacy search API works (with deprecation warning)")
            print(f"Result count: {len(results)}")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            
        # 4. Legacy search with named vector
        if vector_fields:
            for field in vector_fields:
                try:
                    print(f"\nTesting: client.search(query_vector=({field}, vector))")
                    results = client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=(field, test_vector),
                        limit=1
                    )
                    print(f"✓ Success! Legacy search works with named vector field '{field}'")
                    print(f"Result count: {len(results)}")
                except Exception as e:
                    print(f"✗ Failed with field '{field}': {str(e)}")
                    
    except Exception as e:
        print(f"Error testing search API calls: {str(e)}")
        
    # 4. Get sample points to examine payload structure
    try:
        print("\n=== Sample Payload Structure ===")
        # Scroll through some points
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if points and points[0]:
            sample_point = points[0][0]
            print("\nPayload Fields:")
            if hasattr(sample_point, 'payload') and sample_point.payload:
                for key, value in sample_point.payload.items():
                    print(f"- {key} ({type(value).__name__})")
                    # Show value snippet based on type
                    if isinstance(value, str):
                        print(f"  Sample: {value[:100]}...")
                    elif isinstance(value, (list, tuple)):
                        print(f"  Length: {len(value)}, Sample: {str(value[:2])[:100]}...")
                    elif isinstance(value, dict):
                        print(f"  Keys: {list(value.keys())[:5]}")
                    else:
                        print(f"  Value: {value}")
            else:
                print("No payload found in sample point")
    except Exception as e:
        print(f"Error examining payload structure: {str(e)}")

    # 5. Look specifically for fields used in your entity filtering
    try:
        print("\n=== Fields for Entity Filtering ===")
        # Fields typically used in your retrieval code
        important_fields = [
            'document_name', 'article_headline', 'reference_id',
            'content', 'section_id', 'section_title'
        ]
        
        # Check a few points for these fields
        all_samples = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False
        )[0]
        
        field_presence = {field: 0 for field in important_fields}
        
        for point in all_samples:
            if hasattr(point, 'payload') and point.payload:
                for field in important_fields:
                    if field in point.payload:
                        field_presence[field] += 1
        
        print("Fields found in sample payloads:")
        for field, count in field_presence.items():
            print(f"- {field}: found in {count}/{len(all_samples)} samples")
            
            # Show a sample of the first non-empty value
            if count > 0:
                for point in all_samples:
                    if field in point.payload and point.payload[field]:
                        value = point.payload[field]
                        if isinstance(value, str):
                            print(f"  Sample: {value[:100]}...")
                        else:
                            print(f"  Sample: {str(value)[:100]}...")
                        break
            
    except Exception as e:
        print(f"Error checking entity filtering fields: {str(e)}")

if __name__ == "__main__":
    inspect_collection()