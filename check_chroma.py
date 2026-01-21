import chromadb
import json

client = chromadb.PersistentClient(path='./chromaDB')
collections = client.list_collections()

print('Available collections:')
for col in collections:
    print(f'- {col.name} ({col.count()} documents)')

    # Check if this is the "Final" collection
    if col.name == "Final":
        print(f'  *** This is the "Final" collection with {col.count()} documents ***')

    # Peek at first few documents
    try:
        results = col.peek(limit=2)
        if results['documents']:
            print(f'  Sample document: {results["documents"][0][:100]}...')
            if results['metadatas'] and results['metadatas'][0]:
                print(f'  Metadata: {results["metadatas"][0]}')
    except Exception as e:
        print(f'  Error peeking: {e}')
    print()

# Specifically check for the "Final" collection
print('Checking specifically for "Final" collection:')
try:
    final_collection = client.get_collection(name="Final")
    print(f'[SUCCESS] Found "Final" collection with {final_collection.count()} documents')
except Exception as e:
    print(f'[ERROR] "Final" collection not found: {e}')