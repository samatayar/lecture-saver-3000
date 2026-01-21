#!/usr/bin/env python3
"""
Test script for the new database management features
"""

import sys
import os
sys.path.append('.')

def test_database_functions():
    """Test database management functions"""
    try:
        from app import check_existing_collections, get_collection_stats, initialize_chroma_db

        print("Testing database functions...")

        # Test checking existing collections
        collections = check_existing_collections()
        print(f"[OK] Found {len(collections)} collections: {collections}")

        if collections:
            # Test initializing a collection
            collection = initialize_chroma_db(collection_name=collections[0])
            print(f"[OK] Successfully initialized collection: {collections[0]}")

            # Test getting collection stats
            stats = get_collection_stats(collection)
            print(f"[OK] Collection has {stats} documents")

        print("All database functions working correctly!")
        return True

    except Exception as e:
        print(f"[FAIL] Error testing database functions: {e}")
        return False

if __name__ == "__main__":
    success = test_database_functions()
    exit(0 if success else 1)