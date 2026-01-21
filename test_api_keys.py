#!/usr/bin/env python3
"""
Test script to check API key functionality
"""

import sys
import os
sys.path.append('.')

def test_api_keys():
    """Test API key retrieval"""
    try:
        # Import streamlit (this might not work outside streamlit context)
        import streamlit as st

        # Test get_api_keys function
        from app import get_api_keys

        print("Testing API key retrieval...")
        keys = get_api_keys()

        print(f"OpenRouter key: {'Found' if keys['openrouter'] else 'NOT FOUND'}")
        print(f"HuggingFace key: {'Found' if keys['huggingface'] else 'NOT FOUND'}")

        if keys['openrouter']:
            print(f"OpenRouter key length: {len(keys['openrouter'])}")
            print(f"OpenRouter key starts with: {keys['openrouter'][:20]}...")
        else:
            print("OpenRouter key is empty!")

        if keys['huggingface']:
            print(f"HuggingFace key length: {len(keys['huggingface'])}")
            print(f"HuggingFace key starts with: {keys['huggingface'][:10]}...")
        else:
            print("HuggingFace key is empty!")

        return keys['openrouter'] and keys['huggingface']

    except Exception as e:
        print(f"Error testing API keys: {e}")
        return False

if __name__ == "__main__":
    success = test_api_keys()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)