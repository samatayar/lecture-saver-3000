#!/usr/bin/env python3
"""Test basic imports for the Document Chat Assistant"""

def test_basic_imports():
    """Test basic Python imports"""
    try:
        import os
        import tempfile
        import shutil
        from pathlib import Path
        import json
        import re
        import time
        from typing import List, Dict, Any
        print("[OK] Basic imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Basic import failed: {e}")
        return False

def test_streamlit_import():
    """Test Streamlit import"""
    try:
        import streamlit as st
        print("[OK] Streamlit import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Streamlit import failed: {e}")
        return False

def test_docling_import():
    """Test Docling import"""
    try:
        from docling.document_converter import DocumentConverter
        print("[OK] Docling import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Docling import failed: {e}")
        return False

def test_langchain_import():
    """Test LangChain imports"""
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        print("[OK] LangChain imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] LangChain imports failed: {e}")
        return False

def test_sentence_transformers_import():
    """Test Sentence Transformers import"""
    try:
        from sentence_transformers import SentenceTransformer
        print("[OK] Sentence Transformers import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Sentence Transformers import failed: {e}")
        return False

def test_chromadb_import():
    """Test ChromaDB import"""
    try:
        import chromadb
        print("[OK] ChromaDB import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] ChromaDB import failed: {e}")
        return False

def test_openai_import():
    """Test OpenAI import"""
    try:
        from openai import OpenAI
        print("[OK] OpenAI import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] OpenAI import failed: {e}")
        return False

def main():
    """Run all import tests"""
    print("Testing imports for Document Chat Assistant...")
    print("-" * 50)

    tests = [
        test_basic_imports,
        test_streamlit_import,
        test_docling_import,
        test_langchain_import,
        test_sentence_transformers_import,
        test_chromadb_import,
        test_openai_import,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("-" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("All imports successful! Ready to run the application.")
        return True
    else:
        print("Some imports failed. Please install missing packages:")
        print("   pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)