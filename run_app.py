#!/usr/bin/env python3
"""
Script to run the Document Chat Assistant Streamlit application.
This script checks for required dependencies and starts the application.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_secrets():
    """Check if secrets are configured"""
    secrets_file = Path(".streamlit/secrets.toml")
    if not secrets_file.exists():
        print("Warning: .streamlit/secrets.toml not found!")
        print("Please create this file with your API keys.")
        print("See .streamlit/secrets.toml.example for the format.")
        return False

    # Check if OPENROUTER_API_KEY is set
    try:
        with open(secrets_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'OPENROUTER_API_KEY' not in content or 'your-openrouter-api-key' in content:
                print("Warning: OPENROUTER_API_KEY not configured in secrets.toml!")
                return False
    except Exception as e:
        print(f"Error reading secrets file: {e}")
        return False

    return True

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import docling
        import chromadb
        import openai
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install requirements:")
        print("  pip install -r requirements.txt")
        return False

def main():
    """Main function to run the application"""
    print("Document Chat Assistant")
    print("=" * 40)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Check secrets
    if not check_secrets():
        print("\nYou can still run the app, but some features may not work without API keys.")

    # Start Streamlit app
    print("\nStarting Streamlit application...")
    print("Open your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop the application\n")

    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Could not find streamlit. Please install it with:")
        print("  pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()