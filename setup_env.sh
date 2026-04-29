#!/bin/bash
# Setup script for Medical RAG Application

echo "Setting up Medical RAG Application..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Override defaults
# OPENAI_MODEL=gpt-3.5-turbo
# EMBEDDING_MODEL=text-embedding-3-small
# TOP_K_RESULTS=3
# VECTOR_STORE_PATH=vector_store.pkl
EOF
    echo "✅ .env file created. Please edit it and add your OpenAI API key."
else
    echo "⚠️  .env file already exists."
fi

echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENAI_API_KEY"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Run the app: streamlit run app.py"

