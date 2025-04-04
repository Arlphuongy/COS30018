# COS30018 - Vietnamese Legal Assistant with RAG

## Prerequisites

- Python 3.x installed
- Required Python packages (install using `pip install -r requirements.txt`)

## Setup

1. **Create an `env.local` file**
   - In the root directory, create a file named `env.local`
   - Add the following environment variables with your Qdrant database API details:

> Qdrant Vector Database Configuration

QDRANT_URL=https://<your-qdrant-url>.aws.cloud.qdrant.io:6333
QDRANT_API_KEY="your-qdrant-api-key"
QDRANT_COLLECTION="collection-name"

> Google Gemini API Configuration

GEMINI_API_KEY="your-gemini-api-key"

> Optional: Model Configuration

MODEL_NAME=gemini-2.0-flash
EMBEDDING_MODEL=dangvantuan/vietnamese-embedding
LLM_MODEL=gemini-2.0-flash
RERANK_MODEL=nlpHUST/vielectra-base-discriminator

- Replace the keys with your actual credentials
- Note: All Qdrant Keys need to be specific to the database needed

2. **Install dependencies** 
- Run `pip install -r requirements.txt` to install any required Python packages.

## Running the Application

1. Open a terminal in the project directory.
2. Run the following command:

python app.py

3. Once the application starts, a link will be displayed in the terminal (e.g., `  http://127.0.0.1:7860`).
4. Click the link or copy it into your browser to access the application.

## Notes

- Ensure your Qdrant database and Google Gemini API are properly configured and accessible before running the app.
- The application uses the `gemini-2.0-flash` model for language processing, `dangvantuan/vietnamese-embedding` for embeddings, and `nlpHUST/vielectra-base-discriminator` for reranking.

## Troubleshooting

- **"ModuleNotFoundError"**: Ensure all dependencies are installed (`pip install -r requirements.txt`).
- **Connection errors**: Double-check your Qdrant API credentials and URLs in `env.local`.
- **API key issues**: Verify that your API keys are valid and have the necessary permissions.
