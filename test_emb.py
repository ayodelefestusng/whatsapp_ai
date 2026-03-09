import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def test_embeddings(model_name):
    print(f"Testing model: {model_name}")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
        vector = embeddings.embed_query("hello world")
        print(f"Success! Vector length: {len(vector)}")
        return True
    except Exception as e:
        print(f"Failed with {model_name}: {e}")
        return False

if __name__ == "__main__":
    if not test_embeddings("models/embedding-001"):
        test_embeddings("models/gemini-embedding-001")
