import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def test_chat():
    print("Testing Chat model...")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        response = llm.invoke("Hi")
        print(f"Chat Success! Response: {response.content}")
        return True
    except Exception as e:
        print(f"Chat Failed: {e}")
        return False

def test_embeddings():
    print("Testing Embedding model...")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector = embeddings.embed_query("hello world")
        print(f"Embeddings Success! Vector length: {len(vector)}")
        return True
    except Exception as e:
        print(f"Embeddings Failed: {e}")
        return False

if __name__ == "__main__":
    test_chat()
    test_embeddings()
