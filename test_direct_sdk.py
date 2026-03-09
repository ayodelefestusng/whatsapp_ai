import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def test_direct():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY found")
        return
    
    genai.configure(api_key=api_key)
    
    print("Testing direct embedding...")
    try:
        # Use a model name known to work with the Python SDK
        # The SDK often uses different names than LangChain
        model = 'models/embedding-001'
        result = genai.embed_content(
            model=model,
            content="What is the meaning of life?",
            task_type="retrieval_document",
            title="Embedding test"
        )
        print("Success! Embedding length:", len(result['embedding']))
    except Exception as e:
        print(f"Direct embedding failed: {e}")

    print("Testing direct generation...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say hi")
        print(f"Success! Response: {response.text}")
    except Exception as e:
        print(f"Direct generation failed: {e}")

if __name__ == "__main__":
    test_direct()
