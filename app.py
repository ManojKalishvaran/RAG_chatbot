import os 
from openai import OpenAI 
import chromadb 
from dotenv import load_dotenv
from chromadb.utils import embedding_functions 
from chromadb.utils.embedding_functions import EmbeddingFunction
from pypdf import PdfReader
import google.generativeai as genai

gemini_api = "AIzaSyBN1AlkuB9W2EU6rcfJffrKumCzsnrdh4o"

#""""Generator -- LLM - gemini-1.5-flash""""
genai.configure(api_key=gemini_api)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("hi")
print(response.text)


#""""Embedding model """"
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, model = 'models/text-embedding-004'):
        super().__init__()
        self.api_key = api_key
        self.model = model 
        genai.configure(api_key=self.api_key)
    def __call__(self, text):
        embeddings = []
        for txt in text:
            response = genai.embed_content(model=self.model, content=txt)
            embeddings.append(response['embedding'])
        return embeddings
# genai.configure(api_key="AIzaSyBN1AlkuB9W2EU6rcfJffrKumCzsnrdh4o")
# result = genai.embed_content(
#         model="models/text-embedding-004",
#         content="What is the meaning of life?")

# print(str(result['embedding']))

embedding_fn = GeminiEmbeddingFunction(api_key=gemini_api)

### Gemini client 
class GeminiClient: 
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.chat = self.Chat()
    
    class Chat: 
        def __init__(self):
            self.completions = self.Completions()

        class Completions:
            def create(self, model, messages):
                #inititalizing gemini model
                gemini_model = genai.GenerativeModel(model)
                #message formatiing (gemini uses different messages than openai)
                formatted_messages = []
                for message in messages:
                    if message["role"] == "system":
                        formatted_messages.append(f"System: {message['content']}")  
                    elif message["role"] == "user":
                        formatted_messages.append(f"User: {message['content']}")
                    elif message["role"] == "assistant":
                        formatted_messages.append(f"Assistant: {message['content']}")

                response = gemini_model.generate_content('\n'.join(formatted_messages))
                return {'choices':[{'message':{"content":response.text}}]}

#""""Initialize chromaDB """"
chroma_client = chromadb.PersistentClient(path="chroma_client_path")
collection_name = "document_qa_rag_1"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)

#"""Setting Generator UP"""
client = GeminiClient(api_key=gemini_api)
res = client.chat.completions.create(model="gemini-1.5-flash", 
                               messages=[{'role':"system", "content":"you are a helper"}, 
                                         {'role':"user", "content":"what is lifeexpetancy in US?"}])

print("response :", res["choices"][0]["message"]['content'])


#""""Getting document ready""""
def load_document(file_path):
    print("**********", os.path.basename(file_path),"**********")
    print('===Loading Document===')
    document_sentences = []
    if file_path.endswith(".pdf"):
        readed_pdf = PdfReader(file_path)
        page_id = 0
        for page in readed_pdf.pages:
            document_sentences.append({os.path.basename(file_path)[:-4]+f'_page_{page_id}':page.extract_text().split('.')})
            page_id += 1

    elif file_path.endswith(".txt"):
        with open(file_path) as file:
            document_sentences.append({os.path.basename(file_path)[:-4]:file.read().split('.')})
    return document_sentences


