import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import Chroma
from langchain import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
os.environ['GOOGLE_API_KEY']=os.getenv('google_api_key')

HF_api_key=os.getenv('hugging_face_key')
os.environ['HUGGINGFACEHUB_API_TOKEN']=HF_api_key
HF_llm = HuggingFaceHub(repo_id='google/flan-t5-large', model_kwargs={'temperature': 0, 'max_length': 64})

# imports
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


pdf_reader=PyPDFLoader('langchain_document.pdf')
doc=pdf_reader.load()


embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db=Chroma.from_documents(doc,embedding_function)
retriver_type=db.as_retriever(type='similarity',search_kwargs={'k':1})

chain = ConversationalRetrievalChain.from_llm(HF_llm, retriver_type)
chat_history=[]
next_qs=True
while next_qs:
    query=input('user: ')
    result=chain({'question':query,'chat_history':chat_history})
    print(f"AI:{result['answer']}")
    chat_history=[(query,result['answer'])]
    next_qs=input('do you have next qs (yes or no): ').lower()
    if next_qs=='yes':
        next_qs=True
    else:next_qs=False
