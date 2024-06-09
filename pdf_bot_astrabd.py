import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain,LLMChain
#from langchain.vectorstores.cassandra import Cassandra
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain import HuggingFaceHub
from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader
load_dotenv()

HF_api_key=os.getenv('hugging_face_key')
os.environ['HUGGINGFACEHUB_API_TOKEN']=HF_api_key
HF_llm = HuggingFaceHub(repo_id='google/flan-t5-large', model_kwargs={'temperature': 0, 'max_length': 64})


from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
astra_db_token='AstraCS:YEkIbXWfFGWXiCUqUWseeMIk:ee7837ce70208dd578ec5ead4ccb4b22d5cc2ae2566d8a34323705d4e788a22d'
astra_db_id='391b0345-2678-4bb4-a328-f98055fec76c'

pdf_reader=PdfReader('langchain_document.pdf')

from typing_extensions import Concatenate
raw_text=''
for i,page in enumerate(pdf_reader.pages):
    content=page.extract_text()
    if content:
        raw_text+=content
connection=True
while connection:
    try:
        print('connecting ...')
        cassio.init(token=astra_db_token,database_id=astra_db_id)
        connection=False
    except:
        print('connection failed and trying again')
print('The connection established with astra')
astra_vector_store=Cassandra(
    embedding=embedding_function,
     table_name='pdf_query',
    session=None,
    keyspace=None,
 )

print('astra vector store initialised')
from langchain.text_splitter import CharacterTextSplitter
text_splitter=CharacterTextSplitter(
    separator='\n',
    chunk_size=500,
    chunk_overlap=200,
    length_function=len
)

texts=text_splitter.split_text(raw_text)
print(f'The sample split text: \n {texts[:10]}')

astra_vector_store.add_texts(texts[:50])
print('inserted %i lines' %len(texts[:50]))

astra_vector_index=VectorStoreIndexWrapper(vectorestore=astra_vector_store)
print('Index of vector embedding wrapped')

query=input('enter your query: ')
answer=astra_vector_index.query(query,HF_llm)
print(f'The answer: {answer}')