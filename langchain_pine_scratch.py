import pinecone
from langchain.chains import LLMChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.llms.openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from os.path import exists
from langchain.vectorstores import Pinecone
import os

load_dotenv()

def create_vectordb(path: str):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    index_name = "relevantamb"

    # First, check if our index already exists. If it doesn't, we create it


    if index_name not in pinecone.list_indexes():
        # If the index does not exist, create a new one

        # load the document and split it into chunks
        loader = PyPDFDirectoryLoader(path)
        documents = loader.load_and_split()

        # split it into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # load it into Chroma
        pinecone.create_index(name=index_name, metric="cosine", dimension= 384)
        db = Pinecone.from_documents(docs, embedding_function, index_name=index_name)
        return db
    else:
        db = docsearch = Pinecone.from_existing_index(index_name, embedding_function)
        return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k)
    docs_page_content = " ".join(d.page_content for d in docs)
    llm = OpenAI(model='text-davinci-003', temperature=0)



    prompt = PromptTemplate(
        input_variables=['question', 'docs'],
        template='''
        You are a helpful Assistant that can answer questions about text based on the text script.
        Answer the following question in at least 300 words: {question}
        By searching the following text script: {docs}
        Only use the factual information from the transcript to answer the question. 
        If you feel like you don't have enough information to answer the question, say 'I don't know'.
        Your answer should be detailed.  
        '''
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace('\n', '')
    return response


