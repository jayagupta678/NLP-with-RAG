Key = ""
import os
os.environ["OPENAI_API_KEY"] = key

from openai import OpenAI
client = OpenAI()

#%% Loading Document
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('C:\Users\Jaya Gupta\Downloads\thesis_6.pdf')
docs = loader.load()

#%% Chunking PDF document

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
# documents[:2]

#%%Convert chunks to vector embedding 
# Try Langs vector data base

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(documents, OpenAIEmbeddings())


from langchain_community.llms import Ollama
llm = Ollama(model='llama3.2')
llm


#%% Designing chat prompt templet
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
                                            refer the provided context to prepare test plan with following by using the same techincal words mentioned in the context: 
                                            1. Epics: Highest-level goal of a project, providing direction and context to allow teams to effectively plan out the testing process.
                                            - There should be 1 Epic to start, the user may request more, that consists of the following information: Summary 4-8 sentences, Out of Scope as a bullet point list, Assumptions as a bullet point list, Dependency's as a bullet point list, Risks. 
                                            2. Features: A decomposition of the plan split into specific areas of work based on the requirements. Features are big enough to be releases or sprints. 
                                            - There should be 6-15 features, where each feature is defined at least in a one liner. The user may request a specific number of features or range of numbers. 
                                            A feature should consist of a Description 3-5 sentences As a ..., I want..., So that..., Acceptance Criteria with 3-7 bullet points defining the scope of feature, Scenarios including three bullet points of comprised of Given..., when..., then... .
                                            <context>
                                            {context}
                                            </context>
                                            Question: {input}""") 



#%% Chain Introduction, Chain stuff Documents Chain

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)

#%% Retriver
retriever = db.as_retriever()
retriever

#%% Create retreveral chain
from langchain.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)
#%%
response = retrieval_chain.invoke({"input":"please provide the detailed test plan from the content provided and output to be given in the mentioned format in prompt"})
print(response['answer'])
 