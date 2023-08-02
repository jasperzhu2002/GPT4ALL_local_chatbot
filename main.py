import sys
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf import loadPDFs
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain

files = ["review"]
db = loadPDFs(files)

llm = GPT4All(
    model="./1ggml-gpt4all-j-v1.3-groovy.bin",
    backend="gptj",
    verbose=False
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    db.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
)

chat_history = []
while True:
    query = input('Prompt: ')
    if query == "quit":
        sys.exit()
    result = qa({'question': query})
    print('Answer: ' + result['answer'])
    
    # chat_history.append((query, result['answer']))




# import sys
# from langchain.chains import RetrievalQA

# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import GPT4All
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from pdf import loadPDFs
# from langchain.memory import ConversationBufferMemory
# from langchain import PromptTemplate, LLMChain

# files = ["review"]
# db = loadPDFs(files)

# llm = GPT4All(
#     model="./1ggml-model-gpt4all-falcon-q4_0.bin",
#     backend="gptj",
#     verbose=False
# )

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer. 
# Use three sentences maximum and keep the answer as concise as possible. 
# Always say "thanks for asking!" at the end of the answer. 
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # qa = ConversationalRetrievalChain.from_llm(
# qa = RetrievalQA.from_chain_type(
#     llm,
#     retriever=db.as_retriever(search_kwargs={"k": 3}),
#     # memory=memory,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
# )

# chat_history = []
# while True:
#     query = input('Prompt: ')
#     if query == "quit":
#         sys.exit()
#     result = qa({'query': query})
#     print('Answer: ' + result['result'])
    
#     # chat_history.append((query, result['answer']))