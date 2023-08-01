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
    model="./1ggml-model-gpt4all-falcon-q4_0.bin",
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