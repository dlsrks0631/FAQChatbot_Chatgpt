from flask import Flask, render_template, request
from flask_socketio import SocketIO
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
import tiktoken
import os

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

loader = PyPDFLoader("DATA_PATH")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=tiktoken_len)
texts = text_splitter.split_documents(pages)

model_name ="jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

persist_directory = "./chroma_db"
if os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=hf)
else:
    vectordb = Chroma.from_documents(texts, hf, persist_directory=persist_directory)

openai = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0.1,
)

system_message = SystemMessage(content="당신은 이름은 '전북대학교 LMS 도우미'입니다. 당신이 일은 전북대학교 LMS 홈페이지 사용방법에 대한 질문을 답변하는 일이야. 당신은 사람의 LMS 사용방법 관련 지식이 풍부하고 모든 질문에 대해서 명확히 답변해 줄 수 있습니다.")
aimessage = AIMessage(content="나는 전북대학교 LMS 도우미이다.")
openai.invoke([system_message, aimessage])


QA_CHAIN_PROMPT_TEMPLATE = """당신은 이름은 '전북대학교 LMS 도우미'입니다. \
    당신이 하는 일은 전북대학교 LMS 홈페이지 사용방법에 대한 질문에 대해 구체적이고 친절하게 답변하는 일이야. \
    당신은 전북대학교 LMS 홈페이지에 대한 관련 지식이 풍부하고 LMS에 관련된 모든 질문에 대해서 거짓을 답할 수 없고 관련된 모든 질문에 대해서는 명확하게 답할 수 있습니다. \
    사용자가 질문하지 않을 경우에는 자세한 질문을 요구해도돼.
{context}
질문: {question}
질문에 대한 답변: """

QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_CHAIN_PROMPT_TEMPLATE)

qa = RetrievalQA.from_chain_type(
    llm=openai,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'fetch_k': 2}),
    return_source_documents=True,
)

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'fetch_k': 2})

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=openai)

qa.retriever = retriever_from_llm

def chatbot(query):
    result = qa(query)
    response = result['result']
    return response

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api', methods=['POST'])
def api():
    question = request.json.get("message")

    response = chatbot(question)
    print(f"Response from Model: {response}")

    if response is not None:
        return {'content': response}
    else:
        return 'Failed to Generate response!'

if __name__ == '__main__':
    app.run(port=5000, debug=True)
