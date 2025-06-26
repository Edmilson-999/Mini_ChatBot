from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import os
import time

# 1. Configurar o modelo Mistral-7B
llm = LlamaCpp(
    model_path="./model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Atualize o caminho
    n_ctx=2048,
    max_tokens=500,
    temperature=0.7,
    n_gpu_layers=0,
    verbose=True
)

# 2. Carregar e dividir documentos
def load_documents(data_path="data/nutricao"):
    documents = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = load_documents()
chunks = text_splitter.split_documents(documents)

# 3. Gerar embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# 4. Criar vetor store com FAISS
vector_store = FAISS.from_documents(chunks, embeddings)

# 5. Configurar o pipeline RAG
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Interface do Chatbot
def chatbot():
    print("Bem-vindo ao Chatbot de Nutrição! Digite sua pergunta ou 'sair' para encerrar.")
    while True:
        query = input("Pergunta: ")
        if query.lower() == "sair":
            print("Encerrando o chatbot...")
            llm.close()  # Libera o modelo explicitamente
            break
        try:
            start_time = time.time()
            response = qa_chain({"query": query})
            end_time = time.time()
            print("\nResposta:", response["result"])
            print("Fontes:", [doc.metadata for doc in response["source_documents"]])
            print(f"Tempo de resposta: {end_time - start_time:.2f} segundos\n")
            print("\n")
        except Exception as e:
            print(f"Erro: {e}")

# Executar o chatbot
if __name__ == "__main__":
    try:
        chatbot()
    except Exception as e:
        print(f"Erro ao encerrar: {e}")
    finally:
        try:
            llm.close()
        except:
            pass  # Ignora erros na liberação