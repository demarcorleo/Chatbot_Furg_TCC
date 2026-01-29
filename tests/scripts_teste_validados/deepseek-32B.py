from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser

# 1. Carregar documento
loader = PyPDFLoader("relatorio.pdf")
documents = loader.load()

# 2. Embeddings + Chroma
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# 3. Retriever atualizado
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 4. Prompt com contexto
template = """Você é uma IA especializada em responder com base em documentos.
Use o contexto abaixo para responder à pergunta. Seja claro e completo.
Se não souber, diga que não sabe.

Contexto:
{context}

Usuário: {question}
IA:"""

prompt = PromptTemplate.from_template(template)

# 5. LLM atualizado
llm = OllamaLLM(model="gemma3:27b")

# 6. Chain com método atualizado (invoke)
rag_chain = (
    RunnableMap({
        "context": lambda x: "\n".join(
            [doc.page_content for doc in retriever.invoke(x["question"])]
        ),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Chat
print("Chatbot RAG Local (LangChain + Ollama)")
print("Digite 'sair' para encerrar a conversa")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() in {"sair", "exit", "quit"}:
        print("IA: Até logo!")
        break

    resposta = rag_chain.invoke({"question": pergunta})
    print(f"IA: {resposta}\n")
