# =====================================================
# Chatbot RAG Local com LangChain + llama.cpp (limitando contexto)
# =====================================================

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings  # pacote atualizado
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
import sys
import textwrap  # ✅ para formatar a saída no terminal

# 1. Carregar documento PDF
loader = PyPDFLoader("relatorio.pdf")
documents = loader.load()

# 2. Criar embeddings e base vetorial Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ⚠️ Apague a pasta ./chroma_db se trocar o modelo de embeddings
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# 3. Configurar o mecanismo de recuperação (reduzido para 1 documento)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 4. Template do prompt revisado
template = """[INST]
Você é um assistente técnico especializado em relatórios de estágio.
Com base exclusivamente no contexto abaixo, descreva detalhadamente as atividades desenvolvidas,
utilizando frases completas e linguagem técnica formal.

Se a pergunta for sobre as atividades do estágio, utilize as seções 3.1 a 3.7 do contexto.
Se não houver informação suficiente, diga apenas "Não sei com base no documento.".

=== CONTEXTO ===
{context}
=== FIM DO CONTEXTO ===

Pergunta: {question}
Responda de forma estruturada e completa, listando cada atividade de forma clara.
[/INST]"""



prompt = PromptTemplate.from_template(template)

# 5. Carregar o modelo LlamaCpp
llm = LlamaCpp(
    model_path = "/home/demarco/Área de trabalho/TCC/models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,         # tamanho máximo do contexto
    n_threads=8,        # número de threads CPU
    n_gpu_layers=40,    # mais camadas na GPU (aproveita melhor a RTX 3060)
    max_tokens=1024,    # permite respostas mais longas
    streaming=False,     # mostra tokens conforme gerados
    temperature=0.7,    # deixa as respostas mais elaboradas
    top_p=0.9,
    verbose=True
)


# 6. Montar a cadeia RAG (limitando o tamanho do contexto)
rag_chain = (
    RunnableMap({
        "context": lambda x: "\n".join(
    [doc.page_content[:1500] for doc in retriever.invoke(x["question"])],  # ✅ limite de 2000 caracteres
        ),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Loop de chat interativo
print("Chatbot RAG Local (LangChain + llama.cpp)")
print("Digite 'sair' para encerrar a conversa.\n")

while True:
    try:
        pergunta = input("Você: ")
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("IA: Até logo!")
            break

        resposta = rag_chain.invoke({"question": pergunta})
        resposta_limpa = " ".join(resposta.split())

        # ✅ Formatar saída para melhor leitura
        resposta_formatada = textwrap.fill(resposta_limpa, width=120)

        print("\n🧠 Resposta:\n")
        print(resposta_formatada)
        print("\n" + "=" * 80 + "\n")
        sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nIA: Encerrando execução.")
        break
