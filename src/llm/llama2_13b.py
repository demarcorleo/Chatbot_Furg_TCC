# =====================================================
# Chatbot RAG Local (usando banco já criado)
# =====================================================

import os
import sys
import textwrap

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser

# =====================================================
# 1. Carregar banco vetorial existente
# =====================================================

print("💾 Carregando banco vetorial existente...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

print("✅ Banco vetorial carregado com sucesso!\n")

# =====================================================
# 2. Template de prompt
# =====================================================

template = """[INST]
Você é um assistente técnico e deve responder **sempre em português do Brasil**, mesmo que o conteúdo do contexto
ou da pergunta esteja em outro idioma. Responda de forma técnica, clara e estruturada, usando terminologia adequada
a documentos acadêmicos ou de engenharia.

Se não houver informação suficiente, responda apenas com: "Não sei com base nos documentos."

=== CONTEXTO ===
{context}
=== FIM DO CONTEXTO ===

Pergunta: {question}

Responda sempre em português.
[/INST]"""


prompt = PromptTemplate.from_template(template)

# =====================================================
# 3. Configurar o modelo LlamaCpp
# =====================================================

llm = LlamaCpp(
    model_path="/home/demarco/Área de trabalho/TCC/models/llama-2-13b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=35,
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    verbose=False
)

# =====================================================
# 4. Montar a cadeia RAG
# =====================================================

rag_chain = (
    RunnableMap({
        "context": lambda x: "\n".join(
            [doc.page_content[:1500] for doc in retriever.invoke(x["question"])]
        ),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================================
# 5. Chat interativo
# =====================================================

print("🤖 Chatbot RAG Local (LangChain + llama.cpp)")
print("📂 Usando banco vetorial existente: ./chroma_db")
print("Digite 'sair' para encerrar.\n")

while True:
    try:
        pergunta = input("Você: ")
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("IA: Até logo!")
            break

        resposta = rag_chain.invoke({"question": pergunta})
        resposta_limpa = " ".join(resposta.split())
        resposta_formatada = textwrap.fill(resposta_limpa, width=120)

        print("\n🧠 Resposta:\n")
        print(resposta_formatada)
        print("\n" + "=" * 100 + "\n")
        sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nIA: Encerrando execução.")
        break
