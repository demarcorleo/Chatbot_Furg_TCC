# =====================================================
# Criador de Banco Vetorial Chroma (PDF + TXT)
# =====================================================

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def carregar_documentos(pasta: str):
    docs = []
    if not os.path.exists(pasta):
        raise FileNotFoundError(f"❌ Pasta '{pasta}' não encontrada.")

    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)

        if arquivo.lower().endswith(".pdf"):
            print(f"📘 Carregando PDF: {arquivo}")
            loader = PyPDFLoader(caminho)
            docs.extend(loader.load())

        elif arquivo.lower().endswith(".txt"):
            print(f"📄 Carregando TXT: {arquivo}")
            loader = TextLoader(caminho, encoding="utf-8")
            docs.extend(loader.load())

    print(f"\n✅ Total de documentos carregados: {len(docs)} páginas")
    return docs


def criar_banco():
    pasta_docs = "Documentos"
    pasta_banco = "./chroma_db"

    print(f"\n🚀 Iniciando criação do banco vetorial a partir da pasta '{pasta_docs}'")

    documents = carregar_documentos(pasta_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=pasta_banco
    )

    print(f"💾 Banco vetorial criado e salvo em: {pasta_banco}")


if __name__ == "__main__":
    criar_banco()
