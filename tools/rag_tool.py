def retrieve_documents(query: str):

    docs = vectorstore.similarity_search(
        query,
        k=5
    )

    return docs