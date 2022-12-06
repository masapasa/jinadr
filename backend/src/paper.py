from docarray import DocumentArray, Document
from flows import index_flow, search_flow
papers = DocumentArray.from_csv("/home/aswin/Documents/jinadr/backend/src/data/papers.csv", field_resolver={"source_id": "id"})
indexer = index_flow()
with indexer:
    indexer.index(papers, request_size=32)