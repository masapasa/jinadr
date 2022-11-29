
from docarray import DocumentArray
papers = DocumentArray.from_csv("/home/aswin/documents/papers-search/backend/src/data/papers.csv",size=1111, field_resolver={"source_id": "id"})
papers.summary()

for paper in papers:
    paper.text = paper.tags["title"] + "[SEP]" + paper.tags["abstract"]
papers.summary()

from jina import Flow
flow = Flow.load_config("flow.yml")
indexer = flow()
with indexer:
    indexer.index(papers, request_size=32, show_progress=True)
