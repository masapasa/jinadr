from docarray import DocumentArray, Document
from jina import Flow, DocumentArray, Executor, requests
from config import DATA_FILE, NUM_DOCS, PORT
import click

#********** modification 04 08 2022 *******************

import boto3


import json
import configparser

def readConfig():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    return config['Data-S3']

config = readConfig()

#********************************************************

# ***************** Executor to filter out the Documents without embedding *********
#EMB_DIM = 512

class EmbeddingChecker(Executor):
    @requests
    def check(self, docs, **kwargs):
        filtered_docs = DocumentArray()
        for doc in docs:
            if doc.embedding is None:
                continue
            #if doc.embedding.shape[0] != EMB_DIM:
            #    continue
            filtered_docs.append(doc)
        return filtered_docs
    
    

#***********************************************************************************


flow = (
    Flow(protocol="http", port=PORT)
    .add(
        name="encoder",
        uses="jinahub://TransformerTorchEncoder",
        #uses_with={
         #   "pretrained_model_name_or_path": "sentence-transformers/paraphrase-mpnet-base-v2"}
        uses_with={'pretrained_model_name_or_path': 'bert-base-uncased'}
        ,
        install_requirements=True,
    )
    .add(name = "Filterout_docs_without_embeddings", 
             uses = EmbeddingChecker)
    .add(name = "Indexer",uses="jinahub://SimpleIndexer", needs='Filterout_docs_without_embeddings'
         ,  install_requirements=True)
    .needs_all()
)
#flow.plot()


def index(num_docs=NUM_DOCS):
    """qa_docs = DocumentArray.from_csv(
        DATA_FILE, field_resolver={"question": "text"}, size=num_docs
    )"""
    
    # ****************** modification 04 08 2022 '******************
    s3 = boto3.resource('s3')
    bucket_name = config['bucket_name']
    bucket = s3.Bucket(bucket_name)
    docs = DocumentArray()


    # Iterates through all the objects, doing the pagination for you. Each obj
    # is an ObjectSummary, so it doesn't contain the body. You'll need to call
    # get to get the whole body.
    for obj in bucket.objects.all():
        key = obj.key
        body = obj.get()['Body'].read()
        #print(key)

        if key.endswith('.json') :
            #print(type(body))
            #print(body)
            data = body.decode('utf-8') # Decode using the utf-8 encoding
            jdata = json.loads(data)
            #print(jdata)
            #print(type(jdata))
        #print('**********************')
            for item in jdata['body']:
                docs.append(Document(text = item['text']
                       ))
    print(docs.summary)
    #***************************************************************

        
    #qa_docs = 
    with flow:
        #docs = flow.index(docs, on_done = store_embeddings ,show_progress=True)
        flow.index(docs,show_progress=True)
        print(type(docs))
        #print(docs.summary)


def search_grpc(string: str):
    doc = Document(text=string)
    with flow:
        results = flow.search(doc)

    print(results[0].matches)

    for match in results[0].matches:
        print(match.text)



def search():
    #with flow:
     #   flow.block()
    question = Document(text="how much is the training budget?")

    with flow:
        results = flow.search(question)

    print(results[0].matches[0].tags["answer"])

@click.command()
@click.option(
    "--task",
    "-t",
    type=click.Choice(["index", "search"], case_sensitive=False),
)
@click.option("--num_docs", "-n", default=NUM_DOCS)
def main(task: str, num_docs):
    if task == "index":
        index(num_docs=num_docs)
    elif task == "search":
        search()
    else:
        print("Please add '-t index' or '-t search' to your command")


if __name__ == "__main__":
    main()