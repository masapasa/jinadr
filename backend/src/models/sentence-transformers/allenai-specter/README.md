---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
license: apache-2.0
---

# allenai-specter

This model is a conversion of the [AllenAI SPECTER](https://github.com/allenai/specter) model to [sentence-transformers](https://www.SBERT.net). It can be used to map the titles & abstracts of scientific publications to a vector space such that similar papers are close.


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/allenai-specter')
embeddings = model.encode(sentences)
print(embeddings)
```



## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch


def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/allenai-specter')
model = AutoModel.from_pretrained('sentence-transformers/allenai-specter')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```



## Evaluation Results



For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name=sentence-transformers/allenai-specter)



## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

## Citing & Authors

See  [AllenAI SPECTER](https://github.com/allenai/specter)