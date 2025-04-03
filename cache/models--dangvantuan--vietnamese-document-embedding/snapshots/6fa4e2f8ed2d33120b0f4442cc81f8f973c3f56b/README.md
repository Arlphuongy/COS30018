---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- phobert
- vietnamese
- sentence-embedding
license: apache-2.0
language:
- vi
metrics:
- pearsonr
- spearmanr
---
## Model Description:
[**vietnamese-document-embedding**](https://huggingface.co/dangvantuan/vietnamese-document-embedding) is the Document Embedding Model for Vietnamese language  with context length up to 8096 tokens. This model is a specialized long text-embedding trained specifically for the Vietnamese language, which is built upon [gte-multilingual](Alibaba-NLP/gte-multilingual-base) and trained using the Multi-Negative Ranking Loss, Matryoshka2dLoss and SimilarityLoss.

## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False}) with Transformer model: VietnameseModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```
## Training and Fine-tuning process
The model underwent a rigorous four-stage training and fine-tuning process, each tailored to enhance its ability to generate precise and contextually relevant sentence embeddings for the Vietnamese language. Below is an outline of these stages:
#### Stage 1: Training NLI on dataset XNLI: 
- Dataset: [XNLI-vn ](https://huggingface.co/datasets/xnli/viewer/vi)
- Method: Training using Multi-Negative Ranking Loss and Matryoshka2dLoss. This stage focused on improving the model's ability to discern and rank nuanced differences in sentence semantics.
### Stage 2: Fine-tuning for Semantic Textual Similarity on STS Benchmark
- Dataset: [STSB-vn](https://huggingface.co/datasets/doanhieung/vi-stsbenchmark)
- Method: Fine-tuning specifically for the semantic textual similarity benchmark using Siamese BERT-Networks configured with the 'sentence-transformers' library. This stage honed the model's precision in capturing semantic similarity across various types of Vietnamese texts.


## Usage:

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["Hà Nội là thủ đô của Việt Nam", "Đà Nẵng là thành phố du lịch"]


model = SentenceTransformer('dangvantuan/vietnamese-document-embedding', trust_remote_code=True)
embeddings = model.encode(sentences)
print(embeddings)

```


## Evaluation
The model can be evaluated as follows on the [Vienamese data of stsb](https://huggingface.co/datasets/doanhieung/vi-stsbenchmark).

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample
from datasets import load_dataset
def convert_dataset(dataset):
    dataset_samples=[]
    for df in dataset:
        score = float(df['score'])/5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[df['sentence1'], df['sentence2']], label=score)
        dataset_samples.append(inp_example)
    return dataset_samples

# Loading the dataset for evaluation
vi_sts = load_dataset("doanhieung/vi-stsbenchmark")["train"]
df_dev = vi_sts.filter(lambda example: example['split'] == 'dev')
df_test = vi_sts.filter(lambda example: example['split'] == 'test')

# Convert the dataset for evaluation

# For Dev set:
dev_samples = convert_dataset(df_dev)
val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
val_evaluator(model, output_path="./")

# For Test set:
test_samples = convert_dataset(df_test)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path="./")
```




### Metric for all dataset of [Semantic Textual Similarity on STS Benchmark](https://huggingface.co/datasets/anti-ai/ViSTS)

**Spearman score**
| Model                                                                                                               | [STSB]   | [STS12]| [STS13] | [STS14] | [STS15] |    [STS16] | [SICK] | Mean |
|-----------------------------------------------------------|---------|----------|----------|----------|----------|----------|---------|--------|
| [dangvantuan/vietnamese-embedding](https://huggingface.co/dangvantuan/vietnamese-embedding)                                                 |84.84|	79.04|	85.30|	81.38|	87.06|	79.95|	79.58|	82.45|
| [dangvantuan/vietnamese-embedding-LongContext](https://huggingface.co/dangvantuan/vietnamese-embedding-LongContext)  |85.25|	75.77|	83.82|	81.69|	88.48|	81.5|	78.2|	82.10|

## Citation


	@article{reimers2019sentence,
	   title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
	   author={Nils Reimers, Iryna Gurevych},
	   journal={https://arxiv.org/abs/1908.10084},
	   year={2019}
	}


    @article{zhang2024mgte,
      title={mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval},
      author={Zhang, Xin and Zhang, Yanzhao and Long, Dingkun and Xie, Wen and Dai, Ziqi and Tang, Jialong and Lin, Huan and Yang, Baosong and Xie, Pengjun and Huang, Fei and others},
      journal={arXiv preprint arXiv:2407.19669},
      year={2024}
    }
    
    @article{li2023towards,
      title={Towards general text embeddings with multi-stage contrastive learning},
      author={Li, Zehan and Zhang, Xin and Zhang, Yanzhao and Long, Dingkun and Xie, Pengjun and Zhang, Meishan},
      journal={arXiv preprint arXiv:2308.03281},
      year={2023}
    }
    
    @article{li20242d,
      title={2d matryoshka sentence embeddings},
      author={Li, Xianming and Li, Zongxi and Li, Jing and Xie, Haoran and Li, Qing},
      journal={arXiv preprint arXiv:2402.14776},
      year={2024}
    }