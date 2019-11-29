# Baseline models for Verbalization Dataset

## Dataset Repository

- https://github.com/endrikacupaj/QA-Verbalization-Dataset

## Introduction
The repository contains sequence to sequence models for evaluating Verbalization dataset.

The models we use are based on:

- RNN Seq2seq:
    -  [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
    - [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- CNN Seq2seq:
    - [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
- Transformer Seq2seq:
    - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Experiments
### Requirements and Setup
Python version >= 3.7
``` bash
# first download the dataset
git clone https://github.com/endrikacupaj/QA-Verbalization-Dataset.git
cd QA-Verbalization-Dataset
# inside the dataset download the baseline models
git clone https://github.com/endrikacupaj/Verbalization-Dataset-Baseline-Models.git
cd Verbalization-Dataset-Baseline-Models
pip install -r requirements.txt
```

### Run models
RNN-1 with Bahdanau attention:
``` bash
python run.py --model rnn --attention bahdanau
# with covered entities
python run.py --model rnn --attention bahdanau --cover_entities
# query as input
python run.py --model rnn --attention bahdanau --input query
# query as input with covered entities
python run.py --model rnn --attention bahdanau --input query --cover_entities
```
RNN-2 with Luong attention:
``` bash
python run.py --model rnn --attention luong
# with covered entities
python run.py --model rnn --attention luong --cover_entities
# query as input
python run.py --model rnn --attention luong --input query
# query as input with covered entities
python run.py --model rnn --attention luong --input query --cover_entities
```
CNN:
``` bash
python run.py --model cnn
# with covered entities
python run.py --model cnn --cover_entities
# query as input
python run.py --model cnn --input query
# query as input with covered entities
python run.py --model cnn --input query --cover_entities
```
Transformer:
``` bash
python run.py --model transformer
# with covered entities
python run.py --model transformer --cover_entities
# query as input
python run.py --model transformer --input query
# query as input with covered entities
python run.py --model transformer --input query --cover_entities
```

## Results


## License
The repository is under [MIT License](LICENSE).

## References

Links we considered for implementing the baseline models:
- https://github.com/pytorch/fairseq
- https://github.com/IBM/pytorch-seq2seq
- https://github.com/bentrevett/pytorch-seq2seq
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://pytorch.org/tutorials/