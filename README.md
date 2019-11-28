# Baseline models for Verbalization Dataset

## Dataset Repository

- https://github.com/endrikacupaj/QA-Verbalization-Dataset

## Introduction
The repository contains sequence to sequence models for evaluating Verbalization dataset.

The models we use are:

- RNN Seq2seq based:
    -  [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
    - [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- CNN Seq2seq based:
    - [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
- Transformer Seq2seq based:
    - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Our implementations have minor differences with the models described on the papers.

## Experiments
RNN-1 with Bahdanau attention:
``` bash
# with entities
python run.py --model rnn --attention bahdanau
# cover entities
python run.py --model rnn --attention bahdanau --cover_entities
```
RNN-2 with Luong attention:
``` bash
# with entities
python run.py --model rnn --attention luong
# cover entities
python run.py --model rnn --attention luong --cover_entities
```
CNN:
``` bash
# with entities
python run.py --model cnn
# cover entities
python run.py --model cnn --cover_entities
```
Transformer:
``` bash
# with entities
python run.py --model transformer
# cover entities
python run.py --model transformer --cover_entities
```

## Results


## License
The repository is under [MIT lisence](LICENSE).

## References

Links we considered for implementing the baseline models:
- https://github.com/pytorch/fairseq
- https://github.com/IBM/pytorch-seq2seq
- https://github.com/bentrevett/pytorch-seq2seq
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://pytorch.org/tutorials/