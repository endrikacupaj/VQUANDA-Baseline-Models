"""Run baseline model"""
import os
import math
import random
import argparse
from pathlib import Path
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator

from models.seq2seq import Seq2Seq
from models.cnn import EncoderConv, DecoderConv
from utils.checkpoint import Chechpoint
from trainer.trainer import Trainer
from utils.scorer import BleuScorer
from evaluator.predictor import Predictor
from utils.verbaldataset import VerbalDataset
from utils.constants import (
    SEED, CUDA, CPU, PAD_TOKEN, RNN_NAME, CNN_NAME,
    TRANSFORMER_NAME, ATTENTION_1, ATTENTION_2, QUESTION, QUERY
)

ROOT_PATH = Path(os.path.dirname(__file__))
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

def parse_args():
    """Add arguments to parser"""
    parser = argparse.ArgumentParser(description='Verbalization dataset baseline models.')
    parser.add_argument('--model', default=TRANSFORMER_NAME, type=str,
                        choices=[RNN_NAME, CNN_NAME, TRANSFORMER_NAME], help='model to train the dataset')
    parser.add_argument('--input', default=QUESTION, type=str,
                        choices=[QUESTION, QUERY], help='use query as input')
    parser.add_argument('--attention', default=ATTENTION_1, type=str,
                        choices=[ATTENTION_1, ATTENTION_2], help='attention layer for rnn model')
    parser.add_argument('--cover_entities', action='store_true', help='cover entities')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--epochs_num', default=50, type=int, help='number of epochs')
    args = parser.parse_args()
    return args


def main():
    """Main method to run the models"""
    args = parse_args()
    dataset = VerbalDataset(ROOT_PATH)
    query_as_input = True if args.input == QUERY else False
    dataset.load_data_and_fields(cover_entities=args.cover_entities, query_as_input=query_as_input)
    src_vocab, trg_vocab = dataset.get_vocabs()
    train_data, valid_data, test_data = dataset.get_data()

    print('--------------------------------')
    print(f'Model: {args.model}')
    print(f'Model input: {args.input}')
    if args.model == RNN_NAME:
        print(f'Attention: {args.attention}')
    print(f'Cover entities: {args.cover_entities}')
    print('--------------------------------')
    print(f"Training data: {len(train_data.examples)}")
    print(f"Evaluation data: {len(valid_data.examples)}")
    print(f"Testing data: {len(test_data.examples)}")
    print('--------------------------------')
    print(f'Question example: {train_data.examples[0].src}')
    print(f'Answer example: {train_data.examples[0].trg}')
    print('--------------------------------')
    print(f"Unique tokens in questions vocabulary: {len(src_vocab)}")
    print(f"Unique tokens in answers vocabulary: {len(trg_vocab)}")
    print('--------------------------------')
    print(f'Batch: {args.batch_size}')
    print(f'Epochs: {args.epochs_num}')
    print('--------------------------------')

    if args.model == RNN_NAME and args.attention == ATTENTION_1:
        from models.rnn1 import EncoderRNN, DecoderRNN
    elif args.model == RNN_NAME and args.attention == ATTENTION_2:
        from models.rnn2 import EncoderRNN, DecoderRNN

    # create model
    encoder = EncoderRNN(src_vocab, DEVICE) if args.model == RNN_NAME else \
              EncoderConv(src_vocab, DEVICE)
    decoder = DecoderRNN(trg_vocab, DEVICE) if args.model == RNN_NAME else \
              DecoderConv(trg_vocab, DEVICE)
    model = Seq2Seq(encoder, decoder, args.model).to(DEVICE)

    parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters_num:,} trainable parameters')
    print('--------------------------------')

    # create optimizer and criterion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi[PAD_TOKEN])

    # train data
    trainer = Trainer(optimizer, criterion, args.batch_size, DEVICE)
    trainer.train(model, train_data, valid_data, num_of_epochs=args.epochs_num)

    # load model
    model = Chechpoint.load(model)

    # generate test iterator
    test_iterator = BucketIterator(test_data,
                                   batch_size=args.batch_size,
                                   sort_within_batch=True if args.model == RNN_NAME else False,
                                   sort_key=lambda x: len(x.src),
                                   device=DEVICE)

    # evaluate model
    test_loss = trainer.evaluator.evaluate(model, test_iterator)

    # calculate blue score for test data
    scorer = BleuScorer()
    predictor = Predictor(model, src_vocab, trg_vocab, DEVICE)
    scorer.data_score(test_data.examples, predictor)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    print(f'| Average BLEU score {scorer.average_score()} |')


if __name__ == "__main__":
    # set a seed value
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    main()
