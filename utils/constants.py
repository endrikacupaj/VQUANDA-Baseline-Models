"""Constants for the baseline models"""
SEED = 42
QUESTION = 'question'
QUERY = 'query'

RNN_NAME = 'rnn'
CNN_NAME = 'cnn'
TRANSFORMER_NAME = 'trasnformer'

ATTENTION_1 = 'bahdanau'
ATTENTION_2 = 'luong'

GPU = 'gpu'
CPU = 'cpu'
CUDA = 'cuda'

TRAIN_PATH = '/data/train.json'
TEST_PATH = '/data/test.json'
CHECKPOINT_PATH = '/model/'

ANSWER_TOKEN = '<ans>'
ENTITY_TOKEN = '<ent>'
EOS_TOKEN = '<eos>'
SOS_TOKEN = '<sos>'
PAD_TOKEN = '<pad>'

SRC_NAME = 'src'
TRG_NAME = 'trg'
