# Preprocess topical movie dialogs dataset

from multiprocessing import Pool
import argparse
import pickle
import random
import os
from urllib.request import urlretrieve
from zipfile import ZipFile
from pathlib import Path
from tqdm import tqdm
from model.utils import Tokenizer, Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
import pdb
import json
import re
from tqdm.auto import tqdm
import sys
from collections import OrderedDict

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('datasets/')
topical_conv_dir = datasets_dir.joinpath('topical_chat/conversations/')

# Tokenizer
tokenizer = Tokenizer('spacy')

def loadConversations(fileName):
    """
    Args:
        fileName (str): file to load
        field (set<str>): fields to extract
    Return:
        dict<dict<str>>: the extracted fields for each line
    """
    conversations = []

    with open(fileName, 'r') as f:
        conv_set = json.load(f)
        conv_id = []
        # get the id for each conversation
        for key in conv_set.keys():
            conv_id.append(key)
        #extract conversations:
        for id in conv_id:
            conversations.append(conv_set[id])
    return conversations




def tokenize_conversation(lines):
    sentence_list = [tokenizer(line['message']) for line in lines]
    return sentence_list


def pad_sentences(conversations, max_sentence_length, max_conversation_length):
    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation = conversation[:max_conversation_length]
        sentence_length = [min(len(sentence) + 1, max_sentence_length) # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=30)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    # Split Ratio
    split_ratio = [0.8, 0.1, 0.1]

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=20000)
    parser.add_argument('--min_vocab_frequency', type=int, default=5)

    # Multiprocess
    parser.add_argument('--n_workers', type=int, default=os.cpu_count())

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency
    n_workers = args.n_workers

    # Download and extract dialogs if necessary.

    print("Loading conversations...")
    train = loadConversations(topical_conv_dir.joinpath("train.json"))
    valid_freq = loadConversations(topical_conv_dir.joinpath("valid_freq.json"))
    valid_rare = loadConversations(topical_conv_dir.joinpath("valid_rare.json"))
    test_freq = loadConversations(topical_conv_dir.joinpath("test_freq.json"))
    test_rare = loadConversations(topical_conv_dir.joinpath("test_rare.json"))
    print('Number of training conversations:', len(train))



    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    for split_type, conv_objects in [('train', train), ('valid_freq', valid_freq), ('valid_rare', valid_rare), ('test_freq', test_freq),
                                     ('test_rare', test_rare)]:
        print(f'Processing {split_type} dataset...')
        split_data_dir = topical_conv_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)

        print(f'Tokenize.. (n_workers={n_workers})')
        def _tokenize_conversation(conv):
            return tokenize_conversation(conv['content'])

        with Pool(n_workers) as pool:
            conversations = list(tqdm(pool.imap(_tokenize_conversation, conv_objects),
                                     total=len(conv_objects)))

        conversation_length = [min(len(conv['content']), max_conv_len)
                               for conv in conv_objects]

        sentences, sentence_length = pad_sentences(
            conversations,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len)

        print('max conversation turns:', max(conversation_length))
        print('max_sentence_length:', max(flat(sentence_length)))
        print('Saving preprocessed data at', split_data_dir)
        to_pickle(conversation_length, split_data_dir.joinpath('conversation_length.pkl'))
        to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
        to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))

        if split_type == 'train':

            print('Save Vocabulary...')
            vocab = Vocab(tokenizer)
            vocab.add_dataframe(conversations)
            vocab.update(max_size=max_vocab_size, min_freq=min_freq)

            print('Vocabulary size: ', len(vocab))
            vocab.pickle(topical_conv_dir.joinpath('word2id.pkl'), topical_conv_dir.joinpath('id2word.pkl'))

    print('Done!')
