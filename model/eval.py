from solver import Solver, VariationalSolver
from data_loader import get_loader
from configs import get_config
from utils import Vocab, Tokenizer
import os
import pickle
from models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    test_freq_config = get_config(mode='test_freq')
    test_rare_config = get_config(mode='test_rare')

    print('Loading freq Vocabulary...')
    vocab_freq = Vocab()
    vocab_freq.load(test_freq_config.word2id_path, test_freq_config.id2word_path)
    vocab_rare = Vocab()
    vocab_rare.load(test_rare_config.word2id_path, test_rare_config.id2word_path)
    print(f'freq Vocabulary size: {vocab_freq.vocab_size}')
    print(f'rare Vocabulary size: {vocab_rare.vocab_size}')

    test_freq_config.vocab_size = vocab_freq.vocab_size
    test_rare_config.vocab_size = vocab_rare.vocab_size

    freq_data_loader = get_loader(
        sentences=load_pickle(test_freq_config.sentences_path),
        conversation_length=load_pickle(test_freq_config.conversation_length_path),
        sentence_length=load_pickle(test_freq_config.sentence_length_path),
        vocab=vocab_freq,
        batch_size=test_freq_config.batch_size,
        shuffle=False)

    rare_data_loader = get_loader(
        sentences=load_pickle(test_rare_config.sentences_path),
        conversation_length=load_pickle(test_rare_config.conversation_length_path),
        sentence_length=load_pickle(test_rare_config.sentence_length_path),
        vocab=vocab_rare,
        batch_size=test_rare_config.batch_size,
        shuffle=False)

    if test_freq_config.model in VariationalModels:
        solver_freq = VariationalSolver(test_freq_config, None, freq_data_loader, rare_data_loader, vocab=vocab_freq, is_train=False)
        solver_freq.build()
        solver_freq.importance_sample()
    else:
        solver_freq = Solver(test_freq_config, None, freq_data_loader, rare_data_loader, vocab=vocab_freq, is_train=False)
        solver_freq.build()
        freq_perplexity, freq_f1 = solver_freq.test()

    print('Loading rare Vocabulary...')

    vocab_freq = Vocab()
    vocab_freq.load(test_freq_config.word2id_path, test_freq_config.id2word_path)
    vocab_rare = Vocab()
    vocab_rare.load(test_rare_config.word2id_path, test_rare_config.id2word_path)
    print(f'freq Vocabulary size: {vocab_freq.vocab_size}')
    print(f'rare Vocabulary size: {vocab_rare.vocab_size}')

    test_freq_config.vocab_size = vocab_freq.vocab_size
    test_rare_config.vocab_size = vocab_rare.vocab_size

    freq_data_loader = get_loader(
        sentences=load_pickle(test_freq_config.sentences_path),
        conversation_length=load_pickle(test_freq_config.conversation_length_path),
        sentence_length=load_pickle(test_freq_config.sentence_length_path),
        vocab=vocab_freq,
        batch_size=test_freq_config.batch_size,
        shuffle=False)

    rare_data_loader = get_loader(
        sentences=load_pickle(test_rare_config.sentences_path),
        conversation_length=load_pickle(test_rare_config.conversation_length_path),
        sentence_length=load_pickle(test_rare_config.sentence_length_path),
        vocab=vocab_rare,
        batch_size=test_rare_config.batch_size,
        shuffle=False)

    if test_rare_config.model in VariationalModels:
        solver_rare = VariationalSolver(test_rare_config, None, rare_data_loader, rare_data_loader, vocab=vocab_rare, is_train=False)
        solver_rare.build()
        solver_rare.importance_sample()
    else:
        solver_rare = Solver(test_rare_config, None, rare_data_loader, rare_data_loader, vocab=vocab_rare, is_train=False)
        solver_rare.build()
        rare_perplexity, rare_f1 = solver_rare.test()


    print_str_freq = f'freq Word perplexity : {freq_perplexity:.3f}\n'
    print_str_freq_1 = f'freq Unigram F1 : {freq_f1:.3f}\n'
    print_str_rare = f'rare Word perplexity : {rare_perplexity:.3f}\n'
    print_str_rare_1 = f'rare Unigram F1 : {rare_f1:.3f}\n'
    print(print_str_freq)
    print(print_str_freq_1)
    print(print_str_rare)
    print(print_str_rare_1)
