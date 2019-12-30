import random
import torch
from torch import nn
from torch.nn import functional as F
from .rnncells import StackedLSTMCell, StackedGRUCell
from .beam_search import Beam
from .feedforward import FeedForward
from utils import to_var, SOS_ID, UNK_ID, EOS_ID
import math
import pdb
from queue import PriorityQueue
import operator
import numpy



class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class BaseRNNDecoder(nn.Module):
    def __init__(self):
        """Base Decoder Class"""
        super(BaseRNNDecoder, self).__init__()

    @property
    def use_lstm(self):
        return isinstance(self.rnncell, StackedLSTMCell)

    def init_token(self, batch_size, SOS_ID=SOS_ID):
        """Get Variable of <SOS> Index (batch_size)"""
        x = to_var(torch.LongTensor([SOS_ID] * batch_size))
        return x

    def init_h(self, batch_size=None, zero=True, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            # (h, c)
            return (to_var(torch.zeros(self.num_layers,
                                       batch_size,
                                       self.hidden_size)),
                    to_var(torch.zeros(self.num_layers,
                                       batch_size,
                                       self.hidden_size)))
        else:
            # h
            return to_var(torch.zeros(self.num_layers,
                                      batch_size,
                                      self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTMCell)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def decode(self, out):
        """
        Args:
            out: unnormalized word distribution [batch_size, vocab_size]
        Return:
            x: word_index [batch_size]
        """

        # Sample next word from multinomial word distribution
        if self.sample:
            # x: [batch_size] - word index (next input)
            x = torch.multinomial(self.softmax(out / self.temperature), 1).view(-1)

        # Greedy sampling
        else:
            # x: [batch_size] - word index (next input)
            _, x = out.max(dim=1)
        return x

    def forward(self):
        """Base forward function to inherit"""
        raise NotImplementedError

    def forward_step(self):
        """Run RNN single step"""
        raise NotImplementedError

    def embed(self, x):
        """word index: [batch_size] => word vectors: [batch_size, hidden_size]"""

        if self.training and self.word_drop > 0.0:
            if random.random() < self.word_drop:
                embed = self.embedding(to_var(x.data.new([UNK_ID] * x.size(0))))
            else:
                embed = self.embedding(x)
        else:
            embed = self.embedding(x)

        return embed
'''
    def beam_decode(self,
                    init_h=None,
                    encoder_outputs=None, input_valid_length=None,
                    decode=False):
        """
        Args:
            encoder_outputs (Variable, FloatTensor): [batch_size, source_length, hidden_size]
            input_valid_length (Variable, LongTensor): [batch_size] (optional)
            init_h (variable, FloatTensor): [batch_size, hidden_size] (optional)
        Return:
            out   : [batch_size, seq_len]
        """
        batch_size = self.batch_size(h=init_h)

        # [batch_size x beam_size]
        x = self.init_token(batch_size * self.beam_size, SOS_ID)

        # [num_layers, batch_size x beam_size, hidden_size]
        h = self.init_h(batch_size, hidden=init_h).repeat(1, self.beam_size, 1)

        # batch_position [batch_size]
        #   [0, beam_size, beam_size * 2, .., beam_size * (batch_size-1)]
        #   Points where batch starts in [batch_size x beam_size] tensors
        #   Ex. position_idx[5]: when 5-th batch starts
        batch_position = to_var(torch.arange(0, batch_size).long() * self.beam_size)

        # Initialize scores of sequence
        # [batch_size x beam_size]
        # Ex. batch_size: 5, beam_size: 3
        # [0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf]
        score = torch.ones(batch_size * self.beam_size) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size).long() * self.beam_size, 0.0)
        score = to_var(score)

        # Initialize Beam that stores decisions for backtracking
        beam = Beam(
            batch_size,
            self.hidden_size,
            self.vocab_size,
            self.beam_size,
            self.max_unroll,
            batch_position)

        for i in range(self.max_unroll):

            # x: [batch_size x beam_size]; (token index)
            # =>
            # out: [batch_size x beam_size, vocab_size]
            # h: [num_layers, batch_size x beam_size, hidden_size]
            out, h = self.forward_step(x, h,
                                       encoder_outputs=encoder_outputs,
                                       input_valid_length=input_valid_length)
            # log_prob: [batch_size x beam_size, vocab_size]
            log_prob = F.log_softmax(out, dim=1)

            # [batch_size x beam_size]
            # => [batch_size x beam_size, vocab_size]
            score = score.view(-1, 1) + log_prob

            # Select `beam size` transitions out of `vocab size` combinations

            # [batch_size x beam_size, vocab_size]
            # => [batch_size, beam_size x vocab_size]
            # Cutoff and retain candidates with top-k scores
            # score: [batch_size, beam_size]
            # top_k_idx: [batch_size, beam_size]
            #       each element of top_k_idx [0 ~ beam x vocab)

            score, top_k_idx = score.view(batch_size, -1).topk(self.beam_size, dim=1)


            # Get token ids with remainder after dividing by top_k_idx
            # Each element is among [0, vocab_size)
            # Ex. Index of token 3 in beam 4
            # (4 * vocab size) + 3 => 3
            # x: [batch_size x beam_size]
            x = (top_k_idx % self.vocab_size).view(-1)

            # top-k-pointer [batch_size x beam_size]
            #       Points top-k beam that scored best at current step
            #       Later used as back-pointer at backtracking
            #       Each element is beam index: 0 ~ beam_size
            #                     + position index: 0 ~ beam_size x (batch_size-1)
            beam_idx = top_k_idx / self.vocab_size  # [batch_size, beam_size]
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)


            # Select next h (size doesn't change)
            # [num_layers, batch_size * beam_size, hidden_size]
            h = h.index_select(1, top_k_pointer)

            # Update sequence scores at beam
            beam.update(score.clone(), top_k_pointer, x)  # , h)

            # Erase scores for EOS so that they are not expanded
            # [batch_size, beam_size]
            eos_idx = x.data.eq(EOS_ID).view(batch_size, self.beam_size)
            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        # prediction ([batch, k, max_unroll])
        #     A list of Tensors containing predicted sequence
        # final_score [batch, k]
        #     A list containing the final scores for all top-k sequences
        # length [batch, k]
        #     A list specifying the length of each sequence in the top-k candidates
        # prediction, final_score, length = beam.backtrack()
        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length

'''
class DecoderRNN(BaseRNNDecoder):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, rnncell=StackedGRUCell, num_layers=1,
                 dropout=0.0, word_drop=0.0,
                 max_unroll=30, sample=True, temperature=1.0, beam_size=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.word_drop = word_drop
        self.max_unroll = max_unroll
        self.sample = sample
        self.beam_size = beam_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.rnncell = rnncell(num_layers,
                               embedding_size,
                               hidden_size,
                               dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward_step(self, x, h,
                     encoder_outputs=None,
                     input_valid_length=None):
        """
        Single RNN Step
        1. Input Embedding (vocab_size => hidden_size)
        2. RNN Step (hidden_size => hidden_size)
        3. Output Projection (hidden_size => vocab size)

        Args:
            x: [batch_size]
            h: [num_layers, batch_size, hidden_size] (h and c from all layers)

        Return:
            out: [batch_size,vocab_size] (Unnormalized word distribution)
            h: [num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        # x: [batch_size] => [batch_size, hidden_size]
        x = self.embed(x)
        # last_h: [batch_size, hidden_size] (h from Top RNN layer)
        # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
        last_h, h = self.rnncell(x, h)

        if self.use_lstm:
            # last_h_c: [2, batch_size, hidden_size] (h from Top RNN layer)
            # h_c: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
            last_h = last_h[0]

        # Unormalized word distribution
        # out: [batch_size, vocab_size]
        out = self.out(last_h)
        return out, h

    def forward(self, inputs, init_h=None, encoder_outputs=None, input_valid_length=None,
                decode=False, turn = None):
        """
        Train (decode=False)
            Args:
                inputs (Variable, LongTensor): [batch_size, seq_len]
                init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
            Return:
                out   : [batch_size, seq_len, vocab_size]
        Test (decode=True)
            Args:
                inputs: None
                init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
            Return:
                out   : [batch_size, seq_len]
        """
        batch_size = self.batch_size(inputs, init_h)

        # x: [batch_size]
        x = self.init_token(batch_size, SOS_ID)

        # h: [num_layers, batch_size, hidden_size]
        h = self.init_h(batch_size, hidden=init_h)


        if not decode:
            out_list = []
            seq_len = inputs.size(2)
            for i in range(seq_len):

                # x: [batch_size]
                # =>
                # out: [batch_size, vocab_size]
                # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
                out, h = self.forward_step(x, h)

                out_list.append(out)
                x = inputs[:, turn, i]

            # [batch_size, max_target_len, vocab_size]
            return torch.stack(out_list, dim=1)

        elif decode == 'F1':
            x_list = []
            for i in range(self.max_unroll):
                # x: [batch_size]
                # =>
                # out: [batch_size, vocab_size]
                # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
                out, h = self.decode_in_beam(x, h)
                log_prob, indexes = torch.topk(out, 1)
                decoded_t = torch.transpose(indexes, 0, 1)[0]
                # out: [batch_size, vocab_size]
                # => x: [batch_size]
                x_list.append(decoded_t)
                x = inputs[:, turn, i]
            return torch.stack(x_list, dim=1)

        elif decode == 'beam':
            x_list = []
            for i in range(self.max_unroll):
                # x: [batch_size]
                # =>
                # out: [batch_size, vocab_size]
                # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
                out, h = self.decode_in_beam(x, h)
                log_prob, indexes = torch.topk(out, 1)
                decoded_t = torch.transpose(indexes, 0, 1)[0]
                # out: [batch_size, vocab_size]
                # => x: [batch_size]
                x_list.append(decoded_t)
                x = inputs[:, turn, i]
            return torch.stack(x_list, dim=1)

        else:
            x_list = []
            for i in range(self.max_unroll):

                # x: [batch_size]
                # =>
                # out: [batch_size, vocab_size]
                # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
                out, h = self.forward_step(x, h)

                # out: [batch_size, vocab_size]
                # => x: [batch_size]
                x = self.decode(out)
                x_list.append(x)

            # [batch_size, max_target_len]
            return torch.stack(x_list, dim=1)

    def decode_in_beam(self, x, h, encoder_outputs=None):


        # x: [batch_size]
        # =>
        # out: [batch_size, vocab_size]
        # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
        out, h = self.forward_step(x, h)
        out = F.log_softmax(out, dim=1)

        return out, h

    def beam_decode(self, inputs, decoder_hiddens=None, turn=None, encoder_outputs=None):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        beam_width = self.beam_size
        topk = self.beam_size  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        seq_len = inputs.size(0)
        for idx in range(seq_len):
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
            '''
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)
            '''

            # Start with the start of the sentence token
            decoder_input = to_var(torch.LongTensor([[SOS_ID]]))

            # Number of sentence to generate
            endnodes = []
            number_required = topk

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1
            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 999999: break
                nextnodes = []
                # fetch the best node
                for _ in range(nodes.qsize()):
                    score, n = nodes.get()
                    decoder_input = to_var(n.wordid[0])

                    decoder_hidden = n.h

                    if (n.wordid.item() == EOS_ID and n.prevNode != None) or n.leng >= self.max_unroll:
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break

                    # decode for one step using decoder
                    decoder_output, decoder_hidden = self.decode_in_beam(decoder_input, decoder_hidden) #, encoder_output)

                    # PUT HERE REAL BEAM SEARCH OF TOP
                    log_prob, indexes = torch.topk(decoder_output, beam_width)
                    #nextnodes = []
                    for new_k in range(beam_width):
                        decoded_t = indexes[0][new_k].view(1, -1)
                        log_p = log_prob[0][new_k].item()

                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval()
                        nextnodes.append((score, node))

                if len(endnodes) >= number_required:
                    break
                nextnodes = sorted(nextnodes, key=operator.itemgetter(0), reverse=True)
                length = min(len(nextnodes), beam_width)
                for i in range(length):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize

                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            '''
            if len(endnodes) <= beam_width:
                for _ in range(beam_width - len(endnodes)):
                    score, n = nodes.get()
                    endnodes.append((score, node))
            '''

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.cpu().numpy()[0][0])
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.cpu().numpy()[0][0])

                utterance = utterance[::-1]
                utterances.append(utterance)

            final_utterance = utterances[0]
            if len(utterances[0]) < self.max_unroll:
                final_utterance += [3 for _ in range(self.max_unroll-len(utterances[0]))]
            decoded_batch.append(final_utterance)

        return decoded_batch






