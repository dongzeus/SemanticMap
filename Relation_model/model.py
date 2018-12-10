# my relation model
# input bs*N*1024 feature vector
from math import sqrt
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')


class FCOutputModel(nn.Module):
    def __init__(self, outFeatureChannel):
        super(FCOutputModel, self).__init__()

        self.fc3 = nn.Linear(outFeatureChannel, 4)

    def forward(self, x):
        x = F.dropout(x)
        x = self.fc3(x)
        return x


class RN(nn.Module):
    def __init__(self, batch_size, num_objects, vocab_size, embedding_size=256, hidden_size=512, padding_idx = padding_idx,
                 dropout_ratio = 0.5, featureChannel=5, outFeatureChannel=256, dropout= True):
        super(RN, self).__init__()
        if torch.cuda.is_available():
            cuda_flag = True
        else:
            cuda_flag = False

        self.final_feature_size = (featureChannel)*2 + hidden_size
        self.outFeatureChannel = outFeatureChannel
        self.num_objects = num_objects
        self.featureChannel = featureChannel

        self.embedding = EncoderLSTM(vocab_size, embedding_size, hidden_size, padding_idx,
                                     dropout_ratio, bidirectional=False, num_layers=1)
        #self.conv = FasterRCNN()
        

        self.g_fc1 = nn.Linear(((featureChannel)*2 + hidden_size), outFeatureChannel)

        self.f_fc1 = nn.Linear(outFeatureChannel, outFeatureChannel)


        self.fcout = FCOutputModel(outFeatureChannel)



    def forward(self, input_feature, input_sent, input_sentlen):
        # hopefully the input feature is bs * N * 5
        # hopefully the input sentence size is bs * vocabsize
        bs = input_feature.size(0)
        assert bs == input_sent.size(0), "The basesize of input feature != input sentence"
        N = input_feature.size(1) # the number of region proposals
        assert N==self.num_objects, "The number of region proposals are not equal"
        f_size = input_feature.size(2) # the feature size of input feature
        assert f_size == self.featureChannel, 'feature channel not equal!'
        instrucEmbed = self.embedding(input_sent,input_sentlen) # bs * 512


        f_flat = input_feature
        
        instrucEmbed = torch.unsqueeze(instrucEmbed,1)  # bs * 1 * 512
        instrucEmbed = instrucEmbed.repeat(1, N, 1) #bs * N *512
        instrucEmbed = torch.unsqueeze(instrucEmbed, 2) # bs * N * 1 * 512

        #cast al pairs against each other
        f_i = torch.unsqueeze(f_flat, 1)  # bs * 1 * N * 5
        f_i = f_i.repeat(1,N,1,1)   # bs x N x N x 5
        f_j = torch.unsqueeze(f_flat, 2) # bs x N x 1 x 5
        f_j = torch.cat([f_j,instrucEmbed],3)   # bs x N x 1 x 517
        f_j = f_j.repeat(1,1,N,1) # bs x N x N x 517

        f_full = torch.cat([f_i, f_j],3) # bs x N x N x 522

        #reshape for passing the network
        f_ = f_full.view(bs*N*N,self.final_feature_size)
        f_ = self.g_fc1(f_)
        f_ = F.relu(f_)

        #reshape again and sum
        f_g = f_.view(bs, N*N, self.outFeatureChannel)
        f_g = f_g.sum(1).squeeze()

        f_f = self.f_fc1(f_g)
        f_f = F.relu(f_f)

        return self.fcout(f_f)   

        
class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions
                                         )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        if torch.cuda.is_available():
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]

        embedding = nn.Tanh()(self.encoder2decoder(h_t))

        return embedding


if __name__ == '__main__':
    model = RN(batch_size=2, num_objects=7, vocab_size=100).cuda()
    #model = EncoderLSTM(vocab_size=100, embedding_size=256, hidden_size=512, padding_idx=padding_idx,
     #            dropout_ratio=0.5).cuda()
    #input_feature, input_sent, input_sentlen
    batch_size=2
    num_objects=7
    vocab_size=100
    input_feature = torch.randn(batch_size,num_objects, 7).cuda()
    input_sent = torch.LongTensor([[1,2],[3,4]]).cuda()
    input_length = [2,2]
    y = model(input_feature,input_sent, input_length)
    print y.shape
