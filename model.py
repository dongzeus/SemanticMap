import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

End_of_layer = -5
Num_1_filter = 32
Num_3_filter = 32
kernel1 = 1
kernel3 = 3
F_depth1 = 32
F_depth3 = 32

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
        return h0.cuda(), c0.cuda()

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
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c_t


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=2, downsample=None):
        super(Bottleneck, self).__init__()
        expansion = 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if self.stride != 1 or inplanes != planes * expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ImageFeature(nn.Module):
    # model to extract image feature
    def __init__(self):
        super(ImageFeature, self).__init__()
        pre_model = models.resnet101(pretrained=True)
        self.my_image_model = nn.Sequential(*list(pre_model.children())[:End_of_layer])
        self.whole_resnet = nn.Sequential(*list(pre_model.children())[:-1])
        self.conv1 = nn.Conv2d(256, 32, kernel_size=1)
        self.average_pool1 = nn.AvgPool2d((7, 7), stride=(2, 2), padding=(3, 3))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.average_pool2 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, images, depth_info):
        batch_size = images.size(0)
        for param in self.my_image_model.parameters():
            param.requires_grad = False
        for param in self.whole_resnet.parameters():
            param.requires_grad = False    
        assert batch_size == depth_info.size(0)
        feature_map = self.my_image_model(images)
        feature_map = self.conv1(feature_map)
        depth_info = self.average_pool1(depth_info)  # change it to mean select
        depth_info = self.max_pool(depth_info)
        for _ in range(3):
            depth_info = self.average_pool2(depth_info)
        res_feature = self.whole_resnet(images).squeeze()
        return feature_map, depth_info, res_feature

# class Projection and merge semantic map():


class AllMaps(nn.Module):
    # the rest of the network
    # SM: Semantic Map
    # WE: Word Embedding
    # f_depth is the feature depth i.e. channel numbers for the two maps
    # remember to wrap all the torch tensor to Variables
    # won't change the dim of sm
    def __init__(self, we_in_dim, f_depth1, f_depth3):
        super(AllMaps, self).__init__()
        # w_out_dim is the feature size for each kernel (depth into the channel)
        self.we_in_dim = we_in_dim
        self.f_depth1 = f_depth1
        self.f_depth3 = f_depth3
        self.linear_layer1 = torch.nn.Linear(we_in_dim, Num_1_filter * f_depth1 * kernel1 * kernel1, bias=True)
        self.linear_layer3 = torch.nn.Linear(we_in_dim, Num_3_filter * f_depth3 * kernel3 * kernel3, bias=True)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.res_block = Bottleneck(32, 4)
        self.average_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc_size = 16*44*44
        self.linear_layer_last1 = torch.nn.Linear(self.fc_size, 2048)
        self.linear_layer_last3 = torch.nn.Linear(self.fc_size, 2048)
    def forward(self, sm_in, we_in):
        batch_size = we_in.size(0)
        assert sm_in.size(1) == self.f_depth1, "sm_in channel= %d, f_depth1=%d" %(sm_in.size(1),self.f_depth1)
        assert batch_size == sm_in.size(0)
        H = sm_in.size(2)
        W = sm_in.size(3)
        filter1 = self.leakyrelu(self.linear_layer1(we_in))
        filter1 = filter1.view(batch_size, Num_1_filter, self.f_depth1, kernel1, kernel1)
        rjw = Variable(torch.empty(batch_size, Num_1_filter, H, W),requires_grad=True)  # keep the spatial size
        for i in range(0, batch_size):
            sm_in_tmp = sm_in[i].unsqueeze(0)
            rjw[i] = F.conv2d(sm_in_tmp, filter1[i])
        # lost relevant map projection!!
        assert rjw.size(1) == self.f_depth3
        filter3 = self.leakyrelu(self.linear_layer3(we_in))
        filter3 = filter3.view(batch_size, Num_3_filter, self.f_depth3, kernel3, kernel3)
        gjr = Variable(torch.empty(batch_size, Num_3_filter, H, W),requires_grad=True)
        for i in range(0, batch_size):
            rjw_tmp = rjw[i].unsqueeze(0)   # supposed to be rjr
            # may need extra 3*3 conv
            gjr[i] = F.conv2d(rjw_tmp, filter3[i], dilation=3, padding=3)
        rjw = self.average_pool(self.res_block(rjw))
        gjr = self.average_pool(self.res_block(gjr))
        rjw = rjw.view(batch_size, -1)
        gjr = gjr.view(batch_size, -1)
        rjw = self.linear_layer_last1(rjw)
        gjr = self.linear_layer_last3(gjr)
        return rjw, gjr

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn

class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                      dropout_ratio, feature_size=2048*3):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)

    def forward(self, we, images, dept_images, sm_in, action, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).
        we: original word embedding: last output of h from encoder
        images: the raw rgb image input
        dept_images: input of depth images
        sm_in: the semantic map
        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)   # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        img_feature = ImageFeature()
        f_m, d_i, r_f = img_feature(images,dept_images)
        # use feature map to project and get the semantic map
        allmap = AllMaps(self.hidden_size, F_depth1, F_depth3)
        rjw, gjr = allmap(sm_in, we)
        concat_input = torch.cat((action_embeds, r_f, rjw, gjr), 1) # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1, c_1, alpha, logit, sm_in

