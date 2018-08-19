import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class MyResNet(ResNet):
    def __init__(self):
        super(MyResNet,self).__init__(BasicBlock, [3, 4, 6, 3], 1000)
        self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))



    def forward(self, x):   # bs * 3 * 640 * 640
        x = self.conv1(x)   # bs * 64 * 320 * 320
        x = self.bn1(x)     # bs * 64 * 320 * 320
        x = self.relu(x)    # bs * 64 * 320 * 640
        x = self.maxpool(x) # bs * 64 * 160 * 160

        x = self.layer1(x)  # bs * 64 * 160 * 160
        x = self.layer2(x)  # bs * 128 * 80 * 80
        # x = self.layer3(x)  # bs * 256 * 40 * 40
        # x = self.layer4(x)  # bs * 512 * 20 * 20

        # x = self.avgpool(x) # bs * 512 * 14 * 14
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

class DepthResNet(nn.Module):
    def __init__(self):
        super(DepthResNet,self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = nn.Conv2d(1, 1, kernel_size=1,stride=1, bias=False)
        self.layer2 = nn.Conv2d(1, 1, kernel_size=1,stride=2, bias=False)


        nn.init.constant(self.conv1.weight, 1.0 / 49.0)
        # nn.init.constant(self.layer1.weight, 1.0 / 49.0)
        nn.init.constant(self.layer2.weight, 1.0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer2(x)

        return x

class RelevanceMap(nn.Module):
    # the rest of the network
    # SM: Semantic Map
    # WE: Word Embedding
    # f_depth is the feature depth i.e. channel numbers for the two maps
    # remember to wrap all the torch tensor to Variables
    # won't change the dim of sm
    def __init__(self, wordEmbeddingNum,wordEmbeddingDim, featureChannel=128, mapChannel=128):
        super(RelevanceMap, self).__init__()
        # w_out_dim is the feature size for each kernel (depth into the channel)
        self.wordEmbeddingDim = wordEmbeddingDim
        self.embedding = nn.Embedding(num_embeddings=wordEmbeddingNum, embedding_dim=wordEmbeddingDim)
        self.featureChannel = featureChannel
        self.mapChannel = mapChannel
        self.linearLayer1 = torch.nn.Linear(wordEmbeddingDim, mapChannel * featureChannel * 1 * 1, bias=True)
        self.leakyRelu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.average_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc_size = 16*44*44

    def forward(self, featureTopView, word):
        batchSize = featureTopView.size(0)
        assert featureTopView.size(0) == word.size(0)
        H = featureTopView.size(2)
        W = featureTopView.size(3)
        wordEmbedding = self.embedding(word)
        filter1 = self.leakyRelu(self.linearLayer1(wordEmbedding))
        filter1 = filter1.view(batchSize, self.mapChannel, self.featureChannel, 1, 1)

        relevanceMap = None # keep the spatial size
        for i in range(0, batchSize):
            _feature = featureTopView[i].view(1, self.featureChannel, H, W)
            _relevanceMap = F.relu(F.conv2d(_feature, filter1[i]))
            _relevanceMap = _relevanceMap / torch.max(_relevanceMap)

            if relevanceMap is None:
                relevanceMap = _relevanceMap
            else:
                relevanceMap = torch.cat((relevanceMap,_relevanceMap),dim=0)

        relevanceMap = relevanceMap.view(batchSize, self.mapChannel, H * W)
        singleRelevanceMap,idx = torch.max(relevanceMap,dim=1)
        singleRelevanceMap = singleRelevanceMap.view(batchSize, H, W)

        return singleRelevanceMap

if __name__ == '__main__':

    feature = Variable(torch.rand(4,128,80,80))
    wordEm = Variable(torch.rand(4,256))

    Rele = RelevanceMap(wordEmbeddingDim=256, featureChannel=128, mapChannel=128)
    r = Rele(feature, wordEm)

    print r.size()
