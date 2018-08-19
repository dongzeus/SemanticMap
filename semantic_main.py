import os

import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from semantic_model import MyResNet, DepthResNet, RelevanceMap
from semantic_utils import buildVocaulary, generateTopView
from semantic_dataset import DatasetGenerator


class SemanticMapTask():
    def __init__(self):
        self.batchSize = 1
        self.lr = 1e-4
        self.weightDecay = 1e-3
        self.wordEmbeddingDim = 256
        self.pinMemory = False

        self.lossAll = []
        self.lossAllEpoch = []
        self.lossInEpoch = []


        self.objectCategoryPath = './data/category_mapping.tsv'
        self.regionCategoryPath = './data/region_category.txt'
        self.trainSplitPath = './dataset_split/train_scans.txt'
        self.testSplitPath = './dataset_split/test_scans.txt'
        self.trainSplitPath = './dataset_split/tmp.txt'
        self.testSplitPath = './dataset_split/tmp.txt'
        self.imageRootPath = '/Volumes/Dongqiyuan/matterport_dataset'
        self.logRootPath = './log'
        self.logPath = os.path.join(self.logRootPath, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        if not os.path.exists(self.logRootPath):
            os.mkdir(self.logRootPath)
        os.mkdir(self.logPath)

        self.vocabulary, self.vocabularyInv = buildVocaulary(self.regionCategoryPath, self.objectCategoryPath)
        self.trainDataset = DatasetGenerator(scanListPath=self.trainSplitPath, ImageRootPath=self.imageRootPath,
                                             Vocaubulary=self.vocabulary)
        self.testDataset = DatasetGenerator(scanListPath=self.testSplitPath, ImageRootPath=self.imageRootPath,
                                            Vocaubulary=self.vocabulary)
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, shuffle=True, batch_size=self.batchSize,
                                          num_workers=24, pin_memory=self.pinMemory)
        self.testDataLoader = DataLoader(dataset=self.testDataset, shuffle=True, batch_size=self.batchSize,
                                         num_workers=24, pin_memory=self.pinMemory)

        self.rgbResNetModel = MyResNet()
        self.depthModel = DepthResNet()
        self.relevanceModel = RelevanceMap(wordEmbeddingNum=len(self.vocabulary),
                                           wordEmbeddingDim=self.wordEmbeddingDim)

        self.rgbOptimizer = optim.Adam(self.rgbResNetModel.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=self.weightDecay)
        self.relevanceOptimizer = optim.Adam(self.relevanceModel.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=self.weightDecay)

        self.rgbScheduler = ReduceLROnPlateau(self.rgbOptimizer, factor=0.1, patience=10, mode='min')
        self.relevanceScheduler = ReduceLROnPlateau(self.relevanceOptimizer, factor=0.1, patience=10, mode='min')

        self.loss = torch.nn.BCELoss(size_average=True)

    def epochTrain(self):
        lossEpoch = []

        for idx,(rgb, depthOri, targetOri, category) in enumerate(self.trainDataLoader):
            rgb = Variable(rgb)
            depth = Variable(depthOri, requires_grad=False)
            target = Variable(targetOri, requires_grad=False)
            category = Variable(category)

            feature = self.rgbResNetModel(rgb)
            depth = self.depthModel(depth)
            target = self.depthModel(target)

            featureTarget = torch.cat((feature,target),dim=1)
            featureTargetTopView = generateTopView(featureTarget, depth)
            featureTopView = featureTargetTopView[:,0:128,:,:]
            targetTopView = featureTargetTopView[:,128,:,:]

            releMap = self.relevanceModel(featureTopView, category)

            # plt.figure()
            # plt.subplot(3,3,1)
            # plt.imshow(rgb[0,0,:,:].detach().numpy())
            # plt.subplot(3,3,2)
            # plt.imshow(depthOri[0,0,:,:].numpy())
            # plt.subplot(3,3,3)
            # plt.imshow(targetOri[0,0,:,:].numpy())
            # plt.subplot(3,3,4)
            # plt.imshow(feature[0,0,:,:].detach().numpy())
            # plt.subplot(3,3,5)
            # plt.imshow(depth[0,0,:,:].detach().numpy())
            # plt.subplot(3,3,6)
            # plt.imshow(target[0,0,:,:].detach().numpy())
            # plt.subplot(3,3,7)
            # plt.imshow(featureTopView[0,0,:,:].detach().numpy())
            # plt.subplot(3,3,8)
            # plt.imshow(releMap[0,:,:].detach().numpy())
            # plt.subplot(3,3,9)
            # plt.imshow(targetTopView[0,:,:].detach().numpy())
            # plt.ioff()
            # plt.show()

            targetTopView = targetTopView.detach()
            lossValue = self.loss(input=releMap, target=targetTopView)

            lossEpoch.append(lossValue.item())

            self.rgbOptimizer.zero_grad()
            self.relevanceOptimizer.zero_grad()
            lossValue.backward()
            self.rgbOptimizer.step()
            self.relevanceOptimizer.step()


if __name__ == '__main__':
    Task = SemanticMapTask()
    Task.epochTrain()
