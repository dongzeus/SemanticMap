import os
import sys

sys.path.append('./build')

import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


import torch
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader

from semantic_model import MyResNet, DepthResNet, RelevanceMap
from semantic_utils import generateTopView, buildVocaulary
from semantic_dataset import DatasetGenerator

objectCategoryPath = './data/category_mapping.tsv'
regionCategoryPath = './data/region_category.txt'

splitPath = './dataset_split/tmp.txt'
ImageRootPath = '/Volumes/Dongqiyuan/matterport_dataset'


vocabulary = buildVocaulary(regionCategoryPath, objectCategoryPath)
rgbResNet = MyResNet()
depthResNet = DepthResNet()
# relevanceMap = RelevanceMap(wordEmbeddingNum=len(vocabulary))


tmpDataset = DatasetGenerator(scanListPath=splitPath, ImageRootPath=ImageRootPath, Vocaubulary=vocabulary)
tmpDataLoader = DataLoader(dataset=tmpDataset, batch_size=4, shuffle=True, num_workers=4)

for idx, (rgb, depth, target, category) in enumerate(tmpDataLoader):
    rgbVar = Variable(rgb)
    depthVar = Variable(depth)
    categoryVar = Variable(category)

    feature = rgbResNet(rgbVar)
    depthVar = depthResNet(depthVar)

    featureTopView = generateTopView(feature,depthVar)

    releMap = relevanceMap(featureTopView,categoryVar)

    print releMap.size()

