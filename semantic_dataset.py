import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from semantic_utils import buildVocaulary


class DatasetGenerator(Dataset):
    def __init__(self, scanListPath, ImageRootPath, Vocaubulary):

        self.ImageRootPath = ImageRootPath
        self.rgbImage = []
        self.depthImage = []
        self.targetImage = []
        self.category = []

        transformList = []
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.rgbTransformSequence = transforms.Compose(transformList)

        transformList = []
        transformList.append(transforms.ToTensor())
        self.depthTransformSequence = transforms.Compose(transformList)

        objectCategoryPath = './data/category_mapping.tsv'
        regionCategoryPath = './data/region_category.txt'
        self.vocabulary, _ =  buildVocaulary(regionCategoryPath, objectCategoryPath)

        scanListPath = scanListPath
        scans = []
        with open(scanListPath, 'r') as fp:
            for line in fp:
                if line[-1] == '\n':
                    line = line[:-1]
                scans.append(line)

        for scan in scans:
            scanTrainingPath = os.path.join(ImageRootPath, 'generated_semantic_training_images', scan)
            items = [i for i in os.listdir(scanTrainingPath) if len(i) > 10]
            for item in items:
                imageId = item.split('_')[0] + '_' + item.split('_')[1]
                self.rgbImage.append(os.path.join(ImageRootPath, 'sampled_color_images', scan, imageId + '.jpg'))
                self.depthImage.append(os.path.join(ImageRootPath, 'sampled_depth_images', scan, imageId + '.png'))
                self.targetImage.append(os.path.join(ImageRootPath, 'generated_semantic_training_images', scan, item))
                self.category.append(item.split('_')[-1][:-4])

    def __getitem__(self, item):
        rgbImage = plt.imread(self.rgbImage[item])
        depthImage = plt.imread(self.depthImage[item]).reshape(1, 640, 640).astype('float32')
        targetImage = plt.imread(self.targetImage[item]).reshape(1, 640, 640).astype('float32')
        category = self.category[item]

        rgbImage = self.rgbTransformSequence(rgbImage)
        depthImage = torch.from_numpy(depthImage)
        targetImage = torch.from_numpy(targetImage)

        return rgbImage, depthImage, targetImage, self.vocabulary[category]

    def __len__(self):
        return len(self.rgbImage)


if __name__ == '__main__':
    splitPath = './dataset_split/tmp.txt'
    ImageRootPath = '/media/psf/Dongqiyuan/matterport_dataset'

    tmpDataset = DatasetGenerator(scanListPath=splitPath, ImageRootPath=ImageRootPath)
    tmpDataLoader = DataLoader(dataset=tmpDataset, batch_size=4, shuffle=True, num_workers=4)

    for idx,(rgb,depth,target,category) in enumerate(tmpDataLoader):
        pass
