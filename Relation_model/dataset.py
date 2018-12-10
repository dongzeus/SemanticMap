import os
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import get_all_scanId


class DatasetGenerator(Dataset):
    def __init__(self, scanListPath, datasetPath, objNumMax=60):
        self.objNumMax = objNumMax
        self.dataSavePath = scanListPath.replace('.txt', '_data_saved.json')

        # if data file exists, directly load
        if os.path.exists(self.dataSavePath):
            with open(self.dataSavePath, 'r') as fp:
                self.data = json.load(fp)

        else:
            self.scanIds = get_all_scanId(scanListPath)
            jsonFilesPathAll = []
            self.data = []
            for scanId in self.scanIds:
                jsonFiles = [i for i in os.listdir(os.path.join(datasetPath, scanId)) if '.json' in i]
                # (scanId, locId, headingId, jsonFilePath)
                jsonFilesPathAll += [(scanId, jsonF.replace('.json', '').split('_')[0],
                                      jsonF.replace('.json', '').split('_')[1],
                                      os.path.join(datasetPath, scanId, jsonF))
                                     for jsonF in jsonFiles]

            bar = tqdm(jsonFilesPathAll)
            for scanId, locId, headingId, jsonFilesPath in bar:
                objs = self.get_all_boundingbox(jsonFilesPath)  # [ [cateIdx,lowx,lowy,highx,highy], ... ]
                datas = self.get_all_training_data(
                    jsonFilesPath)  # [ [instruction, t_lowx, t_lowy, t_highx, t_highy], ... ]
                for d in datas:
                    # objs: [ [cateIdx,lowx,lowy,highx,highy], ... ]
                    # d   : [instruction, t_lowx, t_lowy, t_highx, t_highy]
                    self.data.append((objs, d[0], d[1:], scanId, locId, headingId))
                    bar.set_description('Loading data... ')

            with open(self.dataSavePath, 'w') as fp:
                json.dump(self.data, fp)

    def get_all_boundingbox(self, path):
        with open(path, 'r') as fp:
            data = json.load(fp)
        objs = [[d['object']['category_index']] + d['boundingbox'] for d in data if d['type'] == 'boundingbox']
        return objs

    def get_all_training_data(self, path):
        with open(path, 'r') as fp:
            data = json.load(fp)

        datas = [[d['instruction']] + d['targetBoundingbox'] for d in data if d['type'] == 'data']
        return datas

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        objs = np.array(data[0]).reshape(-1, 5)  # N * 5
        # make N = self.objNumMax
        objsNum = objs.shape[0]
        objs = np.concatenate((objs, np.zeros((self.objNumMax - objsNum, 5))), axis=0).astype('float32')

        instruction = data[1]
        target_bb = np.array(data[2]).astype('float32').reshape(4)  # 1 * 4

        objs = torch.from_numpy(objs)
        target_bb = torch.from_numpy(target_bb)

        return objs, target_bb, instruction


if __name__ == '__main__':
    scanListPath = '../data/scan45.txt'
    datasetPath = '/home/qiyuand/Projects/Navigation/data_generation/generated_data'
    allDataset = DatasetGenerator(scanListPath=scanListPath, datasetPath=datasetPath)

    obj_num = []
    for objs, target, instr in allDataset:
        obj_num.append(objs.shape[0])
    print np.max(obj_num)
