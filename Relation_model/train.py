import os

import datetime
import time
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from model import RN
from dataset import DatasetGenerator
from utils import Tokenizer, sort_batch


class Task():
    def __init__(self, args):
        print '#' * 60
        print ' ' * 20 + '    Task Created    ' + ' ' * 20
        print '#' * 60

        ######################################################################################################
        # Parameters
        self.batchSize = args.batchSize
        self.lr = args.lr
        self.weightDecay = 1e-4

        self.objNumMax = 60
        self.wordEmbeddingDim = 64
        self.lstmHiddenDim = 128
        self.instructionLength = 10
        self.pinMemory = True
        self.dropout = False

        self.epoch = args.epoch
        self.epoch_i = 0

        self.batchPrint = 100
        self.batchModelSave = args.batchModelSave
        self.checkPoint = args.checkPoint

        # Path
        self.scanListTrain = '../data/scan_list_train.txt'
        self.scanListTest = '../data/scan_list_test.txt'
        self.datasetPath = '../generated_data'
        self.logPath = args.logPath

        # Dataset
        self.tokenizer = Tokenizer(encoding_length=self.instructionLength)
        self.trainDataset = DatasetGenerator(scanListPath=self.scanListTrain, datasetPath=self.datasetPath)
        self.testDataset = DatasetGenerator(scanListPath=self.scanListTest, datasetPath=self.datasetPath)

        # build vocabulary from all instructions in the training dataset
        self.tokenizer.build_vocab_from_dataset(self.trainDataset)

        # DataLoader
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, shuffle=True, batch_size=self.batchSize,
                                          num_workers=12, pin_memory=self.pinMemory)
        self.testDataLoader = DataLoader(dataset=self.testDataset, shuffle=False, batch_size=self.batchSize,
                                         num_workers=12, pin_memory=self.pinMemory)
        # calculate batch numbers
        self.trainBatchNum = int(np.ceil(len(self.trainDataset) / float(self.batchSize)))
        self.testBatchNum = int(np.ceil(len(self.testDataset) / float(self.batchSize)))

        # Create model
        self.RN = RN(batch_size=self.batchSize, num_objects=self.objNumMax,
                     vocab_size=self.tokenizer.get_vocal_length(), embedding_size=self.wordEmbeddingDim,
                     hidden_size=self.lstmHiddenDim, padding_idx=1, dropout=self.dropout)

        # Run task on all available GPUs
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print("Use ", torch.cuda.device_count(), " GPUs!")
                self.RN = nn.DataParallel(self.RN)
            self.RN = self.RN.cuda()
            print 'Model Created on GPUs.'

        # Optermizer
        self.optimizer = optim.Adam(self.RN.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=self.weightDecay)

        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10, mode='min')

        # Loss Function
        self.loss = torch.nn.MSELoss()

        # Load model given a checkPoint
        if self.checkPoint != "":
            self.load(self.checkPoint)

        # create TensorboardX record
        self.writer = SummaryWriter(comment='word_embedding_64_lstm_hidden_state_128')
        self.stepCnt_train = 1
        self.stepCnt_test = 1

    def train(self):

        print 'Training task begin.'
        print '----Batch Size: %d' % self.batchSize
        print '----Learning Rate: %f' % (self.lr)
        print '----Epoch: %d' % self.epoch
        print '----Log Path: %s' % self.logPath

        for self.epoch_i in range(self.epoch):

            # if self.epoch_i == 0:
            #     self.save(batchIdx=0)  # Test the save function

            if self.epoch_i != 0:
                try:
                    self.map = self.map.eval()
                    self.test()
                except Exception, e:
                    print e

            self.RN = self.RN.train()
            self.epochTrain()
            self.save()

    def epochTrain(self):
        s = '#' * 30 + '    Epoch %3d / %3d    ' % (self.epoch_i + 1, self.epoch) + '#' * 30
        print s
        bar = tqdm(self.trainDataLoader)
        for idx, (objs, target_bb, instruction) in enumerate(bar):

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # First sort the batch according to the length of instruction
            batchSize = objs.size(0)
            instruction_idx = None
            for i in range(batchSize):
                if instruction_idx is None:
                    instruction_idx = self.tokenizer.encode_sentence(instruction[i])
                else:
                    instruction_idx = np.concatenate((instruction_idx, self.tokenizer.encode_sentence(instruction[i])),
                                                     axis=0)

            seq_lengths, perm_idx = sort_batch(instruction_idx)  # input in numpy and return in tensor
            instruction_idx = torch.from_numpy(instruction_idx).long()
            seq_lengths = seq_lengths.long()
            
            # Norm
            # print objs.shape,objs[0]

            with torch.no_grad():
                bs, dim_N, fea = objs.size()
                objs_label = objs[:,:,:1]
                objs_feature = objs[:,:,1:]
                num_regions = []    # the number of regions in each batch
                batch_mean = []  # the mean of non-zero element in each batch
                for batch_iter in range(bs):
                    tmpfeature = objs_feature[batch_iter]   # dim_N * 4
                    total_sum = 0
                    for i in range(self.objNumMax):
                        tmp_sum = tmpfeature[i].sum().item()
                        if tmp_sum !=0:
                            total_sum += tmp_sum
                        else:
                            batch_mean.append(total_sum/((fea-1)*i))
                            num_regions.append(i)
                            break

                for batch_iter in range(bs):
                    try:
                        num_r = num_regions[batch_iter]  # number of region proposals
                    except:
                        print "Error!",batch_iter
                        exit(1)
                    tmp_mean = torch.tensor([batch_mean[batch_iter]]).unsqueeze(1).repeat(num_r, fea - 1)
                    tmp_mean = torch.cat((tmp_mean, torch.zeros(self.objNumMax - num_r, fea - 1)),0)
                    objs_feature[batch_iter] = objs_feature[batch_iter] - tmp_mean

                objs = torch.cat((objs_label,objs_feature),2)

                # print objs_feature.size(),bs
                # objs_f_flat = objs_feature.contiguous().view(bs,dim_N * (fea-1))
                # objs_f_mean = torch.mean(objs_f_flat, 1,keepdim=True)
                # objs_f_mean_in = objs_f_mean.unsqueeze(2).repeat(1,dim_N,fea-1)

                # objs_mean_ta = objs_mean.unsqueeze(1).repeat(1,4)
                #objs_max = torch.max(objs_flat, 1)[0]
                # print objs_max.shape
                # exit(1)
                #objs_max_in = objs_max.unsqueeze(1).unsqueeze(2).repeat(1,dim_N,fea)
                # objs_max_ta = objs_max.unsqueeze(1).repeat(1,4)
                # objs_feature = objs_feature - objs_f_mean_in 
                # objs = torch.cat((objs_label,objs_feature),2)
                # target_bb = (target_bb - objs_mean_ta) / objs_max_ta

            # print objs.shape,objs[0]
            # exit(1)

            # to cuda
            if torch.cuda.is_available():
                objs = objs.cuda()
                target_bb = target_bb.cuda()
                instruction_idx = instruction_idx.cuda()
                perm_idx = perm_idx.cuda()

            # sort according the length
            objs = objs[perm_idx]
            target_bb = target_bb[perm_idx]
            instruction_idx = instruction_idx[perm_idx]

            # Go through the models
            output_bb = self.RN(objs, instruction_idx, seq_lengths)  # 1024 * 28 * 28

            # calculate loss
            lossValue = self.loss(input=output_bb, target=target_bb)

            # Tensorboard record
            self.writer.add_scalar('Loss/Train', lossValue.item(), self.stepCnt_train)
            self.stepCnt_train += 1

            # print loss
            bar.set_description('Epoch: %d    Loss: %f' % (self.epoch_i + 1, lossValue.item()))

            # Backward
            self.optimizer.zero_grad()
            lossValue.backward()
            self.optimizer.step()
            self.scheduler.step(lossValue)

            # Save model
            if (idx + 1) % self.batchModelSave == 0:
                self.save(batchIdx=(idx + 1))

            if idx % self.batchPrint == 0:
                s = ''
                output_bb_numpy = output_bb.detach().cpu().numpy()
                target_bb_numpy = target_bb.detach().cpu().numpy()
                for i in range(output_bb_numpy.shape[0]):
                    s += ' ### '
                    for j in range(4):
                        s += str(target_bb_numpy[i][j]) + ', '
                    s += ' & '
                    for j in range(4):
                        s += str(output_bb_numpy[i][j]) + ', '
                self.writer.add_text('Target & Output', s, self.stepCnt_train)

            del lossValue

    def test(self):
        s = '#' * 28 + '  Test  Epoch %3d / %3d    ' % (self.epoch_i + 1, self.epoch) + '#' * 28
        print s

        bar = tqdm(self.testDataLoader)
        for idx, (objs, target_bb, instruction) in enumerate(bar):

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # First sort the batch according to the length of instruction
            batchSize = objs.size(0)
            instruction_idx = None
            for i in range(batchSize):
                if instruction_idx is None:
                    instruction_idx = self.tokenizer.encode_sentence(instruction[i])
                else:
                    instruction_idx = np.concatenate((instruction_idx, self.tokenizer.encode_sentence(instruction[i])),
                                                     axis=0)

            seq_lengths, perm_idx = sort_batch(instruction_idx)  # input in numpy and return in tensor
            instruction_idx = torch.from_numpy(instruction_idx).long()
            seq_lengths = seq_lengths.long()

            # require grad
            objs = objs.requires_grad()
            target_bb = target_bb.requires_grad()
            instruction_idx = instruction_idx.requires_grad()

            # to cuda
            if torch.cuda.is_available():
                objs = objs.cuda()
                target_bb = target_bb.cuda()
                instruction_idx = instruction_idx.cuda()
                perm_idx = perm_idx.cuda()

            # sort according the length
            objs = objs[perm_idx]
            target_bb = target_bb[perm_idx]
            instruction_idx = instruction_idx[perm_idx]

            # Go through the models
            output_bb = self.RN(objs, instruction_idx, seq_lengths, target_bb)  # 1024 * 28 * 28

            # calculate loss
            lossValue = self.loss(input=output_bb, target=target_bb)

            # Tensorboard record
            self.writer.add_scalar('Loss/Test', lossValue.item(), self.stepCnt_test)
            self.stepCnt_test += 1



            del lossValue

    def save(self, batchIdx=None):
        dirPath = os.path.join(self.logPath, 'models')

        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        if batchIdx is None:
            path = os.path.join(dirPath, 'Epoch-%03d-end.pth.tar' % (self.epoch_i + 1))
        else:
            path = os.path.join(dirPath, 'Epoch-%03d-Batch-%04d.pth.tar' % (self.epoch_i + 1, batchIdx))

        torch.save({'epochs': self.epoch_i + 1,
                    'batch_size': self.batchSize,
                    'lr': self.lr,
                    'weight_dacay': self.weightDecay,
                    'RN_model_state_dict': self.RN.state_dict()},
                   path)
        print 'Training log saved to %s' % path

    def load(self, path):
        modelCheckpoint = torch.load(path)
        self.RN.load_state_dict(modelCheckpoint['RN_model_state_dict'])
        print 'Load model from %s' % path
