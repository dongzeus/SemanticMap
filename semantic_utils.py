import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

features = Variable(torch.rand(2,128,80,80))
depth = Variable(torch.rand(2,80,80))
imageScale = 1.0

def generateTopView(features, depth):

    batchSize, channel, H, W = features.size()
    CAMERA_FACTOR = 4000.0
    topViewSize = 80
    imageScale = H / 640.0
    cx = 320.0 * imageScale
    cy = 320.0 * imageScale
    fx = 320.0 * imageScale
    fy = 320.0 * imageScale

    # Cut off depth()
    depth = depth.data.numpy()
    mask = (depth >= 3.0 * depth.mean())
    depth[mask] = 0.0
    # depth = np.squeeze(depth,1)
    # Calculate the coordinates
    pz = depth * 65535 / CAMERA_FACTOR

    mesh_px = np.repeat(np.array(range(0, W)).reshape(1, -1), H, axis=0).reshape(1, H, W)
    mesh_px = np.repeat(mesh_px, batchSize, axis=0).astype('float32')
    px = (mesh_px - cx) * pz / fx

    mesh_py = range(0, H)
    mesh_py.reverse()
    mesh_py = np.repeat(np.array(mesh_py).reshape(-1, 1), W, axis=1).reshape(1, H, W)
    mesh_py = np.repeat(mesh_py, batchSize, axis=0).astype('float32')
    py = (mesh_py - cy) * pz / fy

    # pz = -1 * pz

    px = px.reshape(batchSize, H, W, 1)
    py = py.reshape(batchSize, H, W, 1)
    pz = pz.reshape(batchSize, H, W, 1)

    coor = np.concatenate((px, py, pz), axis=3)
    coor = coor.reshape(batchSize, -1, 3)  # batchSize * 6400 * 3
    features = features.contiguous().view(batchSize, channel, -1) # batch * 128 *  6400
    features = torch.transpose(features,2,1)

    x_min = np.min(coor[:, :, 0], axis=1).reshape(batchSize, 1)
    y_min = np.min(coor[:, :, 1], axis=1).reshape(batchSize, 1)
    z_min = np.min(coor[:, :, 2], axis=1).reshape(batchSize, 1)

    coor[:, :, 0] -= x_min
    coor[:, :, 1] -= y_min
    coor[:, :, 2] -= z_min

    x_max = np.max(coor[:, :, 0], axis=1).reshape(batchSize, 1)
    y_max = np.max(coor[:, :, 1], axis=1).reshape(batchSize, 1)
    z_max = np.max(coor[:, :, 2], axis=1).reshape(batchSize, 1)


    for i in range(batchSize):
        if x_max[i] > z_max[i]:
            zoom_in = (topViewSize - 1) / x_max[i]
        else:
            zoom_in = (topViewSize - 1) / z_max[i]
        coor[i, :, :] *= zoom_in

    coor = np.floor(coor).astype('int32')

    topView = np.zeros((1, topViewSize, topViewSize,channel)).astype('float32')
    topView = np.repeat(topView, batchSize, axis=0)
    topView = Variable(torch.from_numpy(topView))

    for i in range(batchSize):
        x = coor[i,:, 0].reshape(-1)
        # y = coor[i,:, 1].reshape(-1)
        z = coor[i,:, 2].reshape(-1)

        ii = torch.LongTensor([i])
        x = torch.from_numpy(x).long()
        z = torch.from_numpy(z).long()
        z = topViewSize - 1 - z

        topView[ii,z,x,:]= features[ii,:,:]

    topView = topView.transpose(1,3)
    topView = topView.transpose(2,3)

    # for i in range(0,128):
    #     plt.imshow(topView[0,i,:,:].data.numpy() * 255)
    #     plt.ioff()
    #     plt.show()

    return topView


def buildObjectCategory(path):
    category = {}
    with open(path, 'r') as fp:
        for line in fp:
            if line[0] == 'i':
                continue

            item = line.split('\t')
            if item[7] == '':
                item[7] = item[2]
            category[int(item[0])] = item[7].replace(' ', '-')
            # category.append([item[0],item[1],item[2],item[7]])

    category[0] = 'unknown'

    return category


def buildRegionCategory(path):
    category = {}
    with open(path, 'r') as fp:
        for line in fp:
            num = ord(line[1])
            name = line[6:-1] if line[-1] == '\n' else line[6:]
            category[num] = name.replace(' ', '-')
    return category



def buildVocaulary(regionCategoryPath, objectCategoryPath):
    regionCategory = buildRegionCategory(regionCategoryPath)
    objectCategory = buildObjectCategory(objectCategoryPath)

    words = [value for (key,value) in regionCategory.items()] + [value for (key,value) in objectCategory.items()]
    words = list(set(words))

    vocabulary = {}
    vocabularyInverse = {}
    for idx,word in enumerate(words):
        vocabulary[word] = idx
        vocabularyInverse[idx] = word

    return vocabulary, vocabularyInverse

