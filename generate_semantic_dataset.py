import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from semantic_utils import buildObjectCategory, buildRegionCategory

if __name__ == '__main__':
    objectCategoryPath = './data/category_mapping.tsv'
    regionCategoryPath = './data/region_category.txt'

    sampledSemanticImagePath = '/Volumes/Dongqiyuan/matterport_dataset/sampled_semantic_images'
    generatedTrainingImagePath = '/Volumes/Dongqiyuan/matterport_dataset/generated_semantic_training_images'

    objectCategory = buildObjectCategory(objectCategoryPath)
    regionCategory = buildRegionCategory(regionCategoryPath)

    scans = ['gTV8FGcVJC9']
    # scans = scans[4:14]
    # scans = scans[14:28]

    for scan in scans:
        scanSampledSemanticImagePath = os.path.join(sampledSemanticImagePath, scan)
        scanSavePath = os.path.join(generatedTrainingImagePath, scan)
        assert os.path.exists(scanSampledSemanticImagePath)
        if not os.path.exists(scanSavePath):
            os.mkdir(scanSavePath)

        # Only use images taken from horizontal view
        images = [i for i in os.listdir(scanSampledSemanticImagePath) if 12 <= int(i[:-4].split('_')[-1]) <= 23]

        for idx, img_path in enumerate(images):
            img = (plt.imread(os.path.join(scanSampledSemanticImagePath, img_path)) * 255).astype('uint16')
            region = img[:, :, 0]
            object = img[:, :, 1] * 10 + img[:, :, 2] / 20

            region_cnt = sorted([(np.sum(region == i), i) for i in set(region.flat)])
            region_cnt.reverse()

            object_cnt = sorted([(np.sum(object == i), i) for i in set(object.flat)])
            object_cnt.reverse()

            imgId = img_path[:-4]
            for cnt, id in region_cnt:
                if id == 0:
                    continue
                if cnt < 10000:
                    break
                if not regionCategory.has_key(id):
                    continue

                imgPath = os.path.join(scanSavePath, imgId + '_%s.png' % regionCategory[id])
                cv2.imwrite(imgPath, (region == id).astype('uint8') * 255)

            for cnt, id in object_cnt:
                if id == 0:
                    continue
                if cnt < 2000:
                    break
                if not objectCategory.has_key(id) or objectCategory[id] == 'unknown' or objectCategory[id] == 'void':
                    continue

                imgPath = os.path.join(scanSavePath, imgId + '_%s.png' % objectCategory[id])
                cv2.imwrite(imgPath, (object == id).astype('uint8') * 255)

            if (idx + 1) % 10 == 0:
                print '%s finish %d / %d' % (scan, idx + 1, len(images))
