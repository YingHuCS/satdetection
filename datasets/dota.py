import numpy as np
from .custom import CustomDataset


class DOTADataset(CustomDataset):

    CLASSES = (
    'plane', 
    'ship', 
    'storage-tank', 
    'baseball-diamond', 
    'tennis-court', 
    'basketball-court', 
    'ground-track-field', 
    'harbor', 
    'bridge', 
    'small-vehicle', 
    'large-vehicle', 
    'helicopter', 
    'roundabout', 
    'soccer-ball-field', 
    'swimming-pool')

    def get_ann_info(self, idx):

      ann = self.img_infos[idx]['ann']

      bboxes = ann['bboxes']
      bboxes = np.asarray(bboxes, dtype=np.float32)

      labels = ann['labels']
      labels = [self.CLASSES.index(label)+1 for label in labels]
      labels = np.asarray(labels)


      converted_ann = {}
      converted_ann['bboxes'] = bboxes
      converted_ann['labels'] = labels
      converted_ann['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
      return converted_ann
