import os
from tqdm import tqdm
from models import build_detector
from tools import inference_detector
from utils import load_checkpoint, Config
from datasets import DOTADataset

class_names = DOTADataset.CLASSES

def save_result(result, out_txt_path):
    f_w = open(out_txt_path, 'a')
    for i, item in enumerate(result):
        if len(item) != 0:
            label = class_names[i]
            for sub_item in item:
                content = str(sub_item[0]) + ' ' + str(sub_item[1]) + ' ' + str(sub_item[2]) + ' ' + str(sub_item[3]) + ' ' + str(sub_item[4]) + ' ' + label + '\n' 
                f_w.write(content)

    f_w.close()


cfg = Config.fromfile('configs/faster_rcnn_x101_64x4d_fpn_1x_dota.py')
cfg.model.pretrained = None

model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, '/mmdetection/data/satdet_work_dirs/fasterrcnn_dota1.0_original/epoch_2.pth') 


in_dir = '/mmdetection/data/DOTA-v1.0/test/images_splitted_multiscales/images'
out_dir = '/mmdetection/data/DOTA-v1.0/splitted_results'
img_names = os.listdir(in_dir)
selected_img_names = img_names



for name in tqdm(selected_img_names):
    in_img_path = os.path.join(in_dir, name)
    result = inference_detector(model, in_img_path, cfg, device='cuda:0')
    out_txt_path = os.path.join(out_dir, name.replace('.png', '.txt'))
    save_result(result, out_txt_path)


