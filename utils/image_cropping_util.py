import os
from PIL import Image
import json

#####################################################
# Utility for cropping original 1920x1200 images to 
# 1200x1200 images and saving in separate directory.
# Symmetry insures that Pytorch-Lightning repo
# works as designed while also making sure that 
# focal lengths are not affected by resizing as opposed
# to cropping images.
#####################################################

speed_root = "../Data/speed"
split = "val_small"
image_root = os.path.join(speed_root, 'images', split)
sample_ids = []
save_dir = "../Data/speed/images/val_small_symmetric"

with open(os.path.join(speed_root, split + '.json'), 'r') as f:
    label_list = json.load(f)
    # self.sample_id is a list of filenames
    sample_ids = [label['filename'] for label in label_list]

box = (360, 0, 1560, 1200)
for sample_id in sample_ids:
    img_name = os.path.join(image_root, sample_id)
    img = Image.open(img_name)
    crop = img.crop(box)

    save_name = os.path.join(save_dir, sample_id)
    crop.save(save_name)