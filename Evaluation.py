import os, sys
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from Dataset import CustomDataset
from torch.utils.data import DataLoader
import utils

device = 'cuda'


def Dice_eval(model, val_ds):
    model.eval()

    threshold = 0.5

    for f in range(len(val_ds)):
        img, target = val_ds[f]
        prediction = model([img.to(device)])
        pred_mask_list = prediction[0]['masks']
        pred_mask_list = torch.squeeze(pred_mask_list)
        pred_mask_list = pred_mask_list.detach().cpu().numpy()
        combined_mask = np.zeros((pred_mask_list[0].shape[0], pred_mask_list[0].shape[1]), dtype=np.uint8)
        combined_mask = np.maximum.reduce(pred_mask_list, axis=0)
        combined_mask = (combined_mask > threshold) * 1
        gt_mask_list = (target['masks']).numpy()
        gt_mask = np.zeros((gt_mask_list[0].shape[0], gt_mask_list[0].shape[1]), dtype=np.uint8)
        gt_mask = np.maximum.reduce(gt_mask_list, axis=0)
        gt_mask = (gt_mask * 255).astype(np.uint8)
        plt.imshow(gt_mask)
        plt.show()



if __name__ == '__main__':
    model_path = sys.argv[1]
    model = torch.load(model_path).to(device)
    data_root = sys.argv[2]
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])
    val_ds = CustomDataset(root=data_root, transforms=transform)

    Dice_eval(model, val_ds)