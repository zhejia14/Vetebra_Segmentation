import os
import torch
import cv2
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt


def find_contour(mask_img, thickness=-1):
    contours, _ = cv2.findContours(
        mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    separated_instances = np.zeros_like(mask_img)
    color_give = 255
    for i, contour in enumerate(contours):
        cv2.drawContours(separated_instances, [contour], -1, color=color_give-i, thickness=thickness)
    
    return separated_instances



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "label"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        filename = os.path.basename(img_path)
        file = int(os.path.splitext(filename)[0])
        mask_path = os.path.join(self.root, "label", self.masks[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = find_contour(mask)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor(file)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)



def main():
    dataset = CustomDataset(root='./demo')
    img, target = dataset[0]
    file = target['image_id']
    print(file)
    # masks = target['masks']
    # masks = masks.numpy()
    # for i in range(len(masks)):
    #     mask_img = (masks[i] * 255).astype(np.uint8)
    #     plt.imshow(mask_img, cmap='gray')
    #     plt.show()


if __name__ == '__main__':
    main()