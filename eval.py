import os, sys
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from Dataset import CustomDataset


def find_contour(mask_img, thickness=5):
    contours, _ = cv2.findContours(
        mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    separated_instances = np.zeros_like(mask_img)
    separated_instances = cv2.cvtColor(separated_instances, cv2.COLOR_GRAY2RGB)    
    for i, contour in enumerate(contours):
        cv2.drawContours(separated_instances, [contour], -1, color=(255, 0, 0), thickness=thickness)
    
    return separated_instances

def vertebra_dice(pred_img, gt_img):
    dice_score = None
    intersection = np.sum(pred_img * gt_img)
    pred_sum = np.sum(pred_img)
    gt_sum = np.sum(gt_img)

    dice_score = (2. * intersection) / (pred_sum + gt_sum)
    if dice_score is not None:
        return dice_score
    else:
        return 0


if __name__ == '__main__':
    model_path = sys.argv[1]
    device = "cpu"
    model = torch.load(model_path).to(device)
    model.eval()
    test_path = sys.argv[2]
    threshold = float(sys.argv[3])
    
    testDS = CustomDataset(root=test_path, transforms=None) 
    print("Test dataset size:", len(testDS))

    for f in range(len(testDS)):
        img, target = testDS[f]
        transform_img = transforms.ToPILImage()(img)
        transform_img = transforms.ToTensor()(transform_img)
        with torch.no_grad():
            prediction = model([transform_img.to(device)])
            bboxes = prediction[0]['boxes']
            pred_mask_list = prediction[0]['masks']
            scores = (prediction[0]['scores']).numpy()
            pred_mask_list = torch.squeeze(pred_mask_list)
            pred_mask_list = pred_mask_list.numpy()
            center_list = []
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = map(int, bboxes[i])
                center = (abs(x2-x1)//2 + x1 , abs(y2-y1)//2 + y1)
                center_list.append(center)
            sorted_indices = sorted(range(len(center_list)), key=lambda i: center_list[i][1])
            center_list = [center_list[i] for i in sorted_indices]
            sorted_pred_mask_list = [pred_mask_list[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            print("scores:", scores )
            temp_center_list = []
            temp_pred_mask_list = []
            temp_scores = []
            for center, mask, score in zip(center_list, sorted_pred_mask_list, scores):
                if score >= 0.9:
                    temp_center_list.append(center)
                    temp_pred_mask_list.append(mask)
                    temp_scores.append(score)
            center_list = temp_center_list
            sorted_pred_mask_list = temp_pred_mask_list
            scores = temp_scores
            sum = 0
            contour_list = []
            file = "00" + str((target['image_id']).numpy()) + ".png"
            print("img:{}\nNumber of Detect:\nGT:{}\nDetected:{}".format(file, len(target['labels']), len(center_list)))
            gt_mask_list = (target['masks']).numpy()
            for i in range(len(sorted_pred_mask_list)):
                pred = (sorted_pred_mask_list[i] > threshold) * 1
                gt = gt_mask_list[i]
                contour = find_contour((pred * 255).astype(np.uint8))
                dc = vertebra_dice(pred, gt)
                contour_list.append(contour)
                print("V{} DC:{:.4f}".format(i, dc))
                sum = sum + dc
            print("Average DC:{:.4f}".format(sum/len(sorted_pred_mask_list)))            
            combined_mask = np.zeros((sorted_pred_mask_list[0].shape[0], sorted_pred_mask_list[0].shape[1]), dtype=np.uint8)
            combined_mask = np.maximum.reduce(sorted_pred_mask_list, axis=0)
            combined_mask = ((combined_mask > threshold) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(model_path.split(os.path.sep)[1], file), combined_mask)
            color_img = img.copy()
            location_img = img.copy()
            for center in center_list:
                cv2.circle(location_img, center, radius=5, color=(255, 0, 0), thickness=-1)
            contour_mask = np.zeros_like(color_img)
            contour_mask = np.maximum.reduce(contour_list, axis=0)
            contour_img = cv2.addWeighted(contour_mask, 0.3, color_img, 0.7, 0)
            plt.subplot(1, 3, 1)
            plt.imshow(color_img)
            plt.title(file)

            plt.subplot(1, 3, 2)
            plt.imshow(location_img)
            plt.title("Location")

            plt.subplot(1, 3, 3)
            plt.imshow(contour_img)
            plt.title("Contour")
            plt.savefig(os.path.join(model_path.split(os.path.sep)[1], "output_" + file))
            plt.show()

