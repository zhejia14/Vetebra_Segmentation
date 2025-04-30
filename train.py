import os, sys
import torch
from torchvision import transforms
from Dataset import CustomDataset
from torch.utils.data import DataLoader, ConcatDataset
from Maskrcnn_Model import get_instance_segmentation_model
from engine import train_one_epoch, evaluate
import utils



def train():
    
    save_path = sys.argv[1]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    batchsize = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2 # 1 class and 1 background
    lr = 0.005
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])
    dataset_1 = CustomDataset(root='./dataset/f01/', transforms=transform)
    dataset_2 = CustomDataset(root='./dataset/f02/', transforms=transform)
    dataset_3 = CustomDataset(root='./dataset/f03/', transforms=transform)

    dataset = ConcatDataset([dataset_1, dataset_2])
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)

    val_loader = DataLoader(dataset_3, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

    model = get_instance_segmentation_model(num_classes)

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,momentum=0.9, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.002, total_iters=100)

    num_epochs = 150
    save_step = 25
    save_times = 0
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        print("EPOCH:[{}/{}]".format(epoch, num_epochs))
        train_one_epoch(model, optimizer, train_loader, device, epoch)

        # update the learning rate
        # scheduler.step()

        # evaluate on the test dataset
        evaluate(model, val_loader, device=device)
        if epoch != 0 and epoch % save_step == 0:
            save_times = save_step + save_times
            torch.save(model, os.path.join(save_path,str(save_times)+"_output.pth"))
    
    torch.save(model, os.path.join(save_path, str(num_epochs)+"_output.pth"))


if __name__ == '__main__':
    train()
