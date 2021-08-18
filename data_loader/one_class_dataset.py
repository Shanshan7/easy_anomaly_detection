import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type


def data_argument(load_size, input_size):
    data_transforms = transforms.Compose([
        transforms.Resize((load_size, load_size), Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.CenterCrop(input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((load_size, load_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(input_size)])

    inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                              std=[1 / 0.229, 1 / 0.224, 1 / 0.255])

    return data_transforms, gt_transforms, inv_normalize

def get_train_dataloader(train_path, load_size, input_size, batch_size):
    data_transforms, gt_transforms, inv_normalize = data_argument(load_size, input_size)
    image_datasets = MVTecDataset(root=train_path, transform=data_transforms, gt_transform=gt_transforms, phase='train')
    train_loader = DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
    return train_loader

def get_test_dataloader(test_path, load_size, input_size):
    data_transforms, gt_transforms, inv_normalize = data_argument(load_size, input_size)
    test_datasets = MVTecDataset(root=test_path, transform=data_transforms, gt_transform=gt_transforms, phase='test')
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
    return test_loader


def test():
    train_path = "/home/edge/data/VOCdevkit/mvtec_anomaly_detection/mould"
    load_size = 256
    input_size = 224
    batch_size = 32
    data_loader = get_train_dataloader(train_path, load_size, input_size, batch_size)

    for index, batch_data in enumerate(data_loader):
        x, gt, label, file_name, x_type = batch_data
        print(x.shape, gt.shape, label.shape, file_name, x_type)


if __name__ == "__main__":
    test()