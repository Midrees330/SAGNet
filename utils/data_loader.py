import os
import glob
import torch
import torch.utils.data as data
from . import ISTD_transforms
from PIL import Image
import random
from torchvision import transforms
import matplotlib.pyplot as plt

def make_datapath_list(phase="train", rate=0.8):
    
    random.seed(44)

    rootpath = './dataset/' + phase + '/'
    
    # Check if train_A folder exists
    input_folder = rootpath + phase + '_A'
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder not found: {input_folder}")
    
    files_name = os.listdir(input_folder)
    
    if phase == 'train':
        random.shuffle(files_name)
    elif phase == 'test':
        files_name.sort()

    path_A = []  # Input shadow images
    path_B = []  # Shadow mask (if exists) or None
    path_C = []  # Ground truth shadow-free images
    
    # Check dataset structure by looking for train_C folder
    has_mask = os.path.exists(rootpath + phase + '_C')
    
    
    if has_mask:
        # Shadow removal WITH masks: A=input, B=mask, C=GT
        
        for name in files_name:
            path_A.append(rootpath + phase + '_A/' + name)
            path_B.append(rootpath + phase + '_B/' + name)
            path_C.append(rootpath + phase + '_C/' + name)
    else:
        # Shadow removal WITHOUT masks: A=input, B=GT, no mask
        
        for name in files_name:
            path_A.append(rootpath + phase + '_A/' + name)
            path_B.append(None)  # No mask annotation
            path_C.append(rootpath + phase + '_B/' + name)  # GT from B folder

    num = len(path_A)
    print(f"Total images found: {num}")

    if phase == 'train':
        split_idx = int(num * rate)
        path_A, path_A_val = path_A[:split_idx], path_A[split_idx:]
        path_B, path_B_val = path_B[:split_idx], path_B[split_idx:]
        path_C, path_C_val = path_C[:split_idx], path_C[split_idx:]
        
        path_list = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C, 'has_mask': has_mask}
        path_list_val = {'path_A': path_A_val, 'path_B': path_B_val, 'path_C': path_C_val, 'has_mask': has_mask}
        
        print(f"Train split: {len(path_A)} images")
        print(f"Val split: {len(path_A_val)} images")
        print(f"{'='*60}\n")
        
        return path_list, path_list_val

    elif phase == 'test':
        path_list = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C, 'has_mask': has_mask}
        print(f"Test images: {len(path_A)}")
        print(f"{'='*60}\n")
        return path_list


class ImageTransformOwn():
    """Preprocessing images for own images"""
    def __init__(self, size=256, mean=(0.5,), std=(0.5,)):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class ImageTransform():
    """Preprocessing images"""
    def __init__(self, size=286, crop_size=256, mean=(0.5,), std=(0.5,)):
        self.data_transform = {
            'train': ISTD_transforms.Compose([
                ISTD_transforms.Scale(size=size),
                ISTD_transforms.RandomCrop(size=crop_size),
                ISTD_transforms.RandomHorizontalFlip(p=0.5),
                ISTD_transforms.RandomVerticalFlip(p=0.5),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            
            'val': ISTD_transforms.Compose([
                ISTD_transforms.Scale(size=size),
                ISTD_transforms.RandomCrop(size=crop_size),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            
            'test': ISTD_transforms.Compose([
                ISTD_transforms.Scale(size=size),
                ISTD_transforms.RandomCrop(size=crop_size),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            
            'test_no_crop': ISTD_transforms.Compose([
                ISTD_transforms.Resize([256, 256]),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, phase, img):
        return self.data_transform[phase](img)


class ImageDataset(data.Dataset):
   
    def __init__(self, img_list, img_transform, phase):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase
        self.has_mask = img_list.get('has_mask', True)  # Default to True for backward compatibility

    def __len__(self):
        return len(self.img_list['path_A'])

    def __getitem__(self, index):
        
        # Load shadow input image
        img = Image.open(self.img_list['path_A'][index]).convert('RGB')
        
        # Load or create shadow mask
        if self.has_mask and self.img_list['path_B'][index] is not None:
            # Dataset HAS mask annotations: load actual shadow mask from path_B
            gt_shadow = Image.open(self.img_list['path_B'][index])
        else:
            # Dataset DOESN'T have mask annotations: create dummy mask
            # Your model's SAID module will detect shadows automatically
            gt_shadow = Image.new('L', img.size, 0)
        
        # Load shadow-free ground truth
        gt = Image.open(self.img_list['path_C'][index]).convert('RGB')

        # Apply transformations (resize, crop, flip, normalize)
        img, gt_shadow, gt = self.img_transform(self.phase, [img, gt_shadow, gt])

        return img, gt_shadow, gt


# Test code
if __name__ == '__main__':
    
    # Test 1: Try loading as shadow removal dataset
    print("\n1. Testing Shadow Removal Dataset Structure:")
    print("-"*80)
    try:
        train_list, val_list = make_datapath_list(phase='train', rate=0.8)
        print(" Successfully loaded!")
        
        # Create dataset and get a sample
        img_transforms = ImageTransform(size=286, crop_size=256, mean=(0.5,), std=(0.5,))
        train_dataset = ImageDataset(train_list, img_transforms, phase='train')
        
        img, mask, gt = train_dataset[0]

        
    except Exception as e:
        print(f" Error: {e}")
    
    # Test 2: Test dataset
    print("\n2. Testing Test Dataset:")
    print("-"*80)
    try:
        test_list = make_datapath_list(phase='test')
        print(" Successfully loaded test data!")
        
        img_transforms = ImageTransform(size=286, crop_size=256, mean=(0.5,), std=(0.5,))
        test_dataset = ImageDataset(test_list, img_transforms, phase='test_no_crop')
        
        if len(test_dataset) > 0:
            img, mask, gt = test_dataset[0]

        
    except Exception as e:
        print(f" Error: {e}")