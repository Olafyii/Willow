from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Willow(Dataset):
    def __init__(self, mode='train'):
        if mode=='train':
            self.txtpath = 'D:\\data\\Willow\\willowactions\\trainval.txt'
            self.transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        elif mode=='test':
            self.txtpath = 'D:\\data\\Willow\\willowactions\\test.txt'
            self.transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        f = open(self.txtpath)
        self.data = f.readlines()
        self.JPEGroot = 'D:\\data\\Willow\\willowactions\\JPEGImages\\'
        f.close()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, label = self.data[idx].split()
        img = self.JPEGroot + img + '.jpg'
        label = int(label)
        img = Image.open(img)
        return self.transform(img), label