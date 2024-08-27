from torch.utils.data import Dataset
import os
from utils import *
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor() # 转换为张量数据类型
])

class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.name1 = os.listdir(os.path.join(path, 'image2'))  # os.listdir可以获取该文件夹下的所有路径,所以self.name中保存的是每一张图片的名称
        self.name2 = os.listdir(os.path.join(path, 'mask2'))                                                      
    
    def __len__(self):
        return len(self.name1)
    
    def __getitem__(self, index):
        segment_name1 = self.name1[index]  # 拿到每一张图片的名称xx.png
        segment_name2 = self.name2[index]  
        
        
        segment_path = os.path.join(self.path, 'image2', segment_name1)
        image_path = os.path.join(self.path, 'mask2', segment_name2)
        
        segment_image = keep_image_size_open(image_path)
        image = keep_image_size_open(segment_path)

        return transform(image), transform(segment_image)


if __name__ == '__main__':
    data = MyDataset('/home/zhouzhilin/Code_Myself/U_Net/Dataset/duke')    
    print(data.__len__())
    print(data[0][0].shape)
    print(data[0][1].shape)