import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UTKFaceClassificationDataset(Dataset):
    """Dataset für Altersklassifikation"""
    
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.file_list = []
        self.age_labels = []
        
        # Altersklassen definieren
        def age_to_class(age):
            if age <= 12:
                return 0
            elif age <= 19:
                return 1
            elif age <= 29:
                return 2
            elif age <= 49:
                return 3
            else:
                return 4
        
        # Dateien laden
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    age = int(f.split('_')[0])
                    if 0 <= age <= 116:
                        self.file_list.append(f)
                        self.age_labels.append(age_to_class(age))
                except:
                    continue
        
        print(f"Loaded {len(self.file_list)} images")
        print(f"Class distribution: {np.bincount(self.age_labels)}")
        
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.file_list[idx])
        age_class = self.age_labels[idx]  # Integer von 0-4
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, age_class
