import os
import cv2
from tqdm.auto import tqdm

class Cifar100:
    def __init__(self, data_dir='../DATA/CIFARv2', phase='train'):
        self.data_dir = data_dir
        self.phase = phase

    def load_images(self):
        images = []
        image_names = []
        print(f"Loading Cifar-100 dataset from dir : {self.data_dir} !")
        data_folder_path = os.path.join(self.data_dir, self.phase)
        for sub_folder in tqdm(os.listdir(data_folder_path)):
            sub_folder_path = os.path.join(data_folder_path, sub_folder)
            for img_name in os.listdir(sub_folder_path):
                img_path = os.path.join(sub_folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

                img_name = f"{self.phase}/{sub_folder}/{img_name}"
                image_names.append(img_name)
        return images, image_names

loader = Cifar100()
images, image_names = loader.load_images()

print(len(images))
print(len(image_names))
print(type(images[0]))
print(images[0].shape)
print(image_names[100])

