import os
import cv2
from tqdm.auto import tqdm

class Cifar100:
    def __init__(self, data_dir='./DATA/CIFARv2Mini', phase='train'):
        self.data_dir = data_dir
        self.phase = phase

    def load_images(self):
        images = []
        image_names = []
        image_labels = []
        print(f"Loading Cifar-100 dataset from dir : {self.data_dir}/{phase} !")
        data_folder_path = os.path.join(self.data_dir, self.phase)
        for i, sub_folder in tqdm(enumerate(os.listdir(data_folder_path))):
            sub_folder_path = os.path.join(data_folder_path, sub_folder)
            for img_name in os.listdir(sub_folder_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(sub_folder_path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)

                    img_name = f"{self.phase}/{sub_folder}/{img_name}"
                    image_names.append(img_name)

                    image_labels.append(i)

        return images, image_names, image_labels

# loader = Cifar100()
# images, image_names, image_labels = loader.load_images()

# print(len(images))
# print(len(image_names))
# print(type(images[0]))
# print(images[0].shape)
# print(image_names[100])

