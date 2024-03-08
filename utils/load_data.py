import os
import cv2

def get_img(datadir):
    data = []
    names = []
    try:
        img_names = os.listdir(datadir)
        for i in img_names:
            imgdir = datadir + i
            try:
                img = cv2.imread(imgdir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img, (200,200))
                data.append(img)
                names.append(i)
            except AttributeError:
                print('No image')
    except AttributeError:
        print('No folder')

    return data, names