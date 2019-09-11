from torchvision import models
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
import cv2
from cv2 import *

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

# transformations needed
trf = T.Compose([T.Resize(224), # originally 256 and then cropped
                 #T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])


def cv2_to_pil(img): #Since you want to be able to use Pillow (PIL)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Define the helper function


def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


capture = cv2.VideoCapture(0)
while (True):
    ret, frame = capture.read()
    cv2.imshow('camera', frame)

    pil_img = cv2_to_pil(frame)  # convert the image to PIL so you can use it that way.
    inp = trf(pil_img).unsqueeze(0)
    out = fcn(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    rgb = cv2.resize(rgb, (frame.shape[1], frame.shape[0]))

    cv2.imshow('segmented', rgb)

    if cv2.waitKey(1) == 27:
        break
capture.release()
cv2.destroyAllWindows()


