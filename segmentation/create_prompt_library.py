#%%


import math

import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import pymri
import time

emu_path = "/utrecht_data/20260323/tmp/"
emu_path = "/utrecht_exp/data/single_file/20260622/"

handler = pymri.QueuedImageHandler()
recv = pymri.EmuImageReceiver.create(emu_path, handler)

#MRTC Receiver configs
#mrtc_port = 4005 # receiving images from MR
#stack_update_host = '0.0.0.0'
#stack_update_port = 54323   # controlling the MR

#handler = pymri.QueuedImageHandler()
#recv = pymri.MRTCImageReceiver.create(mrtc_port, stack_update_host, stack_update_port, handler, False) 



all_images = []



os.makedirs("/utrecht_exp/segmentation/prompt_library/fiducial_img", exist_ok=True)

print("Receiving images...")
counter = 0

all_images_data = []
all_cosines = []
latest = None

while True:
    image = handler.get_image()

    if image is not None:
        image_data = image['data']
        direction_cosines = int(np.round(math.degrees(math.atan2(image['row_direction_cosines'][1], image['row_direction_cosines'][0]))))
        all_images.append((image_data, direction_cosines))
    
    if len(all_images) >= len(os.listdir(emu_path)):  # Stop after receiving 10 images for demonstration
        break



#%%

all_cosines = [cosines for _, cosines in all_images]
all_images_data = [data for data, _ in all_images]


print(np.unique(all_cosines))


known_angles = []  

os.makedirs("/utrecht_exp/segmentation/prompt_library/20260622", exist_ok=True)

for i in range(len(all_cosines)):
    if all_cosines[i] not in known_angles:
        print('Detected new angle:', all_cosines[i])

        image = all_images_data[i]
        plt.imshow(image, cmap='gray')
        plt.title(f"Image with angle {all_cosines[i]}")
        plt.axis('off')
        print(image.shape)

        sitk.WriteImage(sitk.GetImageFromArray(image), f"/utrecht_exp/segmentation/prompt_library/20260622/image_{all_cosines[i]}.mha")


        known_angles.append(all_cosines[i])
    else:
        pass

#%%

import matplotlib.pyplot as plt
import SimpleITK as sitk


im = sitk.GetArrayFromImage(sitk.ReadImage("/utrecht_exp/segmentation/prompt_library/fiducial_test/image_90.mha"))

plt.figure()
plt.imshow(im, cmap="gray")

