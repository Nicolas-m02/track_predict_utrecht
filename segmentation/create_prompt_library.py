#%%


import math

import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import pymri
import time

emu_path = "/utrecht_data/20260323/tmp/"

handler = pymri.QueuedImageHandler()
recv = pymri.EmuImageReceiver.create(emu_path, handler)

all_images = []

print(len(os.listdir(emu_path)))



print("Receiving images...")
while True:
    image = handler.get_image()

    if image is not None:
        image_data = image['data']
        direction_cosines = math.degrees(math.atan2(image['row_direction_cosines'][1], image['row_direction_cosines'][0]))
        all_images.append((image_data, direction_cosines))
    
    if len(all_images) >= len(os.listdir(emu_path))-5:  # Stop after receiving 10 images for demonstration
        break



#%%

all_cosines = [np.round(cosines, 2) for _, cosines in all_images]
all_images_data = [np.round(data, 2) for data, _ in all_images]


print(np.unique(all_cosines))


known_angles = []  

os.makedirs("/utrecht_exp/segmentation/prompt_library/tmp_frontaal", exist_ok=True)

for i in range(len(all_cosines)):
    if all_cosines[i] not in known_angles:
        print('Detected new angle:', all_cosines[i])

        image = all_images_data[i]
        plt.imshow(image, cmap='gray')
        plt.title(f"Image with angle {all_cosines[i]}")
        plt.axis('off')
        plt.show()
        print(image.shape)

        sitk.WriteImage(sitk.GetImageFromArray(image), f"/utrecht_exp/segmentation/prompt_library/tmp_frontaal/image_{all_cosines[i]}.mha")


        known_angles.append(all_cosines[i])
    else:
        pass

#%%
print(len(all_cosines))
