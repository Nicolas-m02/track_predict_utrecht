#%%
import SimpleITK as sitk
import numpy as np
import torch

import matplotlib.pyplot as plt
import os


prompt = sitk.GetArrayFromImage(sitk.ReadImage("/utrecht_exp/data/prompt.mha"))[0]

first_im = sitk.GetArrayFromImage(sitk.ReadImage("/utrecht_exp/data/test_images/ZZZ_HEARTCINE.MR.VIEWRAY_PHYSIK.0004.0005.2024.12.06.12.05.39.640625.73013714.mha"))[0]



plt.figure(figsize=(8,8))
plt.imshow(first_im, cmap='gray')
plt.contourf(prompt, colors=['r'], levels=[0.5, 1.5], alpha=0.5)
plt.contour(prompt, colors=['r'], levels=[0.5, 1.5])
plt.axis('off')
plt.show()

plt.figure(figsize=(8,8))
plt.imshow(prompt, cmap='gray')
plt.axis('off')
plt.show()
