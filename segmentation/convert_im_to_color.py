#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = '/utrecht_exp/sample_mr_image.png'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(f"Original image shape: {image.shape}, dtype: {image.dtype}")

image = image.astype(np.float32)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
image = image.astype(np.uint8)
prep_image = image.copy()

# prep_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

print(f"Prepared image shape: {prep_image.shape}, dtype: {prep_image.dtype}")



plt.imshow(prep_image)  
plt.show()