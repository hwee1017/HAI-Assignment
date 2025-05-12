from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
image1 = Image.open("C:/Users/user/OneDrive/바탕 화면/HAI-Assignment/lecture05/image1.jpg")
image2 = Image.open("C:/Users/user/OneDrive/바탕 화면/HAI-Assignment/lecture05/image2.jpg")
transform = transforms.ToTensor()
image_tensor1 = transform(image1)
image_tensor2 = transform(image2)
plt.imshow(image_tensor1.permute(1,2,0))
plt.show()
plt.imshow(image_tensor2.permute(1,2,0))
plt.show()
print(image_tensor1.shape)
print(image_tensor1.dtype)
print(image_tensor2.shape)
print(image_tensor2.dtype)
res1 = image_tensor1 * image_tensor2
plt.imshow(res1.permute(1,2,0))
plt.show()
res2 = image_tensor1.view(3,-1).matmul(image_tensor2.view(3,-1).T)
print(f"the matrix multiplication of image1 and image2 is \n{res2}")