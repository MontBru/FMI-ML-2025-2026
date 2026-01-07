from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt

train_transforms = transforms.Compose([
  transforms.ToTensor(), # convert the object into a tensor
  transforms.Resize((128, 128)), # resize the images to be of size 128x128
])

dataset_train = ImageFolder(
  'clouds/clouds_train',
  transform=train_transforms,
)

dataloader_train = data.DataLoader(
  dataset_train,
  shuffle=True,
  batch_size=6,
)

images, labels = next(iter(dataloader_train))
print(images.shape)

rows, cols = 2, 3
fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

for i, ax in enumerate(axes.flat):
    image = images[i].permute(1, 2, 0)
    ax.imshow(image)
    ax.axis("off")

plt.tight_layout()
plt.show()