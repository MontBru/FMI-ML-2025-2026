import streamlit as st
from torchvision import transforms
import tempfile
from torchcodec.decoders import VideoDecoder

import torch
import torchcodec
import numpy as np

image_transforms = transforms.Compose([
  # transforms.ToTensor(),
  transforms.Resize((256, 256)), # resize the images
  transforms.CenterCrop(224),
  transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    #Normalizing the data to be zero-centered.
  #Initially I normalized it by using the
  #dataset mean and std, but I later changed it
  #to use the ImageNet normalization,
  #so that i can use the same data if i try
  #fine-tuning a model.
  ])

def _sample_frames(num_frames, video):
  T = video.shape[0]

  if T >= num_frames:
      idxs = np.linspace(0, T - 1, num_frames).astype(int)
  else:
      idxs = np.pad(
          np.arange(T),
          (0, num_frames - T),
          mode="edge"
      )

  return video[idxs]

from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
import torch.nn as nn

model.classifier[6] = nn.Linear(
    in_features=4096,
    out_features=22
)

model.load_state_dict(torch.load("model_05.pth", map_location=device))

classes = ['barbell biceps curl', 'bench press', 'chest fly machine', 'deadlift', 'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press', 'lat pulldown', 'lateral raise', 'leg extension', 'leg raises', 'plank', 'pull Up', 'push-up', 'romanian deadlift', 'russian twist', 'shoulder press', 'squat', 't bar row', 'tricep Pushdown', 'tricep dips']


st.write(
    """# Drop a video of a training set and see what exercise you're doing""")

uploaded_file = st.file_uploader("", "mp4")
if uploaded_file is not None:
    # 1. Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(uploaded_file)

    st.success("Video uploaded successfully!")

    # 2. Decode video using TorchCodec
    decoder = VideoDecoder(video_path)
    video = decoder[0:-1]
    video = _sample_frames(16, video)
    video = video.float() / 255.0

    video = torch.stack([image_transforms(frame) for frame in video])

    # 3. Predict the execise
    net = model.to(device)

    pred = ""

    with torch.no_grad():
      features = video
      features = features.squeeze()
      features = features.to(device)

      outputs = net(features)
      preds = torch.argmax(outputs, dim=1)
      mean_outputs = torch.mean(outputs, axis=0)
      pred = torch.argmax(mean_outputs)
      pred = classes[pred]


    st.success(pred)

