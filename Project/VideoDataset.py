import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
import random
import numpy as np
import os

class VideoFolderDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_frames=16,
        transform=None,
        device = 'cpu'
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.device = device

        self.samples = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            class_path = os.path.join(root_dir, cls)
            if not os.path.isdir(class_path):
                continue

            self.class_to_idx[cls] = idx

            for fname in os.listdir(class_path):
                if fname.endswith(".mp4"):
                    self.samples.append(
                        (os.path.join(class_path, fname), idx)
                    )
        self.classes = [None] * len(self.class_to_idx)

        for class_name, idx in self.class_to_idx.items():
            self.classes[idx] = class_name


    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video):
        T = video.shape[0]

        if T >= self.num_frames:
            idxs = np.linspace(0, T - 1, self.num_frames).astype(int)
        else:
            idxs = np.pad(
                np.arange(T),
                (0, self.num_frames - T),
                mode="edge"
            )

        return video[idxs]

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video = VideoDecoder(video_path, device=self.device)
        video = video[0:-1]
        video = self._sample_frames(video)
        video = video.float() / 255.0


        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])

        return video, label
