
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class UrbanSoundDataset(Dataset):

    #constructor
    def __init__(self, annotation_file, audio_dir):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir

    #len of the dataset
    def __len__(self):
        return len(self.annotations)

    #how to get item from our dataset
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index,0] )
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index,6]


if __name__ == "__main__":
    #path to the annotation file and to the folder with audio data:
    ANNOTATIONS_FILE = "/home/lorenzoncina/Documents/Machine_Learning/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/home/lorenzoncina/Documents/Machine_Learning/datasets/UrbanSound8K/audio"
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]


