import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class UrbanSoundDataset(Dataset):

    #constructor
    def __init__(self, annotation_file, audio_dir, transformation, target_sample_rate):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    #len of the dataset
    def __len__(self):
        return len(self.annotations)

    #how to get item from our dataset
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        #load audio data
        signal, sr = torchaudio.load(audio_sample_path)
        #signal -> (num_channels, samples) -> example with 2 seconds of audio at 16k sr: (2,16000)
        #solve two problems: convert to mono if stereo and resample to 16000 hz
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        #extra step: trasform the waveform to a melspectrogram
        signal = self.transformation(signal)    #transform is a callable object, so we can pass to it directly the audio
        return signal, label

    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index,0] )
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index,6]

    """
    method to resample the signal 
    """
    def _resample_if_necessary(self, signal, sr):
        #if the sr is already 16000, don't do resampling
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    """
    method to obatin a mono signal from signals with multiple channels (stereo..)
    """
    def _mix_down_if_necessary(self, signal):
        #only when the signal is not mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal



if __name__ == "__main__":
    #path to the annotation file and to the folder with audio data:
    ANNOTATIONS_FILE = "/home/lorenzoncina/Documents/Machine_Learning/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/home/lorenzoncina/Documents/Machine_Learning/datasets/UrbanSound8K/audio"
    SAMPLE_RATE = 16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft= 1024,
        hop_length= 512,
        n_mels= 64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]

    a =1


