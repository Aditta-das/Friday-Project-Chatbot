import pyaudio
import wave
import threading
import time
from threading import Thread
import torchaudio
import torch
import torch.nn as nn
from playsound import playsound
# class Listener:
# 	def __init__(self, sr=8000, recoding_time=2):
# 		self.chunk = 1024  # Record in chunks of 1024 samples
# 		self.recoding_time = recoding_time
# 		self.sample_format = pyaudio.paInt16  # 16 bits per sample
# 		self.p = pyaudio.PyAudio()  # Create an interface to PortAudio
# 		self.channels = 1
# 		self.sr = sr  # Record at 44100 samples per second
# 		self.stream = self.p.open(format=self.sample_format,
#                 channels=self.channels,
#                 rate=self.sr,
#                 frames_per_buffer=self.chunk,
#                 input=True,
#                 output=True)
	
# 	def save_wav(self, waveform, filename="audio.wav"):
# 		wf = wave.open(filename, 'wb')
# 		wf.setnchannels(1)
# 		wf.setsampwidth(self.p.get_sample_size(self.sr))
# 		wf.setframerate(fs)
# 		wf.writeframes(b''.join(waveform))
# 		wf.close()
# 		return filename

# 	def listen(self):
# 		frames = []
# 		while True:
# 			print("Wakeword Engine Running...")
# 			data = self.stream.read(self.chunk)
# 			time.sleep(0.1)
# 			frames.append(data)
# 		thread = threading.Thread(self.run, daemon=True)
# 		thread.start()
# 		return frames
	
# 	def run(self, frames):
# 		filename = self.save_wav(waveform=frames)
# 		waveform, sr = torchaudio.load(filename)
# 		audio_mono = torch.mean(waveform, dim=0, keepdim=True)
# 		tempData = torch.zeros([1, 160000])
# 		if audio_mono.numel() < 160000:
# 			tempData[:, :audio_mono.numel()] = audio_mono
# 		else:
# 			tempData = audio_mono[:, :160000]
# 		audio_mono=tempData
# 		mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
# 		mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
# 		mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
# 		#         print(f'mfcc {mfcc.size()}')
# 		mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
# 		new_feat = torch.cat([mel_specgram, mfcc], axis=1)

# 		data = torch.utils.data.DataLoader(new_feat.permute(0, 2, 1))
# 		model = torch.load("/content/model_epoch_4.pth.tar", map_location=torch.device("cpu"))["model"]
# 		new = torch.load("/content/model_epoch_4.pth.tar", map_location=torch.device("cpu"))["state_dict"]

# 		model.load_state_dict(new)
# 		model.eval().cpu()
# 		with torch.no_grad():
# 			for x in data:
# 				x = x.to("cpu")
# 				output, hidden_state = model(x, (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)))
# 				print(torch.round(torch.sigmoid(output)))
				

# a = Listener().listen()


import pyaudio
import wave

# model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torchaudio
import torch
import torch.nn as nn
from sklearn import model_selection
from sklearn import metrics
from tabulate import tabulate
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
class AudioLSTM(nn.Module):

    def __init__(self, n_feature=5, out_feature=1, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature

        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, out_feature)

    def forward(self, x, hidden):
        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x, hidden)

        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)

        # out.shape (batch, out_feature)
        out = self.fc(out[:, -1, :])

        # return the final output and the hidden state
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        return hidden




chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

waveform, sr = torchaudio.load("output.wav")
audio_mono = torch.mean(waveform, dim=0, keepdim=True)
tempData = torch.zeros([1, 160000])
if audio_mono.numel() < 160000:
    tempData[:, :audio_mono.numel()] = audio_mono
else:
    tempData = audio_mono[:, :160000]
audio_mono=tempData
mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
# #         print(f'mfcc {mfcc.size()}')
mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
new_feat = torch.cat([mel_specgram, mfcc], axis=1)

data = torch.utils.data.DataLoader(new_feat.permute(0, 2, 1))
model = torch.load("model_epoch_4.pth.tar", map_location=torch.device("cpu"))["model"]
new = torch.load("model_epoch_4.pth.tar", map_location=torch.device("cpu"))["state_dict"]

model.load_state_dict(new)
model.eval().cpu()
with torch.no_grad():
    for x in data:
        x = x.to("cpu")
        output, hidden_state = model(x, (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)))
        if torch.round(torch.sigmoid(output)) == 1.0:
            print("Play Response")
            playsound('Messenger.mp3')
        else: 
            print(torch.round(torch.sigmoid(output)))


