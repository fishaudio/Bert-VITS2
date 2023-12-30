import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wav_dir", type=str, default="data/wav")

embs = []
names = []
for file in os.listdir(wav_dir):
    if file.endswith(".npy"):
        xvec = np.load(os.path.join(wav_dir, file))
        embs.append(np.expand_dims(xvec, axis=0))
        names.append(file)

x = np.concatenate(embs, axis=0)
x = np.squeeze(x)
