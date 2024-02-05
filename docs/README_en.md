# This English README is for 1.x versions. WIP for 2.x versions.

# Style-Bert-VITS2

Bert-VITS2 with more controllable voice styles.

https://github.com/litagin02/Style-Bert-VITS2/assets/139731664/b907c1b8-43aa-46e6-b03f-f6362f5a5a1e

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)

Online demo: https://huggingface.co/spaces/litagin/Style-Bert-VITS2-JVNV

This repository is based on [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1, so many thanks to the original author!

- [Update History](docs/CHANGELOG.md)

**Overview**

- Based on Bert-VITS2 v2.1, which generates emotionally rich voices from entered text, this version allows free control of emotions and speaking styles, including intensity.
- Easy to install and train for people without Git or Python (for Windows users), much is borrowed from [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2/). Training on Google Colab is also supported: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)
- If used only for voice synthesis, it can operate on CPU without a graphics card.
- Also includes an API server for integration with others (PR by [@darai0512](https://github.com/darai0512), thank you).
- Originally, Bert-VITS2's strength was to read "happy text happily, sad text sadly", so even without using the added style specification in this fork, you can generate emotionally rich voices.


## How to Use

<!-- For more details, please refer to [here](docs/tutorial.md). -->

### Operating Environment

We have confirmed the operation in Windows Command Prompt, WSL2, and Linux (Ubuntu Desktop) for each UI and API Server (please be creative with path specifications in WSL).

### Installation

#### For Those Unfamiliar with Git or Python

Assuming Windows:

1. Download and unzip [this zip file](https://github.com/litagin02/Style-Bert-VITS2/releases/download/1.3/Style-Bert-VITS2.zip).
   - If you have a graphics card, double-click `Install-Style-Bert-VITS2.bat`.
   - If you don't have a graphics card, double-click `Install-Style-Bert-VITS2-CPU.bat`.
2. Wait for the necessary environment to install automatically.
3. After that, if the WebUI for voice synthesis launches automatically, the installation is successful. The default model will be downloaded, so you can play with it immediately.

For updates, please double-click `Update-Style-Bert-VITS2.bat`.

#### For Those Familiar with Git and Python

```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
python -m venv venv
venv\Scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python initialize.py  # Download necessary models and default TTS model
```
Don't forget the last step.

### Voice Synthesis
Double-click `App.bat` or run `python app.py` to launch the WebUI. The default model is downloaded during installation, so you can use it even if you haven't trained it.

The structure of the model files required for voice synthesis is as follows (you don't need to place them manually):

```
model_assets
├── your_model
│   ├── config.json
│   ├── your_model_file1.safetensors
│   ├── your_model_file2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── another_model
    ├── ...
```

For inference, `config.json`, `*.safetensors`, and `style_vectors.npy` are necessary. If you want to share a model, please share these three files.

Among them, `style_vectors.npy` is a file necessary to control the style. By default, the average style "Neutral" is generated during training.
If you want to use multiple styles for more detailed control, please refer to "Generating Styles" below (even with only the average style, if the training data is emotionally rich, sufficiently emotionally rich voices can be generated).

### Training

Double-click Train.bat or run `python webui_train.py` to launch the WebUI.

### Generating Styles
For those who want to use styles other than the default "Neutral".

- Double-click `Style.bat` or run `python webui_style_vectors.py` to launch the WebUI.
- It is independent of training, so you can do it even during training, and you can redo it any number of times after training is complete (preprocessing must be finished).
- For more details on the specifications of the style, please refer to [clustering.ipynb](../clustering.ipynb).

### Dataset Creation

- Double-click `Dataset.bat` or run `python webui_dataset.py` to launch the WebUI for creating datasets from audio files. You can use this tool to learn from audio files only.

Note: If you want to manually correct the dataset, remove noise, etc., you may find [Aivis](https://github.com/tsukumijima/Aivis) or its Windows-compatible dataset part [Aivis Dataset](https://github.com/litagin02/Aivis-Dataset) useful. However, if there are many files, etc., it may be sufficient to simply cut out and create a dataset with this tool.

Please experiment to see what kind of dataset is best.

### API Server
Run `python server_fastapi.py` in the constructed environment to launch the API server.
Please check the API specification after launching at `/docs`.

By default, CORS settings are allowed for all domains.
As much as possible, change the value of server.origins in `config.yml` and limit it to trusted domains (if you delete the key, you can disable the CORS settings).

### Merging
You can create a new model by mixing two models in terms of "voice", "emotional expression", and "tempo".
Double-click `Merge.bat` or run `python webui_merge.py` to launch the WebUI.

## Relation to Bert-VITS2 v2.1
Basically, it's just a slight modification of the Bert-VITS2 v2.1 model structure. The [pre-trained model](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) is also essentially the same as Bert-VITS2 v2.1 (unnecessary weights have been removed and converted to safetensors).

The differences are as follows:

- Like [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2), it is easy to use even for people who do not know Python or Git.
- Changed the model for emotional embedding (from 1024-dimensional [wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) to 256-dimensional [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM), which is more for speaker identification than emotional embedding)
- Removed vector quantization from embeddings and replaced it with just a fully connected layer.
- By creating a style vector file `style_vectors.npy`, you can generate voices using that style and continuously specify the strength of the effect.
- Various WebUIs created
- Support for bf16 training
- Support for safetensors format, defaulting to using safetensors
- Other minor bug fixes and refactoring