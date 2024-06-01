# CLI

## 0. Install and global paths settings

```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
python -m venv venv
venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Then download the necessary models and the default TTS model, and set the global paths.
```bash
python initialize.py [--skip_default_models] [--dataset_root <path>] [--assets_root <path>]
```

Optional:
- `--skip_default_models`: Skip downloading the default voice models (use this if you only have to train your own models).
- `--dataset_root`: Default: `Data`. Root directory of the training dataset. The training dataset of `{model_name}` should be placed in `{dataset_root}/{model_name}`.
- `--assets_root`: Default: `model_assets`. Root directory of the model assets (for inference). In training, the model assets will be saved to `{assets_root}/{model_name}`, and in inference, we load all the models from `{assets_root}`.


## 1. Dataset preparation

### 1.1. Slice audio files

The following audio formats are supported: ".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a".
```bash
python slice.py --model_name <model_name> [-i <input_dir>] [-m <min_sec>] [-M <max_sec>] [--time_suffix]
```

Required:
- `model_name`: Name of the speaker (to be used as the name of the trained model).

Optional:
- `input_dir`: Path to the directory containing the audio files to slice (default: `inputs`)
- `min_sec`: Minimum duration of the sliced audio files in seconds (default: 2).
- `max_sec`: Maximum duration of the sliced audio files in seconds (default: 12).
- `--time_suffix`: Make the filename end with -start_ms-end_ms when saving wav.

### 1.2. Transcribe audio files

```bash
python transcribe.py --model_name <model_name>
```
Required:
- `model_name`: Name of the speaker (to be used as the name of the trained model).

Optional
- `--initial_prompt`: Initial prompt to use for the transcription (default value is specific to Japanese).
- `--device`: `cuda` or `cpu` (default: `cuda`).
- `--language`: `jp`, `en`, or `en` (default: `jp`).
- `--model`: Whisper model, default: `large-v3`
- `--compute_type`: default: `bfloat16`. Only used if not `--use_hf_whisper`.
- `--use_hf_whisper`: Use Hugging Face's whisper model instead of default faster-whisper (HF whisper is faster but requires more VRAM).
- `--batch_size`: Batch size (default: 16). Only used if `--use_hf_whisper`.
- `--num_beams`: Beam size (default: 1).
- `--no_repeat_ngram_size`: N-gram size for no repeat (default: 10).

## 2. Preprocess

```bash
python preprocess_all.py -m <model_name> [--use_jp_extra] [-b <batch_size>] [-e <epochs>] [-s <save_every_steps>] [--num_processes <num_processes>] [--normalize] [--trim] [--val_per_lang <val_per_lang>] [--log_interval <log_interval>] [--freeze_EN_bert] [--freeze_JP_bert] [--freeze_ZH_bert] [--freeze_style] [--freeze_decoder] [--yomi_error <yomi_error>]
```

Required:
- `model_name`: Name of the speaker (to be used as the name of the trained model).

Optional:
- `--batch_size`, `-b`: Batch size (default: 2).
- `--epochs`, `-e`: Number of epochs (default: 100).
- `--save_every_steps`, `-s`: Save every steps (default: 1000).
- `--num_processes`: Number of processes (default: half of the number of CPU cores).
- `--normalize`: Loudness normalize audio.
- `--trim`: Trim silence.
- `--freeze_EN_bert`: Freeze English BERT.
- `--freeze_JP_bert`: Freeze Japanese BERT.
- `--freeze_ZH_bert`: Freeze Chinese BERT.
- `--freeze_style`: Freeze style vector.
- `--freeze_decoder`: Freeze decoder.
- `--use_jp_extra`: Use JP-Extra model.
- `--val_per_lang`: Validation data per language (default: 0).
- `--log_interval`: Log interval (default: 200).
- `--yomi_error`: How to handle yomi errors (default: `raise`: raise an error after preprocessing all texts, `skip`: skip the texts with errors, `use`: use the texts with errors by ignoring unknown characters).

## 3. Train

Training settings are automatically loaded from the above process.

If NOT using JP-Extra model:
```bash
python train_ms.py [--repo_id <username>/<repo_name>]
```

If using JP-Extra model:
```bash
python train_ms_jp_extra.py [--repo_id <username>/<repo_name>] [--skip_default_style]
```

Optional:
- `--repo_id`: Hugging Face repository ID to upload the trained model to. You should have logged in using `huggingface-cli login` before running this command.
- `--skip_default_style`: Skip making the default style vector. Use this if you want to resume training (since the default style vector has been already made).
