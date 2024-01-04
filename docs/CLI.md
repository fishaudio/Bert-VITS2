# CLI

**WIP**

## Dataset

`Dataset.bat` webui (`python webui_dataset.py`) consists of **slice audio** and **transcribe wavs**.

### Slice audio

```bash
python slice.py -i <input_dir> -o <output_dir> -m <min_sec> -M <max_sec>
```

Required:
- `input_dir`: Path to the directory containing the audio files to slice.
- `output_dir`: Path to the directory where the sliced audio files will be saved.

Optional:
- `min_sec`: Minimum duration of the sliced audio files in seconds (default 2).
- `max_sec`: Maximum duration of the sliced audio files in seconds (default 12).

### Transcribe wavs

```bash
python transcribe.py -i <input_dir> -o <output_file> --speaker_name <speaker_name>
```

Required:
- `input_dir`: Path to the directory containing the audio files to transcribe.
- `output_file`: Path to the file where the transcriptions will be saved.
- `speaker_name`: Name of the speaker.

Optional
- `--initial_prompt`: Initial prompt to use for the transcription (default value is specific to Japanese).
- `--device`: `cuda` or `cpu` (default: `cuda`).
- `--language`: `jp`, `en`, or `en` (default: `jp`).
- `--model`: Whisper model, default: `large-v3`
- `--compute_type`: default: `bfloat16`

## Train

`Train.bat` webui (`python webui_train.py`) consists of the following.

### Preprocess audio
```bash
python resample.py -i <input_dir> -o <output_dir> [--normalize] [--trim]
```

Required:
- `input_dir`: Path to the directory containing the audio files to preprocess.
- `output_dir`: Path to the directory where the preprocessed audio files will be saved.

TO BE WRITTEN (WIP)

これいる？
