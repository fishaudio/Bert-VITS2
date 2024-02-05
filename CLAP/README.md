###### [Overview](#CLAP) | [Setup](#Setup) | [CLAP weights](#CLAP-weights) | [Usage](#Usage) | [Examples](#Examples) | [Citation](#Citation)

# CLAP

CLAP (Contrastive Language-Audio Pretraining) is a model that learns acoustic concepts from natural language supervision and enables “Zero-Shot” inference. The model has been extensively evaluated in 26 audio downstream tasks achieving SoTA in several of them including classification, retrieval, and captioning.

<img width="832" alt="clap_diagrams" src="docs/clap2_diagram.png">

## Setup

First, install python 3.8 or higher (3.11 recommended). Then, install CLAP using either of the following:

```shell
# Install pypi pacakge
pip install msclap

# Or Install latest (unstable) git source
pip install git+https://github.com/microsoft/CLAP.git
```

## CLAP weights
CLAP weights are downloaded automatically (choose between versions _2022_, _2023_, and _clapcap_), but are also available at: [Zenodo](https://zenodo.org/record/8378278) or [HuggingFace](https://huggingface.co/microsoft/msclap)

_clapcap_ is the audio captioning model that uses the 2023 encoders.

## Usage

- Zero-Shot Classification and Retrieval
```python
from msclap import CLAP

# Load model (Choose between versions '2022' or '2023')
# The model weight will be downloaded automatically if `model_fp` is not specified
clap_model = CLAP(version = '2023', use_cuda=False)

# Extract text embeddings
text_embeddings = clap_model.get_text_embeddings(class_labels: List[str])

# Extract audio embeddings
audio_embeddings = clap_model.get_audio_embeddings(file_paths: List[str])

# Compute similarity between audio and text embeddings 
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
```

- Audio Captioning
```python
from msclap import CLAP

# Load model (Choose version 'clapcap')
clap_model = CLAP(version = 'clapcap', use_cuda=False)

# Generate audio captions
captions = clap_model.generate_caption(file_paths: List[str])
```

## Examples
Take a look at [examples](./examples/) for usage examples. 

To run Zero-Shot Classification on the ESC50 dataset try the following:

```bash
> cd examples && python zero_shot_classification.py
```
Output (version 2023)
```bash
ESC50 Accuracy: 93.9%
```

## Citation

Kindly cite our work if you find it useful.

[CLAP: Learning Audio Concepts from Natural Language Supervision](https://ieeexplore.ieee.org/abstract/document/10095889)
```
@inproceedings{CLAP2022,
  title={Clap learning audio concepts from natural language supervision},
  author={Elizalde, Benjamin and Deshmukh, Soham and Al Ismail, Mahmoud and Wang, Huaming},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

[Natural Language Supervision for General-Purpose Audio Representations](https://arxiv.org/abs/2309.05767)
```
@misc{CLAP2023,
      title={Natural Language Supervision for General-Purpose Audio Representations}, 
      author={Benjamin Elizalde and Soham Deshmukh and Huaming Wang},
      year={2023},
      eprint={2309.05767},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2309.05767}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
