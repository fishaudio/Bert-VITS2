import utils
from data_utils import TextAudioSpeakerLoader
from tqdm import tqdm

config_path = 'configs/fzh.json'
hps = utils.get_hparams_from_file(config_path)

train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)

for _ in tqdm(train_dataset):
    pass
for _ in tqdm(eval_dataset):
    pass