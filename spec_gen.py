import utils
from data_utils import TextAudioSpeakerLoader
from tqdm import tqdm

config_path = 'configs/config.json'
hps = utils.get_hparams_from_file(config_path)

train_dataset = TextAudioSpeakerLoader("filelists/train.list", hps.data)
eval_dataset = TextAudioSpeakerLoader("filelists/val.list", hps.data)

for _ in tqdm(train_dataset):
    pass
for _ in tqdm(eval_dataset):
    pass