import pytest
from scipy.io import wavfile

from style_bert_vits2.constants import BASE_DIR
from style_bert_vits2.tts_model import TTSModelHolder


def synthesize(device: str = 'cpu'):

    # モデル一覧を取得
    model_holder = TTSModelHolder(BASE_DIR / 'model_assets', device)

    # モデルが存在する場合、音声合成を実行
    if len(model_holder.models_info) > 0:

        # jvnv-F1-jp モデルを探す
        for model_info in model_holder.models_info:
            if model_info['name'] == 'jvnv-F1-jp':

                # 音声合成を実行
                model = model_holder.get_model(model_info['name'], model_info['files'][0])
                model.load()
                sample_rate, audio_data = model.infer("あらゆる現実を、すべて自分のほうへねじ曲げたのだ。")

                # 音声データを保存
                with open(BASE_DIR / 'tests/test.wav', mode='wb') as f:
                    wavfile.write(f, sample_rate, audio_data)
    else:
        pytest.skip("音声合成モデルが見つかりませんでした。")


def test_synthesize_cpu():
    synthesize(device='cpu')
    assert (BASE_DIR / 'tests/test.wav').exists()


def test_synthesize_cuda():
    synthesize(device='cuda')
    assert (BASE_DIR / 'tests/test.wav').exists()
