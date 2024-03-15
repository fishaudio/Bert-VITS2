import pytest
from scipy.io import wavfile

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.tts_model import TTSModelHolder


def synthesize(device: str = "cpu"):

    # 音声合成モデルが配置されていれば、音声合成を実行
    model_holder = TTSModelHolder(BASE_DIR / "model_assets", device)
    if len(model_holder.models_info) > 0:

        # jvnv-F2-jp モデルを探す
        for model_info in model_holder.models_info:
            if model_info.name == "jvnv-F2-jp":
                # すべてのスタイルに対して音声合成を実行
                for style in model_info.styles:

                    # 音声合成を実行
                    model = model_holder.get_model(model_info.name, model_info.files[0])
                    model.load()
                    sample_rate, audio_data = model.infer(
                        "あらゆる現実を、すべて自分のほうへねじ曲げたのだ。",
                        # 言語 (JP, EN, ZH / JP-Extra モデルの場合は JP のみ)
                        language=Languages.JP,
                        # 話者 ID (音声合成モデルに複数の話者が含まれる場合のみ必須、単一話者のみの場合は 0)
                        speaker_id=0,
                        # 感情表現の強さ (0.0 〜 1.0)
                        sdp_ratio=0.4,
                        # スタイル (Neutral, Happy など)
                        style=style,
                        # スタイルの強さ (0.0 〜 100.0)
                        style_weight=6.0,
                    )

                    # 音声データを保存
                    (BASE_DIR / "tests/wavs").mkdir(exist_ok=True, parents=True)
                    wav_file_path = BASE_DIR / f"tests/wavs/{style}.wav"
                    with open(wav_file_path, "wb") as f:
                        wavfile.write(f, sample_rate, audio_data)

                    # 音声データが保存されたことを確認
                    assert wav_file_path.exists()
                    # wav_file_path.unlink()
    else:
        pytest.skip("音声合成モデルが見つかりませんでした。")


def test_synthesize_cpu():
    synthesize(device="cpu")


# Windows環境ではtorchのcudaが簡単に入らないため、テストをスキップ
# def test_synthesize_cuda():
#     synthesize(device="cuda")
