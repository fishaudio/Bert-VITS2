from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g, infer
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.voice import adjust_voice


# Gradio の import は重いため、ここでは型チェック時のみ import する
# ライブラリとしての利用を考慮し、TTSModelHolder の _for_gradio() 系メソッド以外では Gradio に依存しないようにする
# _for_gradio() 系メソッドの戻り値の型アノテーションを文字列としているのは、Gradio なしで実行できるようにするため
# if TYPE_CHECKING:
#     import gradio as gr


class TTSModel:
    """
    Style-Bert-Vits2 の音声合成モデルを操作するクラス。
    モデル/ハイパーパラメータ/スタイルベクトルのパスとデバイスを指定して初期化し、model.infer() メソッドを呼び出すと音声合成を行える。
    """

    def __init__(
        self,
        model_path: Path,
        config_path: Union[Path, HyperParameters],
        style_vec_path: Union[Path, NDArray[Any]],
        device: str,
    ) -> None:
        """
        Style-Bert-Vits2 の音声合成モデルを初期化する。
        この時点ではモデルはロードされていない (明示的にロードしたい場合は model.load() を呼び出す)。

        Args:
            model_path (Path): モデル (.safetensors) のパス
            config_path (Union[Path, HyperParameters]): ハイパーパラメータ (config.json) のパス (直接 HyperParameters を指定することも可能)
            style_vec_path (Union[Path, NDArray[Any]]): スタイルベクトル (style_vectors.npy) のパス (直接 NDArray を指定することも可能)
            device (str): 音声合成時に利用するデバイス (cpu, cuda, mps など)
        """

        self.model_path: Path = model_path
        self.device: str = device

        # ハイパーパラメータの Pydantic モデルが直接指定された
        if isinstance(config_path, HyperParameters):
            self.config_path: Path = Path("")  # 互換性のため空の Path を設定
            self.hyper_parameters: HyperParameters = config_path
        # ハイパーパラメータのパスが指定された
        else:
            self.config_path: Path = config_path
            self.hyper_parameters: HyperParameters = HyperParameters.load_from_json(
                self.config_path
            )

        # スタイルベクトルの NDArray が直接指定された
        if isinstance(style_vec_path, np.ndarray):
            self.style_vec_path: Path = Path("")  # 互換性のため空の Path を設定
            self.__style_vectors: NDArray[Any] = style_vec_path
        # スタイルベクトルのパスが指定された
        else:
            self.style_vec_path: Path = style_vec_path
            self.__style_vectors: NDArray[Any] = np.load(self.style_vec_path)

        self.spk2id: dict[str, int] = self.hyper_parameters.data.spk2id
        self.id2spk: dict[int, str] = {v: k for k, v in self.spk2id.items()}

        num_styles: int = self.hyper_parameters.data.num_styles
        if hasattr(self.hyper_parameters.data, "style2id"):
            self.style2id: dict[str, int] = self.hyper_parameters.data.style2id
        else:
            self.style2id: dict[str, int] = {str(i): i for i in range(num_styles)}
        if len(self.style2id) != num_styles:
            raise ValueError(
                f"Number of styles ({num_styles}) does not match the number of style2id ({len(self.style2id)})"
            )

        if self.__style_vectors.shape[0] != num_styles:
            raise ValueError(
                f"The number of styles ({num_styles}) does not match the number of style vectors ({self.__style_vectors.shape[0]})"
            )
        self.__style_vector_inference: Optional[Any] = None

        self.__net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra, None] = None

    def load(self) -> None:
        """
        音声合成モデルをデバイスにロードする。
        """
        self.__net_g = get_net_g(
            model_path=str(self.model_path),
            version=self.hyper_parameters.version,
            device=self.device,
            hps=self.hyper_parameters,
        )

    def __get_style_vector(self, style_id: int, weight: float = 1.0) -> NDArray[Any]:
        """
        スタイルベクトルを取得する。

        Args:
            style_id (int): スタイル ID (0 から始まるインデックス)
            weight (float, optional): スタイルベクトルの重み. Defaults to 1.0.

        Returns:
            NDArray[Any]: スタイルベクトル
        """
        mean = self.__style_vectors[0]
        style_vec = self.__style_vectors[style_id]
        style_vec = mean + (style_vec - mean) * weight
        return style_vec

    def __get_style_vector_from_audio(
        self, audio_path: str, weight: float = 1.0
    ) -> NDArray[Any]:
        """
        音声からスタイルベクトルを推論する。

        Args:
            audio_path (str): 音声ファイルのパス
            weight (float, optional): スタイルベクトルの重み. Defaults to 1.0.
        Returns:
            NDArray[Any]: スタイルベクトル
        """

        if self.__style_vector_inference is None:

            # pyannote.audio は scikit-learn などの大量の重量級ライブラリに依存しているため、
            # TTSModel.infer() に reference_audio_path を指定し音声からスタイルベクトルを推論する場合のみ遅延 import する
            try:
                import pyannote.audio
            except ImportError:
                raise ImportError(
                    "pyannote.audio is required to infer style vector from audio"
                )

            # スタイルベクトルを取得するための推論モデルを初期化
            self.__style_vector_inference = pyannote.audio.Inference(
                model=pyannote.audio.Model.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM"
                ),
                window="whole",
            )
            self.__style_vector_inference.to(torch.device(self.device))

        # 音声からスタイルベクトルを推論
        xvec = self.__style_vector_inference(audio_path)
        mean = self.__style_vectors[0]
        xvec = mean + (xvec - mean) * weight
        return xvec

    def __convert_to_16_bit_wav(self, data: NDArray[Any]) -> NDArray[Any]:
        """
        音声データを 16-bit int 形式に変換する。
        gradio.processing_utils.convert_to_16_bit_wav() を移植したもの。

        Args:
            data (NDArray[Any]): 音声データ

        Returns:
            NDArray[Any]: 16-bit int 形式の音声データ
        """
        # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
        if data.dtype in [np.float64, np.float32, np.float16]:  # type: ignore
            data = data / np.abs(data).max()
            data = data * 32767
            data = data.astype(np.int16)
        elif data.dtype == np.int32:
            data = data / 65536
            data = data.astype(np.int16)
        elif data.dtype == np.int16:
            pass
        elif data.dtype == np.uint16:
            data = data - 32768
            data = data.astype(np.int16)
        elif data.dtype == np.uint8:
            data = data * 257 - 32768
            data = data.astype(np.int16)
        elif data.dtype == np.int8:
            data = data * 256
            data = data.astype(np.int16)
        else:
            raise ValueError(
                "Audio data cannot be converted automatically from "
                f"{data.dtype} to 16-bit int format."
            )
        return data

    def infer(
        self,
        text: str,
        language: Languages = Languages.JP,
        speaker_id: int = 0,
        reference_audio_path: Optional[str] = None,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        line_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text: bool = False,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
        pitch_scale: float = 1.0,
        intonation_scale: float = 1.0,
    ) -> tuple[int, NDArray[Any]]:
        """
        テキストから音声を合成する。

        Args:
            text (str): 読み上げるテキスト
            language (Languages, optional): 言語. Defaults to Languages.JP.
            speaker_id (int, optional): 話者 ID. Defaults to 0.
            reference_audio_path (Optional[str], optional): 音声スタイルの参照元の音声ファイルのパス. Defaults to None.
            sdp_ratio (float, optional): DP と SDP の混合比。0 で DP のみ、1で SDP のみを使用 (値を大きくするとテンポに緩急がつく). Defaults to DEFAULT_SDP_RATIO.
            noise (float, optional): DP に与えられるノイズ. Defaults to DEFAULT_NOISE.
            noise_w (float, optional): SDP に与えられるノイズ. Defaults to DEFAULT_NOISEW.
            length (float, optional): 生成音声の長さ（話速）のパラメータ。大きいほど生成音声が長くゆっくり、小さいほど短く早くなる。 Defaults to DEFAULT_LENGTH.
            line_split (bool, optional): テキストを改行ごとに分割して生成するかどうか (True の場合 given_phone/given_tone は無視される). Defaults to DEFAULT_LINE_SPLIT.
            split_interval (float, optional): 改行ごとに分割する場合の無音 (秒). Defaults to DEFAULT_SPLIT_INTERVAL.
            assist_text (Optional[str], optional): 感情表現の参照元の補助テキスト. Defaults to None.
            assist_text_weight (float, optional): 感情表現の補助テキストを適用する強さ. Defaults to DEFAULT_ASSIST_TEXT_WEIGHT.
            use_assist_text (bool, optional): 音声合成時に感情表現の補助テキストを使用するかどうか. Defaults to False.
            style (str, optional): 音声スタイル (Neutral, Happy など). Defaults to DEFAULT_STYLE.
            style_weight (float, optional): 音声スタイルを適用する強さ. Defaults to DEFAULT_STYLE_WEIGHT.
            given_phone (Optional[list[int]], optional): 読み上げテキストの読みを表す音素列。指定する場合は given_tone も別途指定が必要. Defaults to None.
            given_tone (Optional[list[int]], optional): アクセントのトーンのリスト. Defaults to None.
            pitch_scale (float, optional): ピッチの高さ (1.0 から変更すると若干音質が低下する). Defaults to 1.0.
            intonation_scale (float, optional): 抑揚の平均からの変化幅 (1.0 から変更すると若干音質が低下する). Defaults to 1.0.

        Returns:
            tuple[int, NDArray[Any]]: サンプリングレートと音声データ (16bit PCM)
        """

        logger.info(f"Start generating audio data from text:\n{text}")
        if language != "JP" and self.hyper_parameters.version.endswith("JP-Extra"):
            raise ValueError(
                "The model is trained with JP-Extra, but the language is not JP"
            )
        if reference_audio_path == "":
            reference_audio_path = None
        if assist_text == "" or not use_assist_text:
            assist_text = None

        if self.__net_g is None:
            self.load()
        assert self.__net_g is not None
        if reference_audio_path is None:
            style_id = self.style2id[style]
            style_vector = self.__get_style_vector(style_id, style_weight)
        else:
            style_vector = self.__get_style_vector_from_audio(
                reference_audio_path, style_weight
            )
        if not line_split:
            with torch.no_grad():
                audio = infer(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noise_w,
                    length_scale=length,
                    sid=speaker_id,
                    language=language,
                    hps=self.hyper_parameters,
                    net_g=self.__net_g,
                    device=self.device,
                    assist_text=assist_text,
                    assist_text_weight=assist_text_weight,
                    style_vec=style_vector,
                    given_phone=given_phone,
                    given_tone=given_tone,
                )
        else:
            texts = text.split("\n")
            texts = [t for t in texts if t != ""]
            audios = []
            with torch.no_grad():
                for i, t in enumerate(texts):
                    audios.append(
                        infer(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noise_w,
                            length_scale=length,
                            sid=speaker_id,
                            language=language,
                            hps=self.hyper_parameters,
                            net_g=self.__net_g,
                            device=self.device,
                            assist_text=assist_text,
                            assist_text_weight=assist_text_weight,
                            style_vec=style_vector,
                        )
                    )
                    if i != len(texts) - 1:
                        audios.append(np.zeros(int(44100 * split_interval)))
                audio = np.concatenate(audios)
        logger.info("Audio data generated successfully")
        if not (pitch_scale == 1.0 and intonation_scale == 1.0):
            _, audio = adjust_voice(
                fs=self.hyper_parameters.data.sampling_rate,
                wave=audio,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )
        audio = self.__convert_to_16_bit_wav(audio)
        return (self.hyper_parameters.data.sampling_rate, audio)


class TTSModelInfo(BaseModel):
    name: str
    files: list[str]
    styles: list[str]
    speakers: list[str]


class TTSModelHolder:
    """
    Style-Bert-Vits2 の音声合成モデルを管理するクラス。
    model_holder.models_info から指定されたディレクトリ内にある音声合成モデルの一覧を取得できる。
    """

    def __init__(self, model_root_dir: Path, device: str) -> None:
        """
        Style-Bert-Vits2 の音声合成モデルを管理するクラスを初期化する。
        音声合成モデルは下記のように配置されていることを前提とする (.safetensors のファイル名は自由) 。
        ```
        model_root_dir
        ├── model-name-1
        │   ├── config.json
        │   ├── model-name-1_e160_s14000.safetensors
        │   └── style_vectors.npy
        ├── model-name-2
        │   ├── config.json
        │   ├── model-name-2_e160_s14000.safetensors
        │   └── style_vectors.npy
        └── ...
        ```

        Args:
            model_root_dir (Path): 音声合成モデルが配置されているディレクトリのパス
            device (str): 音声合成時に利用するデバイス (cpu, cuda, mps など)
        """

        self.root_dir: Path = model_root_dir
        self.device: str = device
        self.model_files_dict: dict[str, list[Path]] = {}
        self.current_model: Optional[TTSModel] = None
        self.model_names: list[str] = []
        self.models_info: list[TTSModelInfo] = []
        self.refresh()

    def refresh(self) -> None:
        """
        音声合成モデルの一覧を更新する。
        """

        self.model_files_dict = {}
        self.model_names = []
        self.current_model = None
        self.models_info = []

        model_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for model_dir in model_dirs:
            model_files = [
                f
                for f in model_dir.iterdir()
                if f.suffix in [".pth", ".pt", ".safetensors"]
            ]
            if len(model_files) == 0:
                logger.warning(f"No model files found in {model_dir}, so skip it")
                continue
            config_path = model_dir / "config.json"
            if not config_path.exists():
                logger.warning(
                    f"Config file {config_path} not found, so skip {model_dir}"
                )
                continue
            self.model_files_dict[model_dir.name] = model_files
            self.model_names.append(model_dir.name)
            hyper_parameters = HyperParameters.load_from_json(config_path)
            style2id: dict[str, int] = hyper_parameters.data.style2id
            styles = list(style2id.keys())
            spk2id: dict[str, int] = hyper_parameters.data.spk2id
            speakers = list(spk2id.keys())
            self.models_info.append(
                TTSModelInfo(
                    name=model_dir.name,
                    files=[str(f) for f in model_files],
                    styles=styles,
                    speakers=speakers,
                )
            )

    def get_model(self, model_name: str, model_path_str: str) -> TTSModel:
        """
        指定された音声合成モデルのインスタンスを取得する。
        この時点ではモデルはロードされていない (明示的にロードしたい場合は model.load() を呼び出す)。

        Args:
            model_name (str): 音声合成モデルの名前
            model_path_str (str): 音声合成モデルのファイルパス (.safetensors)

        Returns:
            TTSModel: 音声合成モデルのインスタンス
        """

        model_path = Path(model_path_str)
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")
        if model_path not in self.model_files_dict[model_name]:
            raise ValueError(f"Model file `{model_path}` is not found")
        if self.current_model is None or self.current_model.model_path != model_path:
            self.current_model = TTSModel(
                model_path=model_path,
                config_path=self.root_dir / model_name / "config.json",
                style_vec_path=self.root_dir / model_name / "style_vectors.npy",
                device=self.device,
            )

        return self.current_model

    def get_model_for_gradio(self, model_name: str, model_path_str: str):
        import gradio as gr

        model_path = Path(model_path_str)
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")
        if model_path not in self.model_files_dict[model_name]:
            raise ValueError(f"Model file `{model_path}` is not found")
        if (
            self.current_model is not None
            and self.current_model.model_path == model_path
        ):
            # Already loaded
            speakers = list(self.current_model.spk2id.keys())
            styles = list(self.current_model.style2id.keys())
            return (
                gr.Dropdown(choices=styles, value=styles[0]),  # type: ignore
                gr.Button(interactive=True, value="音声合成"),
                gr.Dropdown(choices=speakers, value=speakers[0]),  # type: ignore
            )
        self.current_model = TTSModel(
            model_path=model_path,
            config_path=self.root_dir / model_name / "config.json",
            style_vec_path=self.root_dir / model_name / "style_vectors.npy",
            device=self.device,
        )
        speakers = list(self.current_model.spk2id.keys())
        styles = list(self.current_model.style2id.keys())
        return (
            gr.Dropdown(choices=styles, value=styles[0]),  # type: ignore
            gr.Button(interactive=True, value="音声合成"),
            gr.Dropdown(choices=speakers, value=speakers[0]),  # type: ignore
        )

    def update_model_files_for_gradio(self, model_name: str):
        import gradio as gr

        model_files = [str(f) for f in self.model_files_dict[model_name]]
        return gr.Dropdown(choices=model_files, value=model_files[0])  # type: ignore

    def update_model_names_for_gradio(
        self,
    ):
        import gradio as gr

        self.refresh()
        initial_model_name = self.model_names[0]
        initial_model_files = [
            str(f) for f in self.model_files_dict[initial_model_name]
        ]
        return (
            gr.Dropdown(choices=self.model_names, value=initial_model_name),  # type: ignore
            gr.Dropdown(choices=initial_model_files, value=initial_model_files[0]),  # type: ignore
            gr.Button(interactive=False),  # For tts_button
        )
