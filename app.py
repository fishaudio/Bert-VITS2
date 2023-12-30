import argparse
import datetime
import os
import sys
import warnings
import enum

import gradio as gr
import numpy as np
import torch
from gradio.processing_utils import convert_to_16_bit_wav
from typing import Dict, List

import utils
from config import config
from infer import get_net_g, infer
from tools.log import logger


class Languages(str, enum.Enum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


languages = [l.value for l in Languages]

DEFAULT_SDP_RATIO: float = 0.2
DEFAULT_NOISE: float = 0.6
DEFAULT_NOISEW: float = 0.8
DEFAULT_LENGTH: float = 1
DEFAULT_LINE_SPLIT: bool = True
DEFAULT_SPLIT_INTERVAL: float = 0.5
DEFAULT_STYLE_WEIGHT: float = 0.7
DEFAULT_EMOTION_WEIGHT: float = 1.0


class Model:
    def __init__(self, model_path, config_path, style_vec_path, device):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.style_vec_path = style_vec_path
        self.hps = utils.get_hparams_from_file(self.config_path)
        self.spk2id: Dict[str, int] = self.hps.data.spk2id
        self.id2spk: Dict[int, str] = {v: k for k, v in self.spk2id.items()}
        self.num_styles = self.hps.data.num_styles
        if hasattr(self.hps.data, "style2id"):
            self.style2id = self.hps.data.style2id
        else:
            self.style2id = {str(i): i for i in range(self.num_styles)}

        self.style_vectors = np.load(self.style_vec_path)
        self.net_g = None

    def load_net_g(self):
        self.net_g = get_net_g(
            model_path=self.model_path,
            version=self.hps.version,
            device=self.device,
            hps=self.hps,
        )

    def get_style_vector(self, style_id, weight=1.0):
        mean = self.style_vectors[0]
        style_vec = self.style_vectors[style_id]
        style_vec = mean + (style_vec - mean) * weight
        return style_vec

    def get_style_vector_from_audio(self, audio_path, weight=1.0):
        from style_gen import extract_style_vector

        xvec = extract_style_vector(audio_path)
        mean = self.style_vectors[0]
        xvec = mean + (xvec - mean) * weight
        return xvec

    def infer(
        self,
        text,
        language="JP",
        sid=0,
        reference_audio_path=None,
        sdp_ratio=DEFAULT_SDP_RATIO,
        noise=DEFAULT_NOISE,
        noisew=DEFAULT_NOISEW,
        length=DEFAULT_LENGTH,
        line_split=DEFAULT_LINE_SPLIT,
        split_interval=DEFAULT_SPLIT_INTERVAL,
        style_text="",
        style_weight=DEFAULT_STYLE_WEIGHT,
        use_style_text=False,
        style="0",
        emotion_weight=DEFAULT_EMOTION_WEIGHT,
    ):
        if reference_audio_path == "":
            reference_audio_path = None
        if style_text == "" or not use_style_text:
            style_text = None

        if self.net_g is None:
            self.load_net_g()
        if reference_audio_path is None:
            style_id = self.style2id[style]
            style_vector = self.get_style_vector(style_id, emotion_weight)
        else:
            style_vector = self.get_style_vector_from_audio(
                reference_audio_path, emotion_weight
            )
        if not line_split:
            with torch.no_grad():
                audio = infer(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noisew,
                    length_scale=length,
                    sid=sid,
                    language=language,
                    hps=self.hps,
                    net_g=self.net_g,
                    device=self.device,
                    style_text=style_text,
                    style_weight=style_weight,
                    style_vec=style_vector,
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
                            noise_scale_w=noisew,
                            length_scale=length,
                            sid=sid,
                            language=language,
                            hps=self.hps,
                            net_g=self.net_g,
                            device=self.device,
                            style_text=style_text,
                            style_weight=style_weight,
                            style_vec=style_vector,
                        )
                    )
                    if i != len(texts) - 1:
                        audios.append(np.zeros(int(44100 * split_interval)))
                audio = np.concatenate(audios)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = convert_to_16_bit_wav(audio)
        return (self.hps.data.sampling_rate, audio)


class ModelHolder:
    def __init__(self, root_dir, device):
        self.root_dir = root_dir
        self.device = device
        self.model_files_dict = {}
        self.current_model = None
        self.model_names = []
        self.models = []
        self.refresh()

    def refresh(self):
        self.model_files_dict: Dict[str, List[str]] = {}
        self.model_names: List[str] = []
        self.current_model = None
        model_dirs = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        for model_name in model_dirs:
            model_dir = os.path.join(self.root_dir, model_name)
            model_files = [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".safetensors")
            ]
            if len(model_files) == 0:
                logger.info(
                    f"No model files found in {self.root_dir}/{model_name}, so skip it"
                )
            self.model_files_dict[model_name] = model_files
            self.model_names.append(model_name)

    def load_model(self, model_name, model_path):
        if model_name not in self.model_files_dict:
            raise Exception(f"モデル名{model_name}は存在しません")
        if model_path not in self.model_files_dict[model_name]:
            raise Exception(f"pthファイル{model_path}は存在しません")
        self.current_model = Model(
            model_path=model_path,
            config_path=os.path.join(self.root_dir, model_name, "config.json"),
            style_vec_path=os.path.join(self.root_dir, model_name, "style_vectors.npy"),
            device=self.device,
        )
        styles = list(self.current_model.style2id.keys())
        return (
            gr.Dropdown(choices=styles, value=styles[0]),
            gr.update(interactive=True, value="音声合成"),
        )

    def update_model_files_dropdown(self, model_name):
        model_files = self.model_files_dict[model_name]
        return gr.Dropdown(choices=model_files, value=model_files[0])

    def update_model_names_dropdown(self):
        self.refresh()
        initial_model_name = self.model_names[0]
        initial_model_files = self.model_files_dict[initial_model_name]
        return (
            gr.Dropdown(choices=self.model_names, value=initial_model_name),
            gr.Dropdown(choices=initial_model_files, value=initial_model_files[0]),
            gr.update(interactive=False),  # For tts_button
        )


def tts_fn(
    text,
    language,
    reference_audio_path,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    line_split,
    split_interval,
    style_text,
    style_weight,
    use_style_text,
    emotion,
    emotion_weight,
):
    assert model_holder.current_model is not None

    start_time = datetime.datetime.now()

    sr, audio = model_holder.current_model.infer(
        text=text,
        language=language,
        reference_audio_path=reference_audio_path,
        sdp_ratio=sdp_ratio,
        noise=noise_scale,
        noisew=noise_scale_w,
        length=length_scale,
        line_split=line_split,
        split_interval=split_interval,
        style_text=style_text,
        style_weight=style_weight,
        use_style_text=use_style_text,
        style=emotion,
        emotion_weight=emotion_weight,
    )

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    return f"Success, time: {duration} seconds.", (sr, audio)


initial_text = "こんにちは、初めまして。あなたの名前はなんていうの？"

examples = [
    [initial_text, "JP"],
    [
        """あなたがそんなこと言うなんて、私はとっても嬉しい。
あなたがそんなこと言うなんて、私はとっても怒ってる。
あなたがそんなこと言うなんて、私はとっても驚いてる。
あなたがそんなこと言うなんて、私はとっても辛い。""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった告白セリフ
        """私、ずっと前からあなたのことを見てきました。あなたの笑顔、優しさ、強さに、心惹かれていたんです。
友達として過ごす中で、あなたのことがだんだんと特別な存在になっていくのがわかりました。
えっと、私、あなたのことが好きです！もしよければ、私と付き合ってくれませんか？""",
        "JP",
    ],
    [  # 夏目漱石『吾輩は猫である』
        """吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。なんでも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで始めて人間というものを見た。しかもあとで聞くと、それは書生という、人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕まえて煮て食うという話である。""",
        "JP",
    ],
    [  # 梶井基次郎『桜の樹の下には』
        """桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。
何故って、桜の花があんなにも見事に咲くなんて信じられないことじゃないか。俺はあの美しさが信じられないので、このにさんにち不安だった。
しかしいま、やっとわかるときが来た。桜の樹の下には屍体が埋まっている。これは信じていいことだ。""",
        "JP",
    ],
    [  # ChatGPTと考えた、感情を表すセリフ
        """やったー！テストで満点取れた！私とっても嬉しいな！
どうして私の意見を無視するの？許せない！ムカつく！あんたなんか死ねばいいのに。
あはははっ！この漫画めっちゃ笑える、見てよこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。""",
        "JP",
    ],
    [  # 上の丁寧語バージョン
        """やりました！テストで満点取れましたよ！私とっても嬉しいです！
どうして私の意見を無視するんですか？許せません！ムカつきます！あんたなんか死んでください。
あはははっ！この漫画めっちゃ笑えます、見てくださいこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しいです。""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった音声合成の説明文章
        """音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。
この分野の最新の研究成果を使うと、より自然で表現豊かな音声の生成が可能である。深層学習の応用により、感情やアクセントを含む声質の微妙な変化も再現することが出来る。""",
        "JP",
    ],
    [
        "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
        "EN",
    ],
    ["语音合成是人工制造人类语音。用于此目的的计算机系统称为语音合成器，可以通过软件或硬件产品实现。", "ZH"],
]

initial_md = """
# Style-Bert-VITS2 音声合成

注意: 初期からある[jvnvのモデル](https://huggingface.co/litagin/style_bert_vits2_jvnv)は、[JVNVコーパス（言語音声と非言語音声を持つ日本語感情音声コーパス）](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus)で学習されたモデルです。ライセンスは[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)です。
"""

how_to_md = """
下のように`model_assets`ディレクトリの中にモデルファイルたちを置いてください。
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
各モデルにはファイルたちが必要です：
- `config.json`：学習時の設定ファイル
- `*.safetensors`：学習済みモデルファイル（1つ以上が必要、複数可）
- `style_vectors.npy`：スタイルベクトルファイル

上2つは`Train.bat`による学習で自動的に正しい位置に保存されます。`style_vectors.npy`は`Style.bat`を実行して指示に従って生成してください。

TODO: 現在のところはspeaker_id = 0に固定しており複数話者の合成には対応していません。
"""

style_md = """
- プリセットまたは音声ファイルから読み上げの声音・感情・スタイルのようなものを制御できます。
- デフォルトのNeutralでも、十分に読み上げる文に応じた感情で感情豊かに読み上げられます。このスタイル制御は、それを重み付きで上書きするような感じです。
- 強さを大きくしすぎると発音が変になったり声にならなかったりと崩壊することがあります。
- どのくらいに強さがいいかはモデルやスタイルによって異なるようです。
- 音声ファイルを入力する場合は、学習データと似た声音の話者（特に同じ性別）でないとよい効果が出ないかもしれません。
"""


def make_interactive():
    return gr.update(interactive=True, value="音声合成")


def make_non_interactive():
    return gr.update(interactive=False, value="音声合成（モデルをロードしてください）")


def gr_util(item):
    if item == "プリセットから選ぶ":
        return (gr.update(visible=True), gr.Audio(visible=False, value=None))
    else:
        return (gr.update(visible=False), gr.update(visible=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.out_dir
    )
    args = parser.parse_args()
    model_dir = args.dir

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_holder = ModelHolder(model_dir, device)

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(f"モデルが見つかりませんでした。{model_dir}にモデルを置いてください。")
        sys.exit(1)
    initial_id = 0
    initial_pth_files = model_holder.model_files_dict[model_names[initial_id]]

    with gr.Blocks(theme="NoCrypt/miku") as app:
        gr.Markdown(initial_md)
        with gr.Accordion(label="使い方", open=False):
            gr.Markdown(how_to_md)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=3):
                        model_name = gr.Dropdown(
                            label="モデル一覧",
                            choices=model_names,
                            value=model_names[initial_id],
                        )
                        model_path = gr.Dropdown(
                            label="モデルファイル",
                            choices=initial_pth_files,
                            value=initial_pth_files[0],
                        )
                    refresh_button = gr.Button("更新", scale=1, visible=True)
                    load_button = gr.Button("ロード", scale=1, variant="primary")
                text_input = gr.TextArea(label="テキスト", value=initial_text)

                line_split = gr.Checkbox(label="改行で分けて生成", value=DEFAULT_LINE_SPLIT)
                split_interval = gr.Slider(
                    minimum=0.0,
                    maximum=2,
                    value=DEFAULT_SPLIT_INTERVAL,
                    step=0.1,
                    label="分けた場合に挟む無音の長さ（秒）",
                )
                language = gr.Dropdown(choices=languages, value="JP", label="Language")
                with gr.Accordion(label="詳細設定", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0, maximum=1, value=DEFAULT_SDP_RATIO, step=0.1, label="SDP Ratio"
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1, maximum=2, value=DEFAULT_NOISE, step=0.1, label="Noise"
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1, maximum=2, value=DEFAULT_NOISEW, step=0.1, label="Noise_W"
                    )
                    length_scale = gr.Slider(
                        minimum=0.1, maximum=2, value=DEFAULT_LENGTH, step=0.1, label="Length"
                    )
                    use_style_text = gr.Checkbox(label="Style textを使う", value=False)
                    style_text = gr.Textbox(
                        label="Style text",
                        placeholder="どうして私の意見を無視するの？許せない、ムカつく！死ねばいいのに。",
                        info="このテキストの読み上げと似た声音・感情になりやすくなります。ただ抑揚やテンポ等が犠牲になる傾向があります。",
                        visible=False,
                    )
                    style_text_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_STYLE_WEIGHT,
                        step=0.1,
                        label="Style textの強さ",
                        visible=False,
                    )
                    use_style_text.change(
                        lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                        inputs=[use_style_text],
                        outputs=[style_text, style_text_weight],
                    )
            with gr.Column():
                with gr.Accordion("スタイルについて詳細", open=False):
                    gr.Markdown(style_md)
                style_mode = gr.Radio(
                    ["プリセットから選ぶ", "音声ファイルを入力"],
                    label="スタイルの指定方法",
                    value="プリセットから選ぶ",
                )
                style = gr.Dropdown(
                    label="スタイル（Neutralが平均スタイル）",
                    choices=["モデルをロードしてください"],
                    value="モデルをロードしてください",
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=DEFAULT_EMOTION_WEIGHT,
                    step=0.1,
                    label="スタイルの強さ",
                )
                ref_audio_path = gr.Audio(label="参照音声", type="filepath", visible=False)
                tts_button = gr.Button(
                    "音声合成（モデルをロードしてください）", variant="primary", interactive=False
                )
                text_output = gr.Textbox(label="情報")
                audio_output = gr.Audio(label="結果")
                with gr.Accordion("テキスト例", open=False):
                    gr.Examples(examples, inputs=[text_input, language])

        tts_button.click(
            tts_fn,
            inputs=[
                text_input,
                language,
                ref_audio_path,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                line_split,
                split_interval,
                style_text,
                style_text_weight,
                use_style_text,
                style,
                style_weight,
            ],
            outputs=[text_output, audio_output],
        )

        model_name.change(
            model_holder.update_model_files_dropdown,
            inputs=[model_name],
            outputs=[model_path],
        )

        model_path.change(make_non_interactive, outputs=[tts_button])

        refresh_button.click(
            model_holder.update_model_names_dropdown,
            outputs=[model_name, model_path, tts_button],
        )

        load_button.click(
            model_holder.load_model,
            inputs=[model_name, model_path],
            outputs=[style, tts_button],
        )

        style_mode.change(
            gr_util,
            inputs=[style_mode],
            outputs=[style, ref_audio_path],
        )

    app.launch(inbrowser=True)
