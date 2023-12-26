import argparse
import os

import gradio as gr
import numpy as np
import torch
import warnings
from gradio.processing_utils import convert_to_16_bit_wav

import utils
from infer import get_net_g, infer
from tools.log import logger
from config import config

is_hf_spaces = os.getenv("SYSTEM") == "spaces"
limit = 100


class Model:
    def __init__(self, model_path, config_path, style_vec_path, device):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.style_vec_path = style_vec_path
        self.load()

    def load(self):
        self.hps = utils.get_hparams_from_file(self.config_path)
        self.spk2id = self.hps.data.spk2id
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
        sdp_ratio=0.2,
        noise=0.6,
        noisew=0.8,
        length=1.0,
        line_split=True,
        split_interval=0.2,
        style_text="",
        style_weight=0.7,
        use_style_text=False,
        style="0",
        emotion_weight=1.0,
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
        self.model_files_dict = {}
        self.model_names = []
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
    if is_hf_spaces and len(text) > limit:
        raise Exception(f"文字数が{limit}文字を超えています")

    assert model_holder.current_model is not None

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
    return "Success", (sr, audio)


initial_text = "こんにちは、初めまして。あなたの名前はなんていうの？"

example_local = [
    [initial_text, "JP"],
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
        """やったー！テストで満点取れたよ！私とっても嬉しいな！
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

example_hf_spaces = [
    [initial_text, "JP"],
    ["えっと、私、あなたのことが好きです！もしよければ付き合ってくれませんか？", "JP"],
    ["吾輩は猫である。名前はまだ無い。", "JP"],
    ["どこで生れたかとんと見当がつかぬ。なんでも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。", "JP"],
    ["やったー！テストで満点取れたよ！私とっても嬉しいな！", "JP"],
    ["どうして私の意見を無視するの？許せない！ムカつく！あんたなんか死ねばいいのに。", "JP"],
    ["あはははっ！この漫画めっちゃ笑える、見てよこれ、ふふふ、あはは。", "JP"],
    ["あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。", "JP"],
    ["深層学習の応用により、感情やアクセントを含む声質の微妙な変化も再現されている。", "JP"],
]

initial_md = """
# Bert-VITS2 okiba TTS デモ

[bert_vits2_okiba](https://huggingface.co/litagin/bert_vits2_okiba) のモデルのデモです。
モデル名は[rvc_okiba](https://huggingface.co/litagin/rvc_okiba)のモデル名と対応しています。
モデルは随時追加していきます。現在のモデルはすべてBert-VITS2のver 2.1のものです。

**定形サンプルは[こちら](https://huggingface.co/litagin/bert_vits2_okiba/blob/main/examples.md)から聴くほうが速いです。**

- huggingfaceのcpuで動くので、何故かやたら遅いことが多かったりなんか不安定で動かないときもあるみたいです。
- huggingface上では最大100文字にしています。
- Style textの実装あたりで本家の内部コードを改造しているので、このapp.pyをそのまま本家に使っても今のところは動きません。

現在のところはspeaker_id = 0に固定しています。
"""


def make_interactive():
    return gr.update(interactive=True, value="音声合成")


def make_non_interactive():
    return gr.update(interactive=False, value="音声合成（モデルをロードしてください）")


def gr_util(item):
    if item == "クラスタから選ぶ":
        return (gr.update(visible=True), gr.update(visible=False))
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

    languages = ["JP", "EN", "ZH"]
    examples = example_hf_spaces if is_hf_spaces else example_local

    model_names = model_holder.model_names
    initial_id = 1 if is_hf_spaces else 0
    initial_pth_files = model_holder.model_files_dict[model_names[initial_id]]

    with gr.Blocks(theme="NoCrypt/miku") as app:
        gr.Markdown(initial_md)
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
                    refresh_button = gr.Button(
                        "モデル一覧を更新", scale=1, visible=not is_hf_spaces
                    )
                    load_button = gr.Button("モデルをロード", scale=1)
                text_input = gr.TextArea(label="テキスト", value=initial_text)
                use_style_text = gr.Checkbox(label="Style textを使う", value=False)
                style_text = gr.Textbox(
                    label="Style text",
                    placeholder="どうして私の意見を無視するの？許せない、ムカつく！死ねばいいのに。",
                    info="このテキストの読み上げと似た声音・感情になりやすくなります。ただ抑揚やテンポ等が犠牲になるかも。",
                    visible=False,
                )
                style_text_weight = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.7,
                    step=0.1,
                    label="Style textの強さ",
                    visible=False,
                )
                use_style_text.change(
                    lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                    inputs=[use_style_text],
                    outputs=[style_text, style_text_weight],
                )

                line_split = gr.Checkbox(label="改行で分けて生成", value=True)
                split_interval = gr.Slider(
                    minimum=0.1, maximum=2, value=0.5, step=0.1, label="分けた場合に挟む無音の長さ"
                )
                language = gr.Dropdown(choices=languages, value="JP", label="Language")
                with gr.Accordion(label="詳細設定", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0, maximum=1, value=0.2, step=0.1, label="SDP Ratio"
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise"
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1, maximum=2, value=0.8, step=0.1, label="Noise_W"
                    )
                    length_scale = gr.Slider(
                        minimum=0.1, maximum=2, value=1.0, step=0.1, label="Length"
                    )
            with gr.Column():
                style_mode = gr.Radio(
                    ["クラスタから選ぶ", "音声ファイルを入力"],
                    label="スタイルの指定方法",
                    value="クラスタから選ぶ",
                )
                style = gr.Dropdown(
                    label="スタイル（0が平均スタイル）", choices=list(range(7)), value=0
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=1,
                    step=0.1,
                    label="スタイルの強さ",
                )
                ref_audio_path = gr.Audio(label="参照音声", type="filepath", visible=False)
                tts_button = gr.Button(
                    "音声合成（モデルをロードしてください）", variant="primary", interactive=False
                )
                text_output = gr.Textbox(label="情報")
                audio_output = gr.Audio(label="結果")
                gr.Examples(examples, inputs=[text_input, language], label="テキスト例")

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
