import argparse
import datetime
import json
import os
import sys
from typing import Optional

import gradio as gr
import torch
import yaml

from common.constants import (
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
from common.log import logger
from common.tts_model import ModelHolder
from infer import InvalidToneError
from text.japanese import g2kata_tone, kata_tone2phone_tone, text_normalize

# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # dataset_root = path_config["dataset_root"]
    assets_root = path_config["assets_root"]

languages = [l.value for l in Languages]


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
    assist_text,
    assist_text_weight,
    use_assist_text,
    style,
    style_weight,
    kata_tone_json_str,
    use_tone,
    speaker,
):
    assert model_holder.current_model is not None

    wrong_tone_message = ""
    kata_tone: Optional[list[tuple[str, int]]] = None
    if use_tone and kata_tone_json_str != "":
        if language != "JP":
            logger.warning("Only Japanese is supported for tone generation.")
            wrong_tone_message = "アクセント指定は現在日本語のみ対応しています。"
        if line_split:
            logger.warning("Tone generation is not supported for line split.")
            wrong_tone_message = "アクセント指定は改行で分けて生成を使わない場合のみ対応しています。"
        try:
            kata_tone = []
            json_data = json.loads(kata_tone_json_str)
            # tupleを使うように変換
            for kana, tone in json_data:
                assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                kata_tone.append((kana, tone))
        except Exception as e:
            logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
            wrong_tone_message = f"アクセント指定が不正です: {e}"
            kata_tone = None

    # toneは実際に音声合成に代入される際のみnot Noneになる
    tone: Optional[list[int]] = None
    if kata_tone is not None:
        phone_tone = kata_tone2phone_tone(kata_tone)
        tone = [t for _, t in phone_tone]

    speaker_id = model_holder.current_model.spk2id[speaker]

    start_time = datetime.datetime.now()

    try:
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
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=use_assist_text,
            style=style,
            style_weight=style_weight,
            given_tone=tone,
            sid=speaker_id,
        )
    except InvalidToneError as e:
        logger.error(f"Tone error: {e}")
        return f"Error: アクセント指定が不正です:\n{e}", None, kata_tone_json_str

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    if tone is None and language == "JP":
        # アクセント指定に使えるようにアクセント情報を返す
        norm_text = text_normalize(text)
        kata_tone = g2kata_tone(norm_text)
        kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
    elif tone is None:
        kata_tone_json_str = ""
    message = f"Success, time: {duration} seconds."
    if wrong_tone_message != "":
        message = wrong_tone_message + "\n" + message
    return message, (sr, audio), kata_tone_json_str


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
吾輩はここで初めて人間というものを見た。しかもあとで聞くと、それは書生という、人間中で一番獰悪な種族であったそうだ。
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

style_md = f"""
- プリセットまたは音声ファイルから読み上げの声音・感情・スタイルのようなものを制御できます。
- デフォルトの{DEFAULT_STYLE}でも、十分に読み上げる文に応じた感情で感情豊かに読み上げられます。このスタイル制御は、それを重み付きで上書きするような感じです。
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
        "--dir", "-d", type=str, help="Model directory", default=assets_root
    )
    parser.add_argument(
        "--share", action="store_true", help="Share this app publicly", default=False
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Server name for Gradio app",
    )
    parser.add_argument(
        "--no-autolaunch",
        action="store_true",
        default=False,
        help="Do not launch app automatically",
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
                    label="改行ごとに挟む無音の長さ（秒）",
                )
                line_split.change(
                    lambda x: (gr.Slider(visible=x)),
                    inputs=[line_split],
                    outputs=[split_interval],
                )
                tone = gr.Textbox(
                    label="アクセント調整（数値は 0=低 か1=高 のみ）",
                    info="改行で分けない場合のみ使えます。万能ではありません。",
                )
                use_tone = gr.Checkbox(label="アクセント調整を使う", value=False)
                use_tone.change(
                    lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
                    inputs=[use_tone],
                    outputs=[line_split],
                )
                language = gr.Dropdown(choices=languages, value="JP", label="Language")
                speaker = gr.Dropdown(label="話者")
                with gr.Accordion(label="詳細設定", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_SDP_RATIO,
                        step=0.1,
                        label="SDP Ratio",
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISE,
                        step=0.1,
                        label="Noise",
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISEW,
                        step=0.1,
                        label="Noise_W",
                    )
                    length_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_LENGTH,
                        step=0.1,
                        label="Length",
                    )
                    use_assist_text = gr.Checkbox(label="Assist textを使う", value=False)
                    assist_text = gr.Textbox(
                        label="Assist text",
                        placeholder="どうして私の意見を無視するの？許せない、ムカつく！死ねばいいのに。",
                        info="このテキストの読み上げと似た声音・感情になりやすくなります。ただ抑揚やテンポ等が犠牲になる傾向があります。",
                        visible=False,
                    )
                    assist_text_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_ASSIST_TEXT_WEIGHT,
                        step=0.1,
                        label="Assist textの強さ",
                        visible=False,
                    )
                    use_assist_text.change(
                        lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                        inputs=[use_assist_text],
                        outputs=[assist_text, assist_text_weight],
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
                    label=f"スタイル（{DEFAULT_STYLE}が平均スタイル）",
                    choices=["モデルをロードしてください"],
                    value="モデルをロードしてください",
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=DEFAULT_STYLE_WEIGHT,
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
                assist_text,
                assist_text_weight,
                use_assist_text,
                style,
                style_weight,
                tone,
                use_tone,
                speaker,
            ],
            outputs=[text_output, audio_output, tone],
        )

        model_name.change(
            model_holder.update_model_files_gr,
            inputs=[model_name],
            outputs=[model_path],
        )

        model_path.change(make_non_interactive, outputs=[tts_button])

        refresh_button.click(
            model_holder.update_model_names_gr,
            outputs=[model_name, model_path, tts_button],
        )

        load_button.click(
            model_holder.load_model_gr,
            inputs=[model_name, model_path],
            outputs=[style, tts_button, speaker],
        )

        style_mode.change(
            gr_util,
            inputs=[style_mode],
            outputs=[style, ref_audio_path],
        )

    app.launch(
        inbrowser=not args.no_autolaunch, share=args.share, server_name=args.server_name
    )
