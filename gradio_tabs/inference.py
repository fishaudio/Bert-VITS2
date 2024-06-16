import datetime
import json
from typing import Optional

import gradio as gr

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
    GRADIO_THEME,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.infer import InvalidToneError
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.tts_model import TTSModelHolder


# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# Web UI での学習時の無駄な GPU VRAM 消費を避けるため、あえてここでは BERT モデルの事前ロードを行わない
# データセットの BERT 特徴量は事前に bert_gen.py により抽出されているため、学習時に BERT モデルをロードしておく必要はない
# BERT モデルの事前ロードは「ロード」ボタン押下時に実行される TTSModelHolder.get_model_for_gradio() 内で行われる
# Web UI での学習時、音声合成タブの「ロード」ボタンを押さなければ、BERT モデルが VRAM にロードされていない状態で学習を開始できる

languages = [lang.value for lang in Languages]

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
    [
        "语音合成是人工制造人类语音。用于此目的的计算机系统称为语音合成器，可以通过软件或硬件产品实现。",
        "ZH",
    ],
]

initial_md = """
- Ver 2.5で追加されたデフォルトの [`koharune-ami`（小春音アミ）モデル](https://huggingface.co/litagin/sbv2_koharune_ami) と[`amitaro`（あみたろ）モデル](https://huggingface.co/litagin/sbv2_amitaro) は、[あみたろの声素材工房](https://amitaro.net/)で公開されているコーパス音源・ライブ配信音声を利用して事前に許可を得て学習したモデルです。下記の**利用規約を必ず読んで**からご利用ください。

- Ver 2.5のアップデート後に上記モデルをダウンロードするには、`Initialize.bat`をダブルクリックするか、手動でダウンロードして`model_assets`ディレクトリに配置してください。

- Ver 2.3で追加された**エディター版**のほうが実際に読み上げさせるには使いやすいかもしれません。`Editor.bat`か`python server_editor.py --inbrowser`で起動できます。
"""

terms_of_use_md = """
## お願いとデフォルトモデルのライセンス

最新のお願い・利用規約は [こちら](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md) を参照してください。常に最新のものが適用されます。

Style-Bert-VITS2を用いる際は、以下のお願いを守っていただけると幸いです。ただしモデルの利用規約以前の箇所はあくまで「お願い」であり、何の強制力はなく、Style-Bert-VITS2の利用規約ではありません。よって[リポジトリのライセンス](https://github.com/litagin02/Style-Bert-VITS2#license)とは矛盾せず、リポジトリの利用にあたっては常にリポジトリのライセンスのみが拘束力を持ちます。

### やってほしくないこと

以下の目的での利用はStyle-Bert-VITS2を使ってほしくありません。

- 法律に違反する目的
- 政治的な目的（本家Bert-VITS2で禁止されています）
- 他者を傷つける目的
- なりすまし・ディープフェイク作成目的

### 守ってほしいこと

- Style-Bert-VITS2を利用する際は、使用するモデルの利用規約・ライセンス必ず確認し、存在する場合はそれに従ってほしいです。
- またソースコードを利用する際は、[リポジトリのライセンス](https://github.com/litagin02/Style-Bert-VITS2#license)に従ってほしいです。

以下はデフォルトで付随しているモデルのライセンスです。

### JVNVコーパス (jvnv-F1-jp, jvnv-F2-jp, jvnv-M1-jp, jvnv-M2-jp)

- [JVNVコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) のライセンスは[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)ですので、これを継承します。

### 小春音アミ (koharune-ami) / あみたろ (amitaro)

[あみたろの声素材工房様の規約](https://amitaro.net/voice/voice_rule/) と [あみたろのライブ配信音声・利用規約](https://amitaro.net/voice/livevoice/#index_id6) を全て守らなければなりません。特に、以下の事項を遵守してください（規約を守れば商用非商用問わず利用できます）：

#### 禁止事項

- 年齢制限のある作品・用途への使用
- 新興宗教・政治・マルチ購などに深く関係する作品・用途
- 特定の団体や個人や国家を誹謗中傷する作品・用途
- 生成された音声を、あみたろ本人の声として扱うこと
- 生成された音声を、あみたろ以外の人の声として扱うこと

#### クレジット表記

生成音声を公開する際は（媒体は問わない）、必ず分かりやすい場所に `あみたろの声素材工房 (https://amitaro.net/)` の声を元にした音声モデルを使用していることが分かるようなクレジット表記を記載してください。

クレジット表記例:
- `Style-BertVITS2モデル: 小春音アミ、あみたろの声素材工房 (https://amitaro.net/)`
- `Style-BertVITS2モデル: あみたろ、あみたろの声素材工房 (https://amitaro.net/)`

#### モデルマージ

モデルマージに関しては、[あみたろの声素材工房のよくある質問への回答](https://amitaro.net/voice/faq/#index_id17)を遵守してください：
- 本モデルを別モデルとマージできるのは、その別モデル作成の際に学習に使われた声の権利者が許諾している場合に限る
- あみたろの声の特徴が残っている場合（マージの割合が25%以上の場合）は、その利用は[あみたろの声素材工房様の規約](https://amitaro.net/voice/voice_rule/)の範囲内に限定され、そのモデルに関してもこの規約が適応される
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


def create_inference_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def tts_fn(
        model_name,
        model_path,
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
        pitch_scale,
        intonation_scale,
    ):
        model_holder.get_model(model_name, model_path)
        assert model_holder.current_model is not None

        wrong_tone_message = ""
        kata_tone: Optional[list[tuple[str, int]]] = None
        if use_tone and kata_tone_json_str != "":
            if language != "JP":
                logger.warning("Only Japanese is supported for tone generation.")
                wrong_tone_message = "アクセント指定は現在日本語のみ対応しています。"
            if line_split:
                logger.warning("Tone generation is not supported for line split.")
                wrong_tone_message = (
                    "アクセント指定は改行で分けて生成を使わない場合のみ対応しています。"
                )
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
                noise_w=noise_scale_w,
                length=length_scale,
                line_split=line_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=use_assist_text,
                style=style,
                style_weight=style_weight,
                given_tone=tone,
                speaker_id=speaker_id,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )
        except InvalidToneError as e:
            logger.error(f"Tone error: {e}")
            return f"Error: アクセント指定が不正です:\n{e}", None, kata_tone_json_str
        except ValueError as e:
            logger.error(f"Value error: {e}")
            return f"Error: {e}", None, kata_tone_json_str

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        if tone is None and language == "JP":
            # アクセント指定に使えるようにアクセント情報を返す
            norm_text = normalize_text(text)
            kata_tone = g2kata_tone(norm_text)
            kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
        elif tone is None:
            kata_tone_json_str = ""
        message = f"Success, time: {duration} seconds."
        if wrong_tone_message != "":
            message = wrong_tone_message + "\n" + message
        return message, (sr, audio), kata_tone_json_str

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(
            f"モデルが見つかりませんでした。{model_holder.root_dir}にモデルを置いてください。"
        )
        with gr.Blocks() as app:
            gr.Markdown(
                f"Error: モデルが見つかりませんでした。{model_holder.root_dir}にモデルを置いてください。"
            )
        return app
    initial_id = 0
    initial_pth_files = [
        str(f) for f in model_holder.model_files_dict[model_names[initial_id]]
    ]

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(initial_md)
        gr.Markdown(terms_of_use_md)
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
                pitch_scale = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=1,
                    step=0.05,
                    label="音高(1以外では音質劣化)",
                )
                intonation_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    step=0.1,
                    label="抑揚(1以外では音質劣化)",
                )

                line_split = gr.Checkbox(
                    label="改行で分けて生成（分けたほうが感情が乗ります）",
                    value=DEFAULT_LINE_SPLIT,
                )
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
                    use_assist_text = gr.Checkbox(
                        label="Assist textを使う", value=False
                    )
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
                    maximum=20,
                    value=DEFAULT_STYLE_WEIGHT,
                    step=0.1,
                    label="スタイルの強さ（声が崩壊したら小さくしてください）",
                )
                ref_audio_path = gr.Audio(
                    label="参照音声", type="filepath", visible=False
                )
                tts_button = gr.Button(
                    "音声合成（モデルをロードしてください）",
                    variant="primary",
                    interactive=False,
                )
                text_output = gr.Textbox(label="情報")
                audio_output = gr.Audio(label="結果")
                with gr.Accordion("テキスト例", open=False):
                    gr.Examples(examples, inputs=[text_input, language])

        tts_button.click(
            tts_fn,
            inputs=[
                model_name,
                model_path,
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
                pitch_scale,
                intonation_scale,
            ],
            outputs=[text_output, audio_output, tone],
        )

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        model_path.change(make_non_interactive, outputs=[tts_button])

        refresh_button.click(
            model_holder.update_model_names_for_gradio,
            outputs=[model_name, model_path, tts_button],
        )

        load_button.click(
            model_holder.get_model_for_gradio,
            inputs=[model_name, model_path],
            outputs=[style, tts_button, speaker],
        )

        style_mode.change(
            gr_util,
            inputs=[style_mode],
            outputs=[style, ref_audio_path],
        )

    return app


if __name__ == "__main__":
    from config import get_path_config
    import torch

    path_config = get_path_config()
    assets_root = path_config.assets_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_holder = TTSModelHolder(assets_root, device)
    app = create_inference_app(model_holder)
    app.launch(inbrowser=True)
