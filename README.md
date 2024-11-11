# Style-Bert-VITS2

**利用の際は必ず[お願いとデフォルトモデルの利用規約](/docs/TERMS_OF_USE.md)をお読みください。**

Bert-VITS2 with more controllable voice styles.

https://github.com/litagin02/Style-Bert-VITS2/assets/139731664/e853f9a2-db4a-4202-a1dd-56ded3c562a0

You can install via `pip install style-bert-vits2` (inference only), see [library.ipynb](/library.ipynb) for example usage.

- **解説チュートリアル動画** [YouTube](https://youtu.be/aTUSzgDl1iY)　[ニコニコ動画](https://www.nicovideo.jp/watch/sm43391524)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)
- [**よくある質問** (FAQ)](/docs/FAQ.md)
- [🤗 オンラインデモはこちらから](https://huggingface.co/spaces/litagin/Style-Bert-VITS2-Editor-Demo)
- [Zennの解説記事](https://zenn.dev/litagin/articles/034819a5256ff4)

- [**リリースページ**](https://github.com/litagin02/Style-Bert-VITS2/releases/)、[更新履歴](/docs/CHANGELOG.md)
  - 2024-09-09: Ver 2.6.1: Google colabでうまく学習できない等のバグ修正のみ
  - 2024-06-16: Ver 2.6.0 (モデルの差分マージ・加重マージ・ヌルモデルマージの追加、使い道については[この記事](https://zenn.dev/litagin/articles/1297b1dc7bdc79)参照)
  - 2024-06-14: Ver 2.5.1 (利用規約をお願いへ変更したのみ)
  - 2024-06-02: Ver 2.5.0 (**[利用規約](/docs/TERMS_OF_USE.md)の追加**、フォルダ分けからのスタイル生成、小春音アミ・あみたろモデルの追加、インストールの高速化等)
  - 2024-03-16: ver 2.4.1 (**batファイルによるインストール方法の変更**)
  - 2024-03-15: ver 2.4.0 (大規模リファクタリングや種々の改良、ライブラリ化)
  - 2024-02-26: ver 2.3 (辞書機能とエディター機能)
  - 2024-02-09: ver 2.2
  - 2024-02-07: ver 2.1
  - 2024-02-03: ver 2.0 (JP-Extra)
  - 2024-01-09: ver 1.3
  - 2023-12-31: ver 1.2
  - 2023-12-29: ver 1.1
  - 2023-12-27: ver 1.0

This repository is based on [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1 and Japanese-Extra, so many thanks to the original author!

**概要**

- 入力されたテキストの内容をもとに感情豊かな音声を生成する[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)のv2.1とJapanese-Extraを元に、感情や発話スタイルを強弱込みで自由に制御できるようにしたものです。
- GitやPythonがない人でも（Windowsユーザーなら）簡単にインストールでき、学習もできます (多くを[EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2/)からお借りしました)。またGoogle Colabでの学習もサポートしています: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)
- 音声合成のみに使う場合は、グラボがなくてもCPUで動作します。
- 音声合成のみに使う場合、Pythonライブラリとして`pip install style-bert-vits2`でインストールできます。例は[library.ipynb](/library.ipynb)を参照してください。
- 他との連携に使えるAPIサーバーも同梱しています ([@darai0512](https://github.com/darai0512) 様によるPRです、ありがとうございます)。
- 元々「楽しそうな文章は楽しそうに、悲しそうな文章は悲しそうに」読むのがBert-VITS2の強みですので、スタイル指定がデフォルトでも感情豊かな音声を生成することができます。


## 使い方

- CLIでの使い方は[こちら](/docs/CLI.md)を参照してください。
- [よくある質問](/docs/FAQ.md)も参照してください。

### 動作環境

各UIとAPI Serverにおいて、Windows コマンドプロンプト・WSL2・Linux(Ubuntu Desktop)での動作を確認しています(WSLでのパス指定は相対パスなど工夫ください)。NVidiaのGPUが無い場合は学習はできませんが音声合成とマージは可能です。

### インストール

Pythonライブラリとしてのpipでのインストールや使用例は[library.ipynb](/library.ipynb)を参照してください。

#### GitやPythonに馴染みが無い方

Windowsを前提としています。

1. [このzipファイル](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.6.0/sbv2.zip)を**パスに日本語や空白が含まれない場所に**ダウンロードして展開します。
  - グラボがある方は、`Install-Style-Bert-VITS2.bat`をダブルクリックします。
  - グラボがない方は、`Install-Style-Bert-VITS2-CPU.bat`をダブルクリックします。CPU版では学習はできませんが、音声合成とマージは可能です。
2. 待つと自動で必要な環境がインストールされます。
3. その後、自動的に音声合成するためのエディターが起動したらインストール成功です。デフォルトのモデルがダウンロードされるているので、そのまま遊ぶことができます。

またアップデートをしたい場合は、`Update-Style-Bert-VITS2.bat`をダブルクリックしてください。

ただし2024-03-16の**2.4.1**バージョン未満からのアップデートの場合は、全てを削除してから再びインストールする必要があります。申し訳ありません。移行方法は[CHANGELOG.md](/docs/CHANGELOG.md)を参照してください。

#### GitやPython使える人

Pythonの仮想環境・パッケージ管理ツールである[uv](https://github.com/astral-sh/uv)がpipより高速なので、それを使ってインストールすることをお勧めします。
（使いたくない場合は通常のpipでも大丈夫です。）

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
uv venv venv
venv\Scripts\activate
uv pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
python initialize.py  # 必要なモデルとデフォルトTTSモデルをダウンロード
```
最後を忘れずに。

### 音声合成

音声合成エディターは`Editor.bat`をダブルクリックか、`python server_editor.py --inbrowser`すると起動します（`--device cpu`でCPUモードで起動）。画面内で各セリフごとに設定を変えて原稿を作ったり、保存や読み込みや辞書の編集等ができます。
インストール時にデフォルトのモデルがダウンロードされているので、学習していなくてもそれを使うことができます。

エディター部分は[別リポジトリ](https://github.com/litagin02/Style-Bert-VITS2-Editor)に分かれています。

バージョン2.2以前での音声合成WebUIは、`App.bat`をダブルクリックか、`python app.py`するとWebUIが起動します。または`Inference.bat`でも音声合成単独タブが開きます。

音声合成に必要なモデルファイルたちの構造は以下の通りです（手動で配置する必要はありません）。
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
このように、推論には`config.json`と`*.safetensors`と`style_vectors.npy`が必要です。モデルを共有する場合は、この3つのファイルを共有してください。

このうち`style_vectors.npy`はスタイルを制御するために必要なファイルで、学習の時にデフォルトで平均スタイル「Neutral」が生成されます。
複数スタイルを使ってより詳しくスタイルを制御したい方は、下の「スタイルの生成」を参照してください（平均スタイルのみでも、学習データが感情豊かならば十分感情豊かな音声が生成されます）。

### 学習

- CLIでの学習の詳細は[こちら](docs/CLI.md)を参照してください。
- paperspace上での学習の詳細は[こちら](docs/paperspace.md)、colabでの学習は[こちら](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)を参照してください。

学習には2-14秒程度の音声ファイルが複数と、それらの書き起こしデータが必要です。

- 既存コーパスなどですでに分割された音声ファイルと書き起こしデータがある場合はそのまま（必要に応じて書き起こしファイルを修正して）使えます。下の「学習WebUI」を参照してください。
- そうでない場合、（長さは問わない）音声ファイルのみがあれば、そこから学習にすぐに使えるようにデータセットを作るためのツールを同梱しています。

#### データセット作り

- `App.bat`をダブルクリックか`python app.py`したところの「データセット作成」タブから、音声ファイルを適切な長さにスライスし、その後に文字の書き起こしを自動で行えます。または`Dataset.bat`をダブルクリックでもその単独タブが開きます。
- 指示に従った後、下の「学習」タブでそのまま学習を行うことができます。

#### 学習WebUI

- `App.bat`をダブルクリックか`python app.py`して開くWebUIの「学習」タブから指示に従ってください。または`Train.bat`をダブルクリックでもその単独タブが開きます。

### スタイルの生成

- デフォルトでは、デフォルトスタイル「Neutral」の他、学習フォルダのフォルダ分けに応じたスタイルが生成されます。
- それ以外の方法で手動でスタイルを作成したい人向けです。
- `App.bat`をダブルクリックか`python app.py`して開くWebUIの「スタイル作成」タブから、音声ファイルを使ってスタイルを生成できます。または`StyleVectors.bat`をダブルクリックでもその単独タブが開きます。
- 学習とは独立しているので、学習中でもできるし、学習が終わっても何度もやりなおせます（前処理は終わらせている必要があります）。

### API Server

構築した環境下で`python server_fastapi.py`するとAPIサーバーが起動します。
API仕様は起動後に`/docs`にて確認ください。

- 入力文字数はデフォルトで100文字が上限となっています。これは`config.yml`の`server.limit`で変更できます。
- デフォルトではCORS設定を全てのドメインで許可しています。できる限り、`config.yml`の`server.origins`の値を変更し、信頼できるドメインに制限ください(キーを消せばCORS設定を無効にできます)。

また音声合成エディターのAPIサーバーは`python server_editor.py`で起動します。があまりまだ整備をしていません。[エディターのリポジトリ](https://github.com/litagin02/Style-Bert-VITS2-Editor)から必要な最低限のAPIしか現在は実装していません。

音声合成エディターのウェブデプロイについては[このDockerfile](Dockerfile.deploy)を参考にしてください。

### マージ

2つのモデルを、「声質」「声の高さ」「感情表現」「テンポ」の4点で混ぜ合わせて、新しいモデルを作ったり、また「あるモデルに、別の2つのモデルの差分を足す」等の操作ができます。
`App.bat`をダブルクリックか`python app.py`して開くWebUIの「マージ」タブから、2つのモデルを選択してマージすることができます。または`Merge.bat`をダブルクリックでもその単独タブが開きます。

### 自然性評価

学習結果のうちどのステップ数がいいかの「一つの」指標として、[SpeechMOS](https://github.com/tarepan/SpeechMOS) を使うスクリプトを用意しています:
```bash
python speech_mos.py -m <model_name>
```
ステップごとの自然性評価が表示され、`mos_results`フォルダの`mos_{model_name}.csv`と`mos_{model_name}.png`に結果が保存される。読み上げさせたい文章を変えたかったら中のファイルを弄って各自調整してください。またあくまでアクセントや感情表現や抑揚を全く考えない基準での評価で、目安のひとつなので、実際に読み上げさせて選別するのが一番だと思います。

## Bert-VITS2との関係

基本的にはBert-VITS2のモデル構造を少し改造しただけです。[旧事前学習モデル](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base)も[JP-Extraの事前学習モデル](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra)も、実質Bert-VITS2 v2.1 or JP-Extraと同じものを使用しています（不要な重みを削ってsafetensorsに変換したもの）。

具体的には以下の点が異なります。

- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)のように、PythonやGitを知らない人でも簡単に使える。
- 感情埋め込みのモデルを変更（256次元の[wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)へ、感情埋め込みというよりは話者識別のための埋め込み）
- 感情埋め込みもベクトル量子化を取り払い、単なる全結合層に。
- スタイルベクトルファイル`style_vectors.npy`を作ることで、そのスタイルを使って効果の強さも連続的に指定しつつ音声を生成することができる。
- 各種WebUIを作成
- bf16での学習のサポート
- safetensors形式のサポート、デフォルトでsafetensorsを使用するように
- その他軽微なbugfixやリファクタリング


## References
In addition to the original reference (written below), I used the following repositories:
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)

[The pretrained model](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) and [JP-Extra version](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) is essentially taken from [the original base model of Bert-VITS2 v2.1](https://huggingface.co/Garydesu/bert-vits2_base_model-2.1) and [JP-Extra pretrained model of Bert-VITS2](https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra), so all the credits go to the original author ([Fish Audio](https://github.com/fishaudio)):


In addition, [text/user_dict/](text/user_dict) module is based on the following repositories:
- [voicevox_engine](https://github.com/VOICEVOX/voicevox_engine)
and the license of this module is LGPL v3.

## LICENSE

This repository is licensed under the GNU Affero General Public License v3.0, the same as the original Bert-VITS2 repository. For more details, see [LICENSE](LICENSE).

In addition, [text/user_dict/](text/user_dict) module is licensed under the GNU Lesser General Public License v3.0, inherited from the original VOICEVOX engine repository. For more details, see [LGPL_LICENSE](LGPL_LICENSE).



Below is the original README.md.
---

<div align="center">

<img alt="LOGO" src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" />

# Bert-VITS2

VITS2 Backbone with multilingual bert

For quick guide, please refer to `webui_preprocess.py`.

简易教程请参见 `webui_preprocess.py`。

## 请注意，本项目核心思路来源于[anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS) 一个非常好的tts项目
## MassTTS的演示demo为[ai版峰哥锐评峰哥本人,并找回了在金三角失落的腰子](https://www.bilibili.com/video/BV1w24y1c7z9)

[//]: # (## 本项目与[PlayVoice/vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41; 没有任何关系)

[//]: # ()
[//]: # (本仓库来源于之前朋友分享了ai峰哥的视频，本人被其中的效果惊艳，在自己尝试MassTTS以后发现fs在音质方面与vits有一定差距，并且training的pipeline比vits更复杂，因此按照其思路将bert)

## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应当参阅代码自己学习如何训练。

### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 严禁用于任何政治相关用途。
#### Video:https://www.bilibili.com/video/BV1hp4y1K78E
#### Demo:https://www.bilibili.com/video/BV1TF411k78w
#### QQ Group：815818430
## References
+ [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
+ [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
+ [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
+ [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
+ [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
+ [emotional-vits](https://github.com/innnky/emotional-vits)
+ [fish-speech](https://github.com/fishaudio/fish-speech)
+ [Bert-VITS2-UI](https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI)
## 感谢所有贡献者作出的努力
<a href="https://github.com/fishaudio/Bert-VITS2/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/Bert-VITS2"/>
</a>

[//]: # (# 本项目所有代码引用均已写明，bert部分代码思路来源于[AI峰哥]&#40;https://www.bilibili.com/video/BV1w24y1c7z9&#41;，与[vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41;无任何关系。欢迎各位查阅代码。同时，我们也对该开发者的[碰瓷，乃至开盒开发者的行为]&#40;https://www.bilibili.com/read/cv27101514/&#41;表示强烈谴责。)
