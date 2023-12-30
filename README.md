# Style-Bert-VITS2

Bert-VITS2 with more controllable voice styles.

https://github.com/litagin02/Style-Bert-VITS2/assets/139731664/b907c1b8-43aa-46e6-b03f-f6362f5a5a1e

**注意**: 上記動画のライセンス表記は誤っていました、正しくはCC BY-SA 4.0で商用利用に制限はありません。近日訂正版動画に差し替えます。

Online demo: https://huggingface.co/spaces/litagin/Style-Bert-VITS2-JVNV

This repository is based on [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1, so many thanks to the original author!

- [更新履歴](docs/CHANGELOG.md)

**概要**

- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)のv2.1を元に、正確に感情や発話スタイルを強弱混みで指定して音声を生成することができるようにしたものです。
- GitやPythonがない人でも（WindowsユーザーでNVIDIAのグラボがあるなら）簡単にインストールでき、学習もできます (多くを[EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2/)からお借りしました)。

## 使い方

<!-- 詳しくは[こちら](docs/tutorial.md)を参照してください。 -->

### 動作環境

各UIとAPI Serverにおいて、Windows コマンドプロンプト・WSL2・Linux(Ubuntu Desktop)での動作を確認しています(WSLでのパス指定は相対パスなど工夫ください)。

### インストール

Windows環境で最近のNVIDIA製グラボがあることを前提にしています。

#### GitやPythonに馴染みが無い方

1. [このzipファイル](https://github.com/litagin02/Style-Bert-VITS2/releases/download/1.0/Style-Bert-VITS2.zip)をダウンロードして展開し、中にある`Install-Style-Bert-VITS2.bat`をダブルクリックします。
2. 待つと自動で必要な環境がインストールされます。
3. その後、自動的に音声合成するためのWebUIが起動したらインストール成功です。デフォルトのモデルがダウンロードされるているので、そのまま遊ぶことができます。

またアップデートをしたい場合は、`Update-Style-Bert-VITS2.bat`をダブルクリックしてください。

#### GitやPython使える人

Windowsの通常環境のPython 3.10で動作確認しています。

```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
python -m venv venv
venv\Scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python initialize.py  # 必要なモデルとデフォルトTTSモデルをダウンロード
```
最後を忘れずに。

### 音声合成

`App.bat`をダブルクリックか、`python app.py`するとWebUIが起動します。インストール時にデフォルトのモデルがダウンロードされているので、学習していなくてもそれを使うことができます。

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
このように、推論には`config.json`と`*.safetensors`と`style_vectors.npy`が必要です。学習の段階で前者の2つは自動で作成されますが、`style_vectors.npy`は自分で作成する必要があります: 下の「スタイルの生成」を参照してください。

### 学習

`Train.bat`をダブルクリックか`python webui_train.py`するとWebUIが起動します。

### スタイルの生成

- `Style.bat`をダブルクリックか`python webui_style_vectors.py`するとWebUIが起動します。
- この手順は、音声ファイルたちからスタイルを作るのに必要な手順です。
- 学習とは独立しているので、学習中でもできるし、学習が終わっても何度もやりなおせます。
- スタイルについての詳細は[clustering.ipynb](clustering.ipynb)を参照してください。

### データセット作り

- `Dataset.bat`をダブルクリックか`python webui_dataset.py`すると、音声ファイルからデータセットを作るためのWebUIが起動します。音声ファイルのみからでもこれを使って学習できます。

注意: データセットの手動修正やノイズ除去や、より高品質なデータセットを作りたい場合は、[Aivis](https://github.com/tsukumijima/Aivis)や、そのデータセット部分のWindows対応版 [Aivis Dataset](https://github.com/litagin02/Aivis-Dataset) を使うのをおすすめします。

### API Server

構築した環境下で`python server_fastapi.py`するとAPIサーバーが起動します。
API仕様は起動後に`/docs`にて確認ください。

デフォルトではCORS設定を全てのドメインで許可しています。
できる限り、`config.yml`の`server.origins`の値を変更し、信頼できるドメインに制限ください(キーを消せばCORS設定を無効にできます)。

## Bert-VITS2 v2.1との関係

基本的にはBert-VITS2 v2.1のモデル構造を少し改造しただけです。[事前学習モデル](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base)も、実質Bert-VITS2 v2.1と同じものを使用しています（不要な重みを削ってsafetensorsに変換したもの）。

具体的には以下の点が異なります。

- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)のように、PythonやGitを知らない人でも簡単に使える。
- 感情埋め込みのモデルを変更（1024次元の[wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)から256次元の[wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)へ、感情埋め込みというよりは話者識別のための埋め込み）
- 埋め込みもベクトル量子化を取り払い、単なる全結合層に。
- スタイルベクトルファイル`style_vectors.npy`を作ることで、そのスタイルを使って効果の強さも連続的に指定しつつ音声を生成することができる。
- 各種WebUIを作成
- bf16での学習のサポート
- safetensors形式のサポート、デフォルトでsafetensorsを使用するように
- その他軽微なbugfixやリファクタリング

## TODO
- [ ] LinuxやWSL等、Windowsの通常環境以外でのサポート？
- [ ] 複数話者学習での音声合成対応（学習は現在でも可能）
- [ ] 本家のver 2.1, 2.2, 2.3モデルの推論対応？（ver 2.1以外は明らかにめんどいのでたぶんやらない）
- [x] `server_fastapi.py`の対応、とくにAPIで使えるようになると嬉しい人が増えるのかもしれない
- [ ] モデルのマージで声音と感情表現を混ぜる機能の実装
- [ ] 英語等多言語対応？


## 実験したいこと
- [ ] 複数話者での学習の実験（原理的にはできるはず、スタイルがどう効くかが未知）
- [ ] むしろ複数話者で単一話者扱いで学習しても、スタイル埋め込みが話者埋め込みでそこに話者の情報が含まれているので、スタイルベクトルを作ることで複数話者の音声を生成できるのではないか？
- [ ] もしそうなら、大量の人数の音声を学習させれば、ある意味「話者空間から適当に選んだ話者（連続的に変えられる）の音声合成」ができるのでは？リファレンス音声も使えばゼロショットでの音声合成もできるのでは？


## References
In addition to the original reference (written below), I used the following repositories:
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)

[The pretrained model](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) is essentially taken from [the original base model of Bert-VITS2 v2.1](https://huggingface.co/Garydesu/bert-vits2_base_model-2.1), so all the credits go to the original author ([Fish Audio](https://github.com/fishaudio)):


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
