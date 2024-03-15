# Paperspace gradient で学習する

詳しいコマンドの叩き方は[こちら](CLI.md)を参照してください。

## 事前準備
- Paperspace のアカウントを作成し必要なら課金する
- Projectを作る
- NotebookはStart from Scratchを選択して空いてるGPUマシンを選ぶ

## 使い方

以下では次のような方針でやっています。

- `/storage/`は永続ストレージなので、事前学習モデルとかを含めてリポジトリをクローンするとよい。
- `/notebooks/`はノートブックごとに変わるストレージなので（同一ノートブック違うランタイムだと共有されるらしい）、データセットやその結果を保存する。ただ容量が多い場合はあふれる可能性があるので`/tmp/`に保存するとよいかもしれない。
- hugging faceアカウントを作り、（プライベートな）リポジトリを作って、学習元データを置いたり、学習結果を随時アップロードする。

### 1. 環境を作る

以下はデフォルトの`Start from Scratch`で作成した環境の場合。[Dockerfile.train](../Dockerfile.train)を使ったカスタムイメージをするとPythonの環境構築の手間がちょっと省けるので、それを使いたい人は`Advanced Options / Container / Name`に[`litagin/mygradient:latest`](https://hub.docker.com/r/litagin/mygradient/tags)を指定すると使えます（pipの箇所が不要になる等）。

まずは永続ストレージにgit clone
```bash
mkdir -p /storage/sbv2
cd /storage/sbv2
git clone https://github.com/litagin02/Style-Bert-VITS2.git
```
環境構築（デフォルトはPyTorch 1.x系、Python 3.9の模様）
```bash
cd /storage/sbv2/Style-Bert-VITS2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install -r requirements.txt
```
事前学習済みモデル等のダウンロード、またパスを`/notebooks/`以下のものに設定
```bash
python initialize.py --skip_jvnv --dataset_root /notebooks/Data --assets_root /notebooks/model_assets
```

### 2. データセットの準備
以下では`username/voices`というデータセットリポジトリにある`Foo.zip`というデータセットを使うことを想定しています。
```bash
cd /notebooks
huggingface-cli login  # 事前にトークンが必要
huggingface-cli download username/voices Foo.zip --repo-type dataset --local-dir .
```

- zipファイル中身が既に`raw`と`esd.list`があるデータ（スライス・書き起こし済み）の場合
```bash
mkdir -p Data/Foo
unzip Foo.zip -d Data/Foo
rm Foo.zip
cd /storage/sbv2/Style-Bert-VITS2
```

- zipファイルが音声ファイルのみの場合
```bash
mkdir inputs
unzip Foo.zip -d inputs
cd /storage/sbv2/Style-Bert-VITS2
python slice.py --model_name Foo -i /notebooks/inputs
python transcribe.py --model_name Foo --use_hf_whisper
```

それが終わったら、以下のコマンドで一括前処理を行う（パラメータは各自お好み、バッチサイズ5か6でVRAM 16GBギリくらい）。
```bash
python preprocess_all.py --model_name Foo -b 5 -e 300 --use_jp_extra
```

### 3. 学習

Hugging faceの`username/sbv2-private`というモデルリポジトリに学習済みモデルをアップロードすることを想定しています。事前に`huggingface-cli login`でログインしておくこと。
```bash
python train_ms_jp_extra.py --repo_id username/sbv2-private
```
(JP-Extraでない場合は`train_ms.py`を使う)

### 4. 学習再開

Notebooksの時間制限が切れてから別Notebooksで同じモデルを学習を再開する場合（環境構築は必要）。
```bash
huggingface-cli login
cd /notebooks
huggingface-cli download username/sbv2-private --include "Data/Foo/*" --local-dir .
cd /storage/sbv2/Style-Bert-VITS2
python train_ms_jp_extra.py --repo_id username/sbv2-private --skip_default_style
```
前回の設定が残っているので特に前処理等は不要。