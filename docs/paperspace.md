# Paperspace gradient で学習する

詳しいコマンドの叩き方は[こちら](CLI.md)を参照してください。

## 事前準備
- Paperspace のアカウントを作成し必要なら課金する
- Projectを作る
- NotebookはStart from Scratchを選択して空いてるGPUマシンを選ぶ

## 使い方

以下では次のような方針でやっています。

- `/storage/`は永続ストレージなので、事前学習モデルとかを含めてリポジトリをクローンするとよい。
- `/notebooks/`は一時ストレージなので、データセットやその結果を保存する。
- hugging faceアカウントを作り、（プライベートな）リポジトリを作って、学習元データを置いたり、学習結果を随時アップロードする。

### 1. 環境を作る

まずは永続ストレージにgit clone
```bash
mkdir -p /storage/sbv2
cd /storage/sbv2
git clone https://github.com/litagin02/Style-Bert-VITS2.git
```
環境構築（デフォルトはPyTorch 1.x系、Python 3.9の模様）
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
事前学習済みモデル等のダウンロード、またパスを`/notebooks/`以下のものに設定
```bash
python initialize.py --skip_jvnv --dataset_root /notebooks/Data --assets_root /notebooks/model_assets
```

### 2. データセットの準備
以下では`username/voices`というデータセットリポジトリにある`Foo.zip`というデータセットを使うことを想定しています。
```bash
cd /nodtebooks
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
python transcribe.py --model_name Foo
```

それが終わったら、以下のコマンドで一括前処理を行う（パラメータは各自お好み、バッチサイズ6でVRAM 16GBギリくらい）。
```bash
python preprocess_all.py --model_name Foo -b 6 -e 300 --use_jp_extra
```

### 3. 学習

Hugging faceの`username/sbv2-private`というモデルリポジトリに学習済みモデルをアップロードすることを想定しています。事前に`huggingface-cli login`でログインしておくこと。
```bash
python train_ms_jp_extra.py --repo_id username/sbv2-private
```
(JP-Extraでない場合は`train_ms.py`を使う)
