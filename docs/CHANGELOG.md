# Changelog

## v2.6.1 (2024-09-09)

- Google colabで、torchのバージョン由来でエラーが発生する不具合の修正（たぶん）
- WebUIからのスタイル作成での、サブフォルダによるスタイル分けでエラーが発生していた点の修正

## v2.6.0 (2024-06-16)

### 新機能
モデルのマージ時に、今までの `new = (1 - weight) * A + weight * B` の他に、次を追加

- `new = A + weight * (B - C)`: 差分マージ
- `new = a * A + b * B + c * C`: 加重和マージ
- `new = A + weight * B`: ヌルモデルのマージ

差分マージは、例えばBを「Cと同じ話者だけど囁いているモデル」とすると、`B - C`が囁きベクトル的なものだと思えるので、それをAに足すことで、Aの話者が囁いているような音声を生成できるようになります。

また、加重和で`new = A - B`を作って、それをヌルモデルマージで別のモデルに足せば、実質差分マージを実現できます。また謎に`new = -A`や`new = 41 * A`等のモデルも作ることができます。

これらのマージの活用法については各自いろいろ考えて実験してみて、面白い使い方があればぜひ共有してください。

囁きについて実験的に作ったヌルモデルを[こちら](https://huggingface.co/litagin/sbv2_null_models)に置いています。これをヌルモデルマージで使うことで、任意のモデルを囁きモデルにある程度は変換できます。

### 改善

- スタイルベクトルのマージ部分のUIの改善
- WebUIの`App.bat`の起動が少し重いので、それぞれの機能を分割した`Dataset.bat`, `Inference.bat`, `Merge.bat`, `StyleVectors.bat`, `Train.bat`を追加 (今までの`App.bat`もこれまで通り使えます)

## v2.5.1 (2024-06-14)

ライセンスとのコンフリクトから、[利用規約](/docs/TERMS_OF_USE.md)を[開発陣からのお願いとデフォルトモデルの利用規約](/docs/TERMS_OF_USE.md)に変更しました。

## v2.5.0 (2024-06-02)

このバージョンから[利用規約](/docs/TERMS_OF_USE.md)が追加されました。ご利用の際は必ずお読みください。

### 新機能等

- デフォルトモデルに [あみたろの声素材工房](https://amitaro.net/) のあみたろ様が公開しているコーパスとライブ配信音声を利用して学習した[**小春音アミ**](https://huggingface.co/litagin/sbv2_koharune_ami)と[**あみたろ**](https://huggingface.co/litagin/sbv2_amitaro)モデルを追加（あみたろ様には事前に連絡して許諾を得ています）
    - アプデの場合は`Initialize.bat`をダブルクリックすればモデルをダウンロードできます（手動でダウンロードして`model_assets`フォルダに入れることも可能）
- 学習時に音声データをスタイルごとにフォルダ分けしておくことで、そのフォルダごとのスタイルを学習時に自動的に作成するように
    - `inputs`からスライスして使う場合は`inputs`直下に作りたいスタイルだけサブフォルダを作りそこに音声ファイルを配置
    - `Data/モデル名/raw`から使う場合も`raw`直下に同様に配置
    - サブフォルダの個数が0または1の場合は、今まで通りのNeutralスタイルのみが作成されます
- batファイルでのインストールの大幅な高速化（Pythonのライブラリインストールに[uv](https://github.com/astral-sh/uv)を使用）
- 学習時に「カスタムバッチサンプラーを無効化」オプションを追加。これにより、長い音声ファイルも学習に使われるようになりますが、使用VRAMがかなり増えたり学習が不安定になる可能性があります。
- [よくある質問](/docs/FAQ.md)を追加
- 英語の音声合成の速度向上（[gordon0414](https://github.com/gordon0414)さんによる[PR](https://github.com/litagin02/Style-Bert-VITS2/pull/124)です、ありがとうございます！）
- エディターの各種機能改善（多くが[kamexy](https://github.com/kamexy)様による[エディターリポジトリ](https://github.com/litagin02/Style-Bert-VITS2-Editor)へのプルリク群です、ありがとうございます！）
    - 選択した行の下に新規の行を作成できるように
    - Mac使用時に日本語変換のエンターで音声合成が走るバグの修正
    - ペースト時に改行を含まない場合は通常のペーストの振る舞いになるように修正


### その他の改善

- 上のスタイル自動作成機能を既存モデルでも使えるような機能追加。具体的には、スタイル作成タブにて、フォルダ分けされた音声ファイルのディレクトリを任意に指定し、そのフォルダ分けを使って既存のモデルのスタイルの作成が可能に
- 音声書き起こしに[kotoba-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.1)を追加
- 音声書き起こし時にHugging FaceのWhisperモデルを使う際に、書き起こしを順次保存するように改善
- 音声書き起こしのデフォルトをfaster-whiperからHugging FaceのWhisperモデルへ変更
- （**ライブラリとしてのみ**）依存関係の軽量化、音声合成時に読み上げテキストの読みを表す音素列を指定する機能を追加 + 様々な改善 ([tsukumijimaさん](https://github.com/tsukumijima)による[プルリク](https://github.com/litagin02/Style-Bert-VITS2/pull/118)です、ありがとうございます！)

### 内部変更

- これまでpath管理に`configs/paths.yml`を使っていたが、`configs/default_paths.yml`にリネームし、`configs/paths.yml`はgitの管理対象外に変更

### バグ修正

- Gradioのアップデートにより、モデル選択時やスタイルのDBSCAN作成時等に`TypeError: Type is not JSON serializable: WindowsPath`のようなエラーが出る問題を修正
- TensorboardをWebUIから立ち上げた際にエラーが出る問題の修正 ([#129](https://github.com/litagin02/Style-Bert-VITS2/issues/129))


## v2.4.1 (2024-03-16)

**batファイルでのインストール・アップデート方法の変更**（それ以外の変更はありません）

諸事情により、インストール・アップデートのbatファイルを変更しました（Gitが使えないのでバージョンアップ時のアップデートの対応が困難だったため、Gitがない環境の場合はPortableGitをダウンロードして使うように）。

伴って、これまでWindowsでbatファイルをダブルクリックしてインストールしていた方は**再インストールが必須**となります。大変申し訳ありません。

### インストール手順

（インストールの流れは変わりませんが、batファイルは変わっているので、新しいzipを必ずダウンロードしてください）

- [sbv2.zip](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.4.1/sbv2.zip)をダウンロードし、解凍してください。
- グラボがある方は、`Install-Style-Bert-VITS2.bat`をダブルクリックします。
- グラボがない方は、`Install-Style-Bert-VITS2-CPU.bat`をダブルクリックします。CPU版では学習はできませんが、音声合成とマージは可能です。

### アップデート手順

**以前のバージョンからのアップデート**

今までの環境を全て削除して新しくインストールする必要があります。
移行方法：
- 重要なデータが入っている可能性のある`Data`フォルダと`model_assets`フォルダをバックアップ
- 上のインストール手順から、新しい場所にStyle-Bert-VITS2をインストール
- インストールが終了したら、バックアップした`Data`フォルダと`model_assets`フォルダを新しい`Style-Bert-VITS2`フォルダにコピー
- これまでインストールされていたフォルダ（batファイルたち含む）は削除しても構いません

**今後のアップデート**

今後は、新しくインストールされた中の`Update-Style-Bert-VITS2.bat`をダブルクリックしてください。今までの`Update-Style-Bert-VITS2.bat`等のファイルは使えません。

## v2.4.0 (2024-03-15)

大規模リファクタリング・日本語処理のワーカー化と機能追加等。データセット作り・学習・音声合成・マージ・スタイルWebUIは全て`app.py` (`App.bat`) へ統一されましたのでご注意ください。

### アップデート手順
- 2.3未満（辞書・エディター追加前）からのアップデートの場合は、[Update-to-Dict-Editor.bat](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.4.0/Update-to-Dict-Editor.bat)をダウンロードし、`Style-Bert-VITS2`フォルダがある場所（インストールbatファイルとかがあったところ）においてダブルクリックしてください。
- それ以外の場合は、単純に今までの`Update-Style-Bert-VITS2.bat`でアップデートできます。
- ただしアップデートにより多くのファイルが移動したり不要になったりしたので、それらを削除したい場合は[Clean.bat](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.4.0/Clean.bat)を`Update-Style-Bert-VITS2.bat`と同じ場所に保存して実行してください。

### 内部改善

- [tsukumijimaさんによる大規模リファクタリングのプルリク](https://github.com/litagin02/Style-Bert-VITS2/pull/92) によって、内部コードが非常に整理され可読性が高まりライブラリ化もされた。[tsukumijimaさん](https://github.com/tsukumijima) 大変な作業を本当にありがとうございます！
- ライブラリとして`pip install style-bert-vits2`によりすぐにインストールでき、音声合成部分の機能が使えます（使用例は[/library.ipynb](/library.ipynb)を参照してください）
- その他このプルリクに動機づけられ、多くのコードのリファクタリング・型アノテーションの追加等を行った
- 日本語処理のpyopenjtalkをソケット通信を用いて別プロセス化し、複数同時に学習や音声合成を立ち上げても辞書の競合エラーが起きないように。[kale4eat](https://github.com/kale4eat) さんによる[PR](https://github.com/litagin02/Style-Bert-VITS2/pull/89) です、ありがとうございます！

### バグ修正

- 上記にもある通り、音声合成と学習前処理など、日本語処理を扱うものを2つ以上起動しようとするとエラーが発生する仕様の解決。ユーザー辞書は追加すれば常にどこからでも適応されます。
- `raw`フォルダの直下でなくサブフォルダ内に音声ファイルがある場合に、`wavs`フォルダでもその構造が保たれてしまい、書き起こしファイルとの整合性が取れなくなる挙動を修正し、常に`wav`フォルダ直下へ`wav`ファイルを保存するように変更
- スライス時に元ファイル名にピリオド `.` が含まれると、スライス後のファイル名がおかしくなるバグの修正

### 機能改善・追加

- 各種WebUIを一つ`app.py` `App.bat` に統一
- その他以下の変更や、軽微なUI・説明文の改善等

**データセット作成**

- スライス処理の高速化（マルチスレッドにした、大量にスライス元ファイルファイルがある場合に高速になります）、またスライス元のファイルを`wav`以外の`mp3`や`ogg`などの形式にも対応
- スライス処理時に、ファイル名にスライスされた開始終了区間を含めるオプションを追加（[aka7774](https://github.com/aka7774) さんによるPRです、ありがとうございます！）
- 書き起こしの高速化、またHugging FaceのWhisperモデルを使うオプションを追加。バッチサイズを上げることでVRAMを食う代わりに速度が大幅に向上します。

**学習**

- 学習元の音声ファイル（`Data/モデル名/raw`にいれるやつ）を、`wav`以外の`mp3`や`ogg`などの形式にも対応（前処理段階で自動的に`wav`ファイルに変換されます）（ただし変わらず1ファイル2-12秒程度の範囲の長さが望ましい）

**音声合成**

- 音声合成時に、生成音声の音の高さ（音高）と抑揚の幅を調整できるように（ただし音質が少し劣化する）。`App.bat`や`Editor.bat`のどちらからでも使えます。
- `Editor.bat`の複数話者モデルでの話者指定を可能に
- `Editor.bat`で、改行を含む文字列をペーストすると自動的に欄が増えるように。また「↑↓」キーで欄を追加・行き来できるように（エディター側で以前に既にアプデしていました）
- `Editor.bat`でモデル一覧のリロードをメニューに追加

**API**

- `server_fastapi.py`の実行時に全てのモデルファイルを読み込もうとする挙動を修正。音声合成がリクエストされて初めてそのモデルを読み込むように変更（APIを使わない音声合成のときと同じ挙動）
- `server_fastapi.py`の音声合成エンドポイント`/voice`について、GETメソッドに加えてPOSTメソッドを追加。GETメソッドでは多くの制約があるようなのでPOSTを使うことが推奨されます。

**CLI**

- `preprocess_text.py`で、書き起こしファイルでの音声ファイル名を自動的に正しい`Data/モデル名/wavs/`へ書き換える`--correct_path`オプションの追加（WebUIでは今までもこの挙動でした）
- その他上述のデータセット作成の機能追加に伴うCLIのオプションの追加（詳しくは[CLI.md](/docs/CLI.md)を参照）

## v2.3.1 (2024-02-27)

### バグ修正
- colabの学習用ノートブックが動かなかったのを修正
- `App.bat`や`server_fastapi.py`では読めない文字でまだエラーが発生するようになっていたので、推論時は必ず読めない文字を無視して強引に読むように挙動を変更

### 改善
- 読みが取得できない場合に、テキスト前処理完了時にエラーで中断する今までの挙動に加えて、「読み取得失敗ファイルを学習に使わずに進める」もしくは「読めない文字を無視して読んでファイルを学習に使い進める」というオプションを追加。
- マージ方法に線形補間の他に球面線形補完を追加 （[@frodo821](https://github.com/frodo821) さんによるPRです、ありがとうございます！）
- デプロイ用`.dockerignore`を更新

### アップデート手順
- 2.3未満からのアップデートの場合は、[Update-to-Dict-Editor.bat](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.3/Update-to-Dict-Editor.bat)をダウンロードし、`Style-Bert-VITS2`フォルダがある場所（インストールbatファイルとかがあったところ）においてダブルクリックしてください。
- 2.3からのアップデートの場合は、単純に今までの`Update-Style-Bert-VITS2.bat`でアップデートできます。

## v2.3 (2024-02-26)

### 大きな変更

大きい変更をいくつかしたため、**アップデートはまた専用の手順**が必要です。下記の指示にしたがってください。

#### ユーザー辞書機能
あらかじめ辞書に固有名詞を追加することができ、それが**学習時**・**音声合成時**の読み取得部分に適応されます。辞書の追加・編集は次のエディタ経由で行ってください。または、手持ちのOpenJTalkのcsv形式の辞書がある場合は、`dict_data/default.csv`ファイルを直接上書きや追加しても可能です。

使えそうな辞書（ライセンス等は各自ご確認ください）（他に良いのがあったら教えて下さい）：

- [WariHima/Kanayomi-dict](https://github.com/WariHima/KanaYomi-dict)
- [takana-v/tsumu_dic](https://github.com/takana-v/tsumu_dic)


辞書機能部分の[実装](/text/user_dict/) は、中のREADMEにある通り、[VOICEVOX Editor](https://github.com/VOICEVOX/voicevox) のものを使っており、この部分のコードライセンスはLGPL-3.0です。

#### 音声合成専用エディタ

[🤗 オンラインデモはこちらから](https://huggingface.co/spaces/litagin/Style-Bert-VITS2-Editor-Demo)

音声合成専用エディタを追加。今までのWebUIでできた機能のほか、次のような機能が使えます（つまり既存の日本語音声合成ソフトウェアのエディタを真似ました）：
- セリフ単位でキャラや設定を変更しながら原稿を作り、それを一括で生成したり、原稿を保存等したり読み込んだり
- GUIよる分かりやすいアクセント調整
- ユーザー辞書への単語追加や編集

`Editor.bat`をダブルクリックか`python server_editor.py --inbrowser`で起動します。エディター部分は[こちらの別リポジトリ](https://github.com/litagin02/Style-Bert-VITS2-Editor)になります。フロントエンド初心者なのでプルリクや改善案等をお待ちしています。

### バグ修正

- 特定の状況で読みが正しく取得できず `list index out of range` となるバグの修正
- 前処理時に、書き起こしファイルのある行の形式が不正だと、書き起こしファイルのそれ以降の内容が消えてしまうバグの修正
- faster-whisperが1.0.0にメジャーバージョンアップされ（今のところ）大幅に劣化したので、バージョンを0.10.1へ固定

### 改善

- テキスト前処理時に、読みの取得の失敗等があった場合に、処理を中断せず、エラーがおきた箇所を`text_error.log`ファイルへ保存するように変更。
- 音声合成時に、読めない文字があったときはエラーを起こさず、その部分を無視して読み上げるように変更（学習段階ではエラーを出します）
- コマンドラインで前処理や学習が簡単にできるよう、前処理を行う`preprocess_all.py`を追加（詳しくは[CLI.md](/docs/CLI.md)を参照）
- 学習の際に、自動的に自分のhugging faceリポジトリへ結果をアップロードするオプションを追加。コマンドライン引数で`--repo_id username/my_model`のように指定してください（詳しくは[CLI.md](/docs/CLI.md)を参照）。🤗の無制限ストレージが使えるのでクラウドでの学習に便利です。
- 学習時にデコーダー部分を凍結するオプションの追加。品質がもしかしたら上がるかもしれません。
- `initialize.py`に引数`--dataset_root`と`--assets_root`を追加し、`configs/paths.yml`をその時点で変更できるようにした

### その他

- [paperspaceでの学習の手引きを追加](/docs/paperspace.md)、paperspaceでのimageに使える[Dockerfile](/Dockerfile.train)を追加
- [CLIでの各種処理の実行の仕方を追加](/docs/CLI.md)
- [Hugging Face spacesで遊べる音声合成エディタ](https://huggingface.co/spaces/litagin/Style-Bert-VITS2-Editor-Demo)をデプロイするための[Dockerfile](Dockerfile.deploy)を追加

### アップデート手順

- [Update-to-Dict-Editor.bat](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.3/Update-to-Dict-Editor.bat)をダウンロードし、`Style-Bert-VITS2`フォルダがある場所（インストールbatファイルとかがあったところ）においてダブルクリックしてください。

- 手動での場合は、以下の手順で実行してください：
```bash
git pull
venv\Scripts\activate
pip uninstall pyopenjtalk-prebuilt
pip install -U -r requirements.txt
# python initialize.py  # これを1.x系からのアップデートの場合は実行してください
python server_editor.py --inbrowser
```

### 新規インストール手順
[このzip](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.3/Style-Bert-VITS2.zip)をダウンロードし、解凍してください。
を展開し、`Install-Style-Bert-VITS2.bat`をダブルクリックしてください。


## v2.2 (2024-02-09)

### 変更・機能追加
- bfloat16オプションはデメリットしか無さそうなので、常にオフで学習するよう変更
- バッチサイズのデフォルトを4から2に変更。学習が遅い場合はバッチサイズを下げて試してみて、VRAMに余裕があれば上げてください。JP-Extra使用時でのバッチサイズごとのVRAM使用量目安は、1: 6GB, 2: 8GB, 3: 10GB, 4: 12GB くらいのようです。
- 学習の際の検証データ数をデフォルトで0に変更し、また検証データ数を学習用WebUIで指定できるようにした
- Tensorboardのログ間隔を学習用WebUIで指定できるようにした
- UIのテーマを`common/constants.py`の`GRADIO_THEME`で指定できるようにした

### バグ修正
- JP-Extra使用時にバッチサイズが1だと学習中にエラーが発生するバグを修正
- 「こんにちは!?!?!?!?」等、感嘆符等の記号が連続すると学習・音声合成でエラーになるバグを修正
- `—` (em dash, U+2014) や `―` (quotation dash, U+2015) 等のダッシュやハイフンの各種変種が、種類によって`-`（通常の半角ハイフン）に正規化されたりされていなかったりする処理を、全て正規化するように修正

## v2.1 (2024-02-07)

### 変更
- 学習の際、デフォルトではbfloat16オプションを使わないよう変更（学習が発散したり質が下がることがある模様）
- 学習の際のメモリ使用量を削減しようと頑張った

### バグ修正や改善
- 学習WebUIからTensorboardのログを見れるように
- 音声合成（やそのAPI）において、同時に別の話者が選択され音声合成がリクエストされた場合に発生するエラーを修正
- モデルマージ時に、そのレシピを`recipe.json`ファイルへ保存するように変更
- 「改行で分けて生成」がより感情が乗る旨の明記等、軽微な説明文の改善
- 「`ーーそれは面白い`」や「`なるほど。ーーーそういうことか。`」等、長音記号の前が母音でない場合、長音記号`ー`でなくダッシュ`―`の勘違いだと思われるので、ダッシュ記号として処理するように変更

## v2.0.1 (2024-02-05)

軽微なバグ修正や改善
- スタイルベクトルに`NaN`が含まれていた場合（主に音声ファイルが極端に短い場合に発生）、それを学習リストから除外するように修正
- colabにマージの追加
- 学習時のプログレスバーの表示がおかしかったのを修正
- デフォルトのjvnvモデルをJP-Extra版にアップデート。新しいモデルを使いたい方は手動で[こちら](https://huggingface.co/litagin/style_bert_vits2_jvnv/tree/main)からダウンロードするか、`python initialize.py`をするか、[このbatファイル](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.0.1/Update-to-JP-Extra.bat)を`Style-Bert-VITS2`フォルダがある場所（インストールbatファイルとかがあったところ）においてダブルクリックしてください。

## v2.0 (2024-02-03)

### 大きい変更
モデル構造に [Bert-VITS2の日本語特化モデル JP-Extra](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta) を取り込んだものを使えるように変更、[事前学習モデル](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra)も[Bert-VITS2 JP-Extra](https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra)のものを改造してStyle-Bert-VITS2で使えるようにしました (モデル構造を見直して日本語での学習をしていただいた [@Stardust-minus](https://github.com/Stardust-minus) 様に感謝します)
- これにより、日本語の発音やアクセントや抑揚や自然性が向上する傾向があります
- スタイルベクトルを使ったスタイルの操作は変わらず使えます
- ただしJP-Extraでは英語と中国語の音声合成は（現状は）できません
- 旧モデルも引き続き使うことができ、また旧モデルで学習することもできます
- デフォルトのJVNVモデルは現在は旧verのままです

### 改善
- `Merge.bat`で、声音マージを、より細かく「声質」と「声の高さ」の点でマージできるように。

### バグ修正
- PyTorchのバージョンに由来するバグを修正（torchのバージョンを2.1.2に固定）
- `―`（ダッシュ、長音記号ではない）が2連続すると学習・音声合成でエラーになるバグを修正
- 「三円」等「ん＋母音」のアクセントの仮名表記が「サネン」等になり、また偶にエラーが発生する問題を修正（「ん」の音素表記を内部的には「N」で統一）

## v1.3 (2024-01-09)

### 大きい変更
- 元々のBert-VITS2に存在した、日本語の発音・アクセント処理部分のバグを修正・リファクタリング
    - `車両`が`シャリヨオ`、`思う`が`オモオ`、`見つける`が`ミッケル`等に発音・学習されており、その単語以降のアクセント情報が全て死んでいた
    - `私はそれを見る`のアクセントが`ワ➚タシ➘ワ　ソ➚レ➘オ　ミ➘ル`だったのを`ワ➚タシワ　ソ➚レオ　ミ➘ル`に修正
    - 学習・音声合成で無視されていたアルファベット・ギリシャ文字を無視しないように変更（基本はアルファベット読みだけど簡単な単語は読めるらしい、学習の際は念のためカタカナ等にしたほうがよいです）
    - 修正の影響で、前処理時に（今まで無視されていた）読めない漢字等で引っかかるようになりました。その場合は書き起こしを確認して修正するようにしてください。
- アクセントを調整して音声合成できるように（完全に制御できるわけではないが改善される場合がある）。

これまでのモデルもこれまで通り使え、アクセントや発音等が改善される可能性があります。新しいバージョンで学習し直すとより良くなる可能性もあります。が劇的に良くなるかは分かりません。

### 改善
- `Dataset.bat`の音声スライスと書き起こしをよりカスタマイズできるように（スライスの秒数設定や書き起こしのWhisperモデル指定や言語指定等）
- `Style.bat`のスタイル分けで、スタイルごとのサンプル音声を指定した数だけ複数再生できるように。また新しい次元削減方法（UMAP）と新しいスタイル分けの方法（DBSCAN）を追加（UMAPのほうがよくスタイルが分かれるかもしれません）
- `App.bat`での音声合成時に複数話者モデルの場合に話者を指定できるように
- colabの[ノートブック](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)で、音声ファイルのみからデータセットを作成するオプション部分を追加
- クラウド実行等の際にパスの指定をこちらでできるように、パスの設定を`configs/paths.yml`にまとめた（colabの[ノートブック](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)もそれに伴って更新）。デフォルトは`dataset_root: Data`と`assets_root: model_assets`なので、クラウド等でやる方はここを変更してください。
- どのステップ数の出力がよいかの「一つの」指標として [SpeechMOS](https://github.com/tarepan/SpeechMOS) を使うスクリプトを追加：
```bash
python speech_mos.py -m <model_name>
```
ステップごとの自然性評価が表示され、`mos_results`フォルダの`mos_{model_name}.csv`と`mos_{model_name}.png`に結果が保存される。読み上げさせたい文章を変えたかったら中のファイルを弄って各自調整してください。あくまでアクセントや感情表現や抑揚を全く考えない基準での評価で、目安のひとつなので、実際に読み上げさせて選別するのが一番だと思います。
- 学習時のウォームアップオプションを機能するように（ [@kale4eat](https://github.com/kale4eat) 様によるPRです、ありがとうございます！）。前処理時に生成される`config.json`の`train`の`warmup_epochs`を変更することで、ウォームアップのエポック数を変更できます。デフォルトは`0`で今までと同じ学習率の挙動です。

### その他
- `Dataset.bat`の音声スライスでノーマライズ機能を削除（学習前処理で行えるため）
- `Train.bat`の音量ノーマライズと無音切り詰めをデフォルトでオフに変更
- 学習時の進捗を全体エポック数で表示し、学習全体の進捗を見やすいように( [@RedRayz](https://github.com/RedRayz) 様によるPRです、ありがとうございます！)
- その他バグ修正等（ [@tinjyuu](https://github.com/@tinjyuu) 様、 [@darai0512](https://github.com/darai0512) 様ありがとうございます！）
- `config.json`にスタイル埋め込み部分を学習しない`freeze_style`オプションを追加（デフォルトは`false`）

### TIPS
- 日本語学習の場合、`config.json`の`freeze_bert`と`freeze_en_bert`を`true`にしておくと、英語と中国語の発話能力が学習の過程で落ちないかもしれませんが、あまり比較していなので分かりません。

## v1.2 (2023-12-31)

- グラボがないユーザーでの音声合成をサポート、`Install-Style-Bert-VITS2-CPU.bat`でインストール。
- Google Colabでの学習をサポート、[ノートブック](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)を追加
- 音声合成のAPIサーバーを追加、`python server_fastapi.py`で起動します。API仕様は起動後に`/docs`にて確認ください。（ [@darai0512](https://github.com/darai0512) 様によるPRです、ありがとうございます！）
- 学習時に自動的にデフォルトスタイル Neutral を生成するように。特にスタイル指定が必要のない方は、学習したらそのまま音声合成を試せます。これまで通りスタイルを自分で作ることもできます。
- マージ機能の新規追加: `Merge.bat`, `webui_merge.py`
- 前処理のリサンプリング時に音声ファイルの開始・終了部分の無音を削除するオプションを追加（デフォルトでオン）
- `スタイルテキスト (style text)`がスタイル指定と紛らわしかったので、`アシストテキスト (assist text)`に変更
- その他コードのリファクタリング

## v1.1 (2023-12-29)
- TrainとDatasetのWebUIの改良・調整（一括事前処理ボタン等）
- 前処理のリサンプリング時に音量を正規化するオプションを追加（デフォルトでオン）

## v1.0 (2023-12-27)
- 初版
