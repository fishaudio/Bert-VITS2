# Changelog

## v1.3

- `Dataset.bat`の音声スライスと書き起こしをよりカスタマイズできるように（秒数指定や書き起こしのWhisperモデル指定や言語指定等）
- `Style.bat`のスタイル作成で、新しい方法（DBSCAN）を追加（こちらのほうがスタイル数を指定できない代わりに特徴がよく出るかもしれません）。
- クラウド実行等の際にパスの指定をこちらでできるように、パスの設定を`configs/paths.yml`にまとめました（colabの[ノートブック](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)もそれに伴って変えたので新しいものを使ってください）。デフォルトは`dataset_root: Data`と`assets_root: model_assets`です。クラウド等でやる方はここを変更してください。
- どのステップ数の出力がよいかの「一つの」指標として [SpeechMOS](https://github.com/tarepan/SpeechMOS) を使うスクリプトを追加：
```bash
python speech_mos.py -m <model_name>
```
ステップごとの自然性評価が表示され、`mos_{model_name}.csv`と`mos_{model_name}.png`に結果が保存される。読み上げさせたい文章を変えたかったら中のファイルを弄って各自調整してください。またあくまで目安のひとつなので、実際に読み上げさせて選別するのが一番だと思います。

## v1.2 (2023-12-31)

- グラボがないユーザーでの音声合成をサポート、`Install-Style-Bert-VITS2-CPU.bat`でインストール。
- Google Colabでの学習をサポート、[ノートブック](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)を追加
- 音声合成のAPIサーバーを追加、`python server_fastapi.py`で起動します。API仕様は起動後に`/docs`にて確認ください。（ @darai0512 様によるPRです、ありがとうございます！）
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
