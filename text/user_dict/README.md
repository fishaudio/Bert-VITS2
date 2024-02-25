このフォルダのコードは、[voicevox_engine](https://github.com/VOICEVOX/voicevox_engine)のものを使わせていただいています。

引用元:

https://github.com/VOICEVOX/voicevox_engine/tree/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/user_dict

https://github.com/VOICEVOX/voicevox_engine/blob/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/model.py#L207

改変部分は以下のとおりです。
- ファイル名の書き換え、それに伴うimport文の書き換え
- VOICEVOX固有の部分のコメントアウト
- mutexを使用している部分のコメントアウト
- 参照しているpyopenjtalkの違いによるメソッド名書き換え
- UserDictWordのmora_countのデフォルト値をNoneに指定
- Pydanticのモデルは必要な箇所のみを抽出

ライセンス: LGPL

[LGPL_LICENSE](LGPL_LICENSE)を参照してください。
