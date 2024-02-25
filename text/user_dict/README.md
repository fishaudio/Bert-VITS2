このモジュールは、[voicevox_engine](https://github.com/VOICEVOX/voicevox_engine)を使わせていただいています。

引用元:

https://github.com/VOICEVOX/voicevox_engine/tree/709527be089c0410c08e989df95a4a1d78439423/voicevox_engine/user_dict

改変部分は以下のとおりです。
- ファイル名の書き換え、それに伴うimport文の書き換え
- VOICEVOX固有の他のモジュールへの依存のコメントアウト
- mutexを使用している部分のコメントアウト
- 参照しているpyopenjtalkの違いによるメソッド名書き換え
- UserDictWordのmora_countのデフォルト値をNoneに指定

ライセンス: LGPL

[LGPL_LICENSE](LGPL_LICENSE)を参照してください。
