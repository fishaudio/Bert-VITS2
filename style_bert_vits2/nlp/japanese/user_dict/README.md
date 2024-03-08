
## ユーザー辞書関連のコードについて

このフォルダに含まれるユーザー辞書関連のコードは、[VOICEVOX ENGINE](https://github.com/VOICEVOX/voicevox_engine) プロジェクトのコードを改変したものを使用しています。  
VOICEVOX プロジェクトのチームに深く感謝し、その貢献を尊重します。

### 元のコード

- [voicevox_engine/user_dict/](https://github.com/VOICEVOX/voicevox_engine/tree/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/user_dict)
- [voicevox_engine/model.py](https://github.com/VOICEVOX/voicevox_engine/blob/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/model.py#L207)

### 改変の詳細

- ファイル名の書き換えおよびそれに伴う import 文の書き換え。
- VOICEVOX 固有の部分をコメントアウト。
- mutex を使用している部分をコメントアウト。
- 参照している pyopenjtalk の違いによるメソッド名の書き換え。
- UserDictWord の mora_count のデフォルト値を None に指定。
- `model.py` のうち、必要な Pydantic モデルのみを抽出。

### ライセンス

元の VOICEVOX ENGINE のリポジトリのコードは、LGPL v3 と、ソースコードの公開が不要な別ライセンスのデュアルライセンスの下で使用されています。  
当プロジェクトにおけるこのモジュールも LGPL ライセンスの下にあります。

詳細については、プロジェクトのルートディレクトリにある [LGPL_LICENSE](/LGPL_LICENSE) ファイルをご参照ください。  
また、元の VOICEVOX ENGINE プロジェクトのライセンスについては、[こちら](https://github.com/VOICEVOX/voicevox_engine/blob/master/LICENSE) をご覧ください。
