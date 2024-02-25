このフォルダに含まれるユーザー辞書関連のコードは、[VOICEVOX engine](https://github.com/VOICEVOX/voicevox_engine)プロジェクトのコードを改変したものを使用しています。VOICEVOXプロジェクトのチームに深く感謝し、その貢献を尊重します。

**元のコード**:

- [voicevox_engine/user_dict/](https://github.com/VOICEVOX/voicevox_engine/tree/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/user_dict)
- [voicevox_engine/model.py](https://github.com/VOICEVOX/voicevox_engine/blob/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/model.py#L207)

**改変の詳細**:

- ファイル名の書き換えおよびそれに伴うimport文の書き換え。
- VOICEVOX固有の部分をコメントアウト。
- mutexを使用している部分をコメントアウト。
- 参照しているpyopenjtalkの違いによるメソッド名の書き換え。
- UserDictWordのmora_countのデフォルト値をNoneに指定。
- Pydanticのモデルで必要な箇所のみを抽出。

**ライセンス**:

元のVOICEVOX engineのリポジトリのコードは、LGPL v3 と、ソースコードの公開が不要な別ライセンスのデュアルライセンスの下で使用されています。当プロジェクトにおけるこのモジュールもLGPLライセンスの下にあります。詳細については、プロジェクトのルートディレクトリにある[LGPL_LICENSE](/LGPL_LICENSE)ファイルをご参照ください。また、元のVOICEVOX engineプロジェクトのライセンスについては、[こちら](https://github.com/VOICEVOX/voicevox_engine/blob/master/LICENSE)をご覧ください。
