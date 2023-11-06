"""
翻译api
"""
from config import config

import random
import hashlib
import requests


def translate(Sentence: str, to_Language: str = "jp", from_Language: str = ""):
    """
    :param Sentence: 待翻译语句
    :param from_Language: 待翻译语句语言
    :param to_Language: 目标语言
    :return: 翻译后语句 出错时返回None

    常见语言代码：中文 zh 英语 en 日语 jp
    """
    appid = config.translate_config.app_key
    key = config.translate_config.secret_key
    if appid == "" or key == "":
        return "请开发者在config.yml中配置app_key与secret_key"
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    texts = Sentence.splitlines()
    outTexts = []
    for t in texts:
        if t != "":
            # 签名计算 参考文档 https://api.fanyi.baidu.com/product/113
            salt = str(random.randint(1, 100000))
            signString = appid + t + salt + key
            hs = hashlib.md5()
            hs.update(signString.encode("utf-8"))
            signString = hs.hexdigest()
            if from_Language == "":
                from_Language = "auto"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            payload = {
                "q": t,
                "from": from_Language,
                "to": to_Language,
                "appid": appid,
                "salt": salt,
                "sign": signString,
            }
            # 发送请求
            try:
                response = requests.post(
                    url=url, data=payload, headers=headers, timeout=3
                )
                response = response.json()
                if "trans_result" in response.keys():
                    result = response["trans_result"][0]
                    if "dst" in result.keys():
                        dst = result["dst"]
                        outTexts.append(dst)
            except Exception:
                return Sentence
        else:
            outTexts.append(t)
    return "\n".join(outTexts)
