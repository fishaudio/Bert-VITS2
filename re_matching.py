import re


def extract_language_and_text_updated(speaker, dialogue):
    # 使用正则表达式匹配<语言>标签和其后的文本
    pattern_language_text = r"<(\S+?)>([^<]+)"
    matches = re.findall(pattern_language_text, dialogue, re.DOTALL)
    speaker = speaker[1:-1]
    # 清理文本：去除两边的空白字符
    matches_cleaned = [(lang.upper(), text.strip()) for lang, text in matches]
    matches_cleaned.append(speaker)
    return matches_cleaned


def validate_text(input_text):
    # 验证说话人的正则表达式
    pattern_speaker = r"(\[\S+?\])((?:\s*<\S+?>[^<\[\]]+?)+)"

    # 使用re.DOTALL标志使.匹配包括换行符在内的所有字符
    matches = re.findall(pattern_speaker, input_text, re.DOTALL)

    # 对每个匹配到的说话人内容进行进一步验证
    for _, dialogue in matches:
        language_text_matches = extract_language_and_text_updated(_, dialogue)
        if not language_text_matches:
            return (
                False,
                "Error: Invalid format detected in dialogue content. Please check your input.",
            )

    # 如果输入的文本中没有找到任何匹配项
    if not matches:
        return (
            False,
            "Error: No valid speaker format detected. Please check your input.",
        )

    return True, "Input is valid."


def text_matching(text: str) -> list:
    speaker_pattern = r"(\[\S+?\])(.+?)(?=\[\S+?\]|$)"
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    result = []
    for speaker, dialogue in matches:
        result.append(extract_language_and_text_updated(speaker, dialogue))
    return result


def cut_para(text):
    splitted_para = re.split("[\n]", text)  # 按段分
    splitted_para = [
        sentence.strip() for sentence in splitted_para if sentence.strip()
    ]  # 删除空字符串
    return splitted_para


def cut_sent(para):
    para = re.sub("([。！;？\?])([^”’])", r"\1\n\2", para)  # 单字符断句符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


if __name__ == "__main__":
    text = """
    [说话人1]
    [说话人2]<zh>你好吗？<jp>元気ですか？<jp>こんにちは，世界。<zh>你好吗？
    [说话人3]<zh>谢谢。<jp>どういたしまして。
    """
    text_matching(text)
    # 测试函数
    test_text = """
    [说话人1]<zh>你好，こんにちは！<jp>こんにちは，世界。
    [说话人2]<zh>你好吗？
    """
    text_matching(test_text)
    res = validate_text(test_text)
    print(res)
