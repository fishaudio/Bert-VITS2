# flake8: noqa: E402

from faster_whisper import WhisperModel
import traceback
import time
import logging
import os
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

'''
Audio Annotation Tool
Automatically transcribe all audio files in the folder and generate a file named "genshin.list" suitable for training.
'''


def get_text_from_lab(filename):
    """Read text content from .lab file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip().replace('\n', '')


def generate_list_file(directory, output_file, language, speaker_name):
    """Generate the result file 'genshin.list"""
    audio_files = [f for f in os.listdir(directory_path) if os.path.splitext(f)[1] in supported_extensions]
    if language == 'zh':
        language = 'ZH'

    with open(output_file, 'w', encoding='utf-8') as out:
        for wav_file in audio_files:
            lab_file = os.path.splitext(wav_file)[0] + '.lab'
            lab_file_path = os.path.join(directory, lab_file)

            if os.path.exists(lab_file_path):
                text = get_text_from_lab(lab_file_path)
                if text == '' or text is None:
                    continue
                line = f"./dataset/{speaker_name}/{os.path.splitext(wav_file)[0]}.wav|{speaker_name}|{language}|{text}\n"

                out.write(line)


supported_extensions = ['.wav']
english_punctuation = set(string.punctuation)
chinese_punctuations = set("，。？！；：「」『』、〈〉《》")
all_punctuations = english_punctuation | chinese_punctuations


def join_list_to_string(lst, language='zh'):
    if not lst:
        return ""
    connector = "，" if language == 'zh' else ","
    end_char = "。" if language == 'zh' else "."

    segments = [lst[0]]
    for i in range(1, len(lst)):
        separator = connector if lst[i - 1][-1] not in all_punctuations else ""
        segments.append(separator + lst[i])

    result = ''.join(segments)

    if result[-1] not in all_punctuations:
        result += end_char

    return result


def filter_file_content(file_path, max_character=110):
    """
    Filter out illusions that appear with 'whisper'
    :param max_character:
    :param file_path:
    :return:
    """
    keywords_to_filter = ["普通话", "字幕"]
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # filter over 110 character
    filtered_lines = [
        line for line in lines
        if not any(keyword in line for keyword in keywords_to_filter)
           and len(line.strip()) <= max_character
    ]

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(filtered_lines)


class WhisperTool:
    '''
    device: cuda:0 / cuda:1
    model: base / small / medium / large / large-v2
    compute_type:  int8 float16 int16
    '''

    def __init__(self, device='cuda',
                 device_index=[1],
                 model_name="large-v2",
                 compute_type='float32'):
        self.device = device
        self.device_index = device_index
        self.model_name = model_name
        self.compute_type = compute_type
        self.model = WhisperModel(model_name,
                                  device=device,
                                  device_index=device_index,
                                  compute_type=compute_type)

    def process(self, directory_path,
                speaker_name='keqing',
                output_file='genshin.list',
                language='zh'):
        """
        Annotate all audio files in the folder and format them as {wav_path}|{speaker_name}|{language}|{text} and save to 'filelist'.
        :param directory_path:
        :param output_file:
        :param language:
        :param speaker_name:
        :return:
        """
        self.process_directory(directory_path, language)
        generate_list_file(directory_path, output_file, language, speaker_name)
        filter_file_content(output_file)

    def process_directory(self, directory_path, language='zh') -> None:
        """
        Transcribe audio files in the folder into .lab text files
        :param directory_path:
        :return:
        """
        try:
            logger.info(f"Transcribe ：{directory_path}")
            start = time.time()
            audio_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                           os.path.splitext(f)[1] in supported_extensions]
            for audio_file in audio_files:
                self.process_file(audio_file)
            costs = time.time() - start
            logger.info(f"complete transcribe ：costs: {costs}s {directory_path}")
        except Exception as e:
            logger.error(f"Whisper transcribe error occurred while processing whisper_save_txt: {str(e)}")
            logger.error(traceback.format_exc())

    def process_file(self, file_path, language='zh') -> None:
        """
        Transcribe audio files into .lab text files
        :param language:
        :param file_path:
        :return:
        """

        try:
            file_path = os.path.abspath(file_path)
            if os.path.isfile(file_path):
                output_directory = os.path.dirname(file_path)
                file_prefix = os.path.splitext(os.path.basename(file_path))[0]
            initial_prompt = None
            if language == 'zh':
                initial_prompt = '以下是普通话的句子'
            segments, info = self.model.transcribe(file_path,
                                                   beam_size=5,
                                                   word_timestamps=False,
                                                   vad_filter=False,
                                                   language=language, initial_prompt=initial_prompt)
            with open(os.path.join(output_directory, file_prefix + '.lab'), 'w') as file:
                v = []
                for segment in segments:
                    v.append(segment.text.strip())
                content = join_list_to_string(v, language)
                logger.info(content)
                file.write(content)
            logger.info(f"complete transcribe： {file_path}")
        except Exception as e:
            logger.error(f"Whisper: An error occurred while processing whisper_save_txt: {str(e)}")
            logger.error(traceback.format_exc())


if __name__ == "__main__":

    whisper_tool = WhisperTool()

    directory_path = './raw/keqing'
    speaker_name = 'keqing'
    language = 'zh'

    whisper_tool.process(directory_path, speaker_name=speaker_name, language=language)
