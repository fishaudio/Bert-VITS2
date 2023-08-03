from flask import Flask, request, Response
from io import BytesIO
import ffmpeg
import base64
import os
import sys
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import time
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io import wavfile

# Get ffmpeg path
ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg")

# Flask Init
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# Text Preprocess
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# Load Generator
hps_mt = utils.get_hparams_from_file("/GPUFS/sysu_hpcedu_123/vits/configs/genshin_xm37.json")

net_g_mt = SynthesizerTrn(
    len(symbols),
    hps_mt.data.filter_length // 2 + 1,
    hps_mt.train.segment_size // hps_mt.data.hop_length,
    n_speakers=hps_mt.data.n_speakers,
    **hps_mt.model).cuda()
_ = net_g_mt.eval()

_ = utils.load_checkpoint("/GPUFS/sysu_hpcedu_123/vits/logs/xm37/G_xm37_361200.pth", net_g_mt, None)

npcList = ['空', '荧', '派蒙', '纳西妲', '阿贝多', '温迪', '枫原万叶', '钟离', '荒泷一斗', '八重神子', '艾尔海森', '提纳里', '迪希雅', '卡维', '宵宫', '莱依拉', '赛诺', '诺艾尔', '托马', '凝光', '莫娜', '北斗', '神里绫华', '雷电将军', '芭芭拉', '鹿野院平藏', '五郎', '迪奥娜', '凯亚', '安柏', '班尼特', '琴', '柯莱', '夜兰', '妮露', '辛焱', '珐露珊', '魈', '香菱', '达达利亚', '砂糖', '早柚', '云堇', '刻晴', '丽莎', '迪卢克', '烟绯', '重云', '珊瑚宫心海', '胡桃', '可莉', '流浪者', '久岐忍', '神里绫人', '甘雨', '戴因斯雷布', '优菈', '菲谢尔', '行秋', '白术', '九条裟罗', '雷泽', '申鹤', '迪娜泽黛', '凯瑟琳', '多莉', '坎蒂丝', '萍姥姥', '罗莎莉亚', '留云借风真君', '绮良良', '瑶瑶', '七七', '奥兹', '米卡', '夏洛蒂', '埃洛伊', '博士', '女士', '大慈树王', '三月七', '娜塔莎', '希露瓦', '虎克', '克拉拉', '丹恒', '希儿', '布洛妮娅', '瓦尔特', '杰帕德', '佩拉', '姬子', '艾丝妲', '白露', '星', '穹', '桑博', '伦纳德', '停云', '罗刹', '卡芙卡', '彦卿', '史瓦罗', '螺丝咕姆', '阿兰', '银狼', '素裳', '丹枢', '黑塔', '景元', '帕姆', '可可利亚', '半夏', '符玄', '公输师傅', '奥列格', '青雀', '大毫', '青镞', '费斯曼', '绿芙蓉', '镜流', '信使', '丽塔', '失落迷迭', '缭乱星棘', '伊甸', '伏特加女孩', '狂热蓝调', '莉莉娅', '萝莎莉娅', '八重樱', '八重霞', '卡莲', '第六夜想曲', '卡萝尔', '姬子', '极地战刃', '布洛妮娅', '次生银翼', '理之律者', '真理之律者', '迷城骇兔', '希儿', '魇夜星渊', '黑希儿', '帕朵菲莉丝', '天元骑英', '幽兰黛尔', '德丽莎', '月下初拥', '朔夜观星', '暮光骑士', '明日香', '李素裳', '格蕾修', '梅比乌斯', '渡鸦', '人之律者', '爱莉希雅', '爱衣', '天穹游侠', '琪亚娜', '空之律者', '终焉之律者', '薪炎之律者', '云墨丹心', '符华', '识之律者', '维尔薇', '始源之律者', '芽衣', '雷之律者', '苏莎娜', '阿波尼亚', '陆景和', '莫弈', '夏彦', '左然', '标贝']

@app.route("/",methods=['GET','POST'])
def main():
    if request.method == 'GET':
        try:
            speaker = request.args.get('speaker')
            text = request.args.get('text').replace("/n","")
            noise = float(request.args.get("noise", 0.5))
            noisew = float(request.args.get("noisew", 0.6))
            length = float(request.args.get("length", 1.3))
            if length >= 2:
                return "Too big length"
            if len(text) >=200:
                return "Too long text"
            fmt = request.args.get("format", "wav")
            if None in (speaker, text):
                return "Missing Parameter"
            if fmt not in ("mp3", "wav"):
                return "Invalid Format"
        except:
            return "Invalid Parameter"

        stn_tst_mt = get_text(text, hps_mt)

        with torch.no_grad():
            x_tst_mt = stn_tst_mt.cuda().unsqueeze(0)
            x_tst_mt_lengths = torch.LongTensor([stn_tst_mt.size(0)]).cuda()
            sid_mt = torch.LongTensor([npcList.index(speaker)]).cuda()
            audio_mt = net_g_mt.infer(x_tst_mt, x_tst_mt_lengths, sid=sid_mt, noise_scale=noise, noise_scale_w=noisew, length_scale=length)[0][0,0].data.cpu().float().numpy()

        wav = BytesIO()
        wavfile.write(wav, hps_mt.data.sampling_rate, audio_mt)
        torch.cuda.empty_cache()
        if fmt == "mp3":
            process = (
	        ffmpeg
            .input("pipe:", format='wav', channel_layout="mono")
            .output("pipe:", format='mp3', audio_bitrate="320k")
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, cmd=ffmpeg_path)
        )
            out, _ = process.communicate(input=wav.read())
            return Response(out, mimetype="audio/mpeg")
        return Response(wav.read(), mimetype="audio/wav")
    elif request.method == 'POST':
        receive = request.get_data(as_text=True)
        data = json.loads(receive)
        speaker = data["speaker"]
        text = data["text"].replace("/n","")
        stn_tst_mt = get_text(text, hps_mt)

        with torch.no_grad():
            x_tst_mt = stn_tst_mt.cuda().unsqueeze(0)
            x_tst_mt_lengths = torch.LongTensor([stn_tst_mt.size(0)]).cuda()
            sid_mt = torch.LongTensor([npcList.index(speaker)]).cuda()
            audio_mt = net_g_mt.infer(x_tst_mt, x_tst_mt_lengths, sid=sid_mt, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.15)[0][0,0].data.cpu().float().numpy()

        wav = BytesIO()
        wavfile.write(wav, hps_mt.data.sampling_rate, audio_mt)
        torch.cuda.empty_cache()
        return Response(base64.b64encode(wav.read()))
        #return Response(wav.read(), mimetype="audio/wav")
