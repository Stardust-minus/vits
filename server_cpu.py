from flask import Flask, request, Response
from io import BytesIO
import ffmpeg

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

# Text Preprocess
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# Load Generator
hps_mt = utils.get_hparams_from_file("/GPUFS/sysu_hpcedu_123/vits/configs/genshin_xm34.json")

net_g_mt = SynthesizerTrn(
    len(symbols),
    hps_mt.data.filter_length // 2 + 1,
    hps_mt.train.segment_size // hps_mt.data.hop_length,
    n_speakers=hps_mt.data.n_speakers,
    **hps_mt.model)
_ = net_g_mt.eval()

_ = utils.load_checkpoint("/GPUFS/sysu_hpcedu_123/vits/logs/xm34/G_xm34_472000.pth", net_g_mt, None)

npcList = ['黑希儿', '鹿野院平藏', '魈', '魇夜星渊', '香菱', '雷电将军', '雷泽', '雷之律者', '阿贝多', '阿波尼亚', '阿晃', '阿扎尔', '钟离', '重云', '迷城骇兔', '迪希雅', '迪娜泽黛', '迪奥娜', '迪卢克', '达达利亚', '辛焱', '赛诺', '诺艾尔', '识之律者', '行秋', '薪炎之律者', '萝莎莉娅', '萍姥姥', '菲谢尔', '菲尔戈黛特', '莱依拉', '莫弈', '莫娜', '莉莉娅', '荧', '荒泷一斗', '芽衣', '芭芭拉', '艾尔海森', '艾伯特', '胡桃', '羽生田千鹤', '罗莎莉亚', '缭乱星棘', '维尔薇', '纳西妲', '第六夜想曲', '符华', '空之律者', '空', '神里绫华', '神里绫人', '砂糖', '知易', '白术', '留云借风真君', '申鹤', '田铁嘴', '甘雨', '瑶瑶', '琴', '琪亚娜', '理之律者', '班尼特', '珐露珊', '珊瑚宫心海', '玛格丽特', '玛乔丽', '狂热蓝调', '爱衣', '爱莉希雅', '烟绯', '温迪', '渡鸦', '海芭夏', '流浪者', '派蒙', '次生银翼', '梅比乌斯', '格蕾修', '柯莱', '枫原万叶', '极地战刃', '杜拉夫', '李素裳', '朔夜观星', '月下初拥', '暮光骑士', '昆钧', '早柚', '提纳里', '掇星攫辰天君', '拉赫曼', '托马', '戴因斯雷布', '德丽莎', '式大将', '幽兰黛尔', '常九爷', '帕朵菲莉丝', '希儿', '布洛妮娅', '左然', '宵宫', '安柏', '姬子', '妮露', '女士', '奥拉夫', '奥兹', '失落迷迭', '天穹游侠', '天叔', '天元骑英', '大肉丸', '大慈树王', '夜兰', '多莉', '夏彦', '埃洛伊', '坎蒂丝', '哲平', '可莉', '卡萝尔', '卡莲', '博士', '北斗', '刻晴', '凯瑟琳', '凯亚', '凝光', '八重霞', '八重神子', '八重樱', '元素女孩-琪亚娜', '优菈', '伏特加女孩', '伊甸', '伊利亚斯', '人之律者', '五郎', '云墨丹心', '云堇', '九条镰治', '九条裟罗', '久岐忍', '丽莎', '丽塔', '七七', '一心传名刀', '一平']

@app.route("/")
def main():
    try:
        speaker = request.args.get('speaker')
        text = request.args.get('text')
        noise = float(request.args.get("noise", 0.667))
        noisew = float(request.args.get("noisew", 0.8))
        length = float(request.args.get("length", 1.2))
        fmt = request.args.get("format", "mp3")
        if None in (speaker, text):
            return "Missing Parameter"
        if fmt not in ("mp3", "wav"):
            return "Invalid Format"
    except:
        return "Invalid Parameter"

    stn_tst_mt = get_text(text, hps_mt)

    with torch.no_grad():
        x_tst_mt = stn_tst_mt.cpu().unsqueeze(0)
        x_tst_mt_lengths = torch.LongTensor([stn_tst_mt.size(0)]).cpu()
        sid_mt = torch.LongTensor([npcList.index(speaker)]).cpu()
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

