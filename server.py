from flask import Flask, request,Response
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pyffmpeg import FFmpeg
import time
from io import BytesIO
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io import wavfile

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

app = Flask(__name__)

hps_mt = utils.get_hparams_from_file("/GPUFS/sysu_hpcedu_123/vits/configs/genshin.json")

net_g_mt = SynthesizerTrn(
    len(symbols),
    hps_mt.data.filter_length // 2 + 1,
    hps_mt.train.segment_size // hps_mt.data.hop_length,
    n_speakers=hps_mt.data.n_speakers,
    **hps_mt.model).cuda()
_ = net_g_mt.eval()

_ = utils.load_checkpoint("/GPUFS/sysu_hpcedu_123/vits/logs/genshin/G_447000.pth", net_g_mt, None)

npc_list = ['ÅÉÃÉ', '¿­ÑÇ', '°²°Ø', 'ÀöÉ¯', 'ÇÙ', 'ÏãÁâ', '·ãÔ­ÍòÒ¶',
           'µÏÂ¬¿Ë', 'ÎÂµÏ', '¿ÉÀò', 'ÔçèÖ', 'ÍĞÂí', '°Å°ÅÀ­', 'ÓÅÇ‰',
           'ÔÆİÀ', 'ÖÓÀë', '÷Ì', 'Äı¹â', 'À×µç½«¾ü', '±±¶·',
           '¸ÊÓê', 'ÆßÆß', '¿ÌÇç', 'ÉñÀïç±»ª', '´÷ÒòË¹À×²¼', 'À×Ôó',
           'ÉñÀïç±ÈË', 'ÂŞÉ¯ÀòÑÇ', '°¢±´¶à', '°ËÖØÉñ×Ó', 'Ïü¹¬',
           '»ÄãñÒ»¶·', '¾ÅÌõôÄÂŞ', 'Ò¹À¼', 'Éºº÷¹¬ĞÄº£', 'ÎåÀÉ',
           'É¢±ø', 'Å®Ê¿', '´ï´ïÀûÑÇ', 'ÄªÄÈ', '°àÄáÌØ', 'Éêº×',
           'ĞĞÇï', 'ÑÌç³', '¾ÃáªÈÌ', 'ĞÁìÍ', 'É°ÌÇ', 'ºúÌÒ', 'ÖØÔÆ',
           '·ÆĞ»¶û', 'Åµ°¬¶û', 'µÏ°ÂÄÈ', 'Â¹Ò°ÔºÆ½²Ø']


@app.route("/")
def main():
    try:
        speaker = request.args.get('speaker')
        text = request.args.get('text')
        noise = float(request.args.get("noise", 0.667))
        noisew = float(request.args.get("noisew", 0.8))
        length = float(request.args.get("length", 1.2))
        if None in (speaker, text):
            return "NO"
    except:
        raise
        return "NO"

    stn_tst_mt = get_text(text, hps_mt)

    with torch.no_grad():
        x_tst_mt = stn_tst_mt.cuda().unsqueeze(0)
        x_tst_mt_lengths = torch.LongTensor([stn_tst_mt.size(0)]).cuda()
        sid_mt = torch.LongTensor([npc_list.index(speaker)]).cuda()
        audio_mt = net_g_mt.infer(x_tst_mt, x_tst_mt_lengths, sid=sid_mt, noise_scale=noise, noise_scale_w=noisew, length_scale=length)[0][0,0].data.cpu().float().numpy()
    wav_file_savepath = "/GPUFS/sysu_hpcedu_123/vits/voice/wav/" + str(time.time()) + ".wav"
    wavfile.write(wav_file_savepath, hps_mt.data.sampling_rate, audio_mt)
    res = streamwav(wav_file_savepath)
    return Response(res, mimetype="audio/mp3")
def wav_to_mp3(wav_file_savepath):
    mp3_save_path = "/GPUFS/sysu_hpcedu_123/vits/voice/mp3/" + str(time.time()) + ".mp3"
    output_file = os.system("/GPUFS/sysu_hpcedu_123/ffmpeg/ffmpeg -i " + wav_file_savepath + " " + mp3_save_path)
    return mp3_save_path
def streamwav(wav_file_savepath):
    def generate(mp3_save_path):
        with open(mp3_save_path, "rb") as fmp3:
            data = fmp3.read(1024)
            while data:
                yield data
                data = fmp3.read(1024)
    res = generate(wav_to_mp3(wav_file_savepath))
    return res