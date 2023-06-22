感谢国家超级计算广州中心提供的算力支持，感谢VITS模型作者Jaehyeon Kim, Jungil Kong, and Juhee Son，感谢ContentVEC作者 Kaizhi Qian.
本模型训练时使用的所有音频文件版权属于米哈游科技（上海）有限公司。

支持的说话者：
npcList = ['空', '荧', '派蒙', '纳西妲', '阿贝多', '温迪', '枫原万叶', '钟离', '荒泷一斗', '八重神子', '艾尔海森', '提纳里', '迪希雅', '卡维', '宵宫', '莱依拉', '赛诺', '诺艾尔', '托马', '凝光', '莫娜', '北斗', '神里绫华', '雷电将军', '芭芭拉', '鹿野院平藏', '五郎', '迪奥娜', '凯亚', '安柏', '班尼特', '琴', '柯莱', '夜兰', '妮露', '辛焱', '珐露珊', '魈', '香菱', '达达利亚', '砂糖', '早柚', '云堇', '刻晴', '丽莎', '迪卢克', '烟绯', '重云', '珊瑚宫心海', '胡桃', '可莉', '流浪者', '久岐忍', '神里绫人', '甘雨', '戴因斯雷布', '优菈', '菲谢尔', '行秋', '白术', '九条裟罗', '雷泽', '申鹤', '迪娜泽黛', '凯瑟琳', '多莉', '坎蒂丝', '萍姥姥', '罗莎莉亚', '留云借风真君', '绮良良', '瑶瑶', '七七', '奥兹', '米卡', '夏洛蒂', '埃洛伊', '博士', '女士', '大慈树王', '三月七', '娜塔莎', '希露瓦', '虎克', '克拉拉', '丹恒', '希儿', '布洛妮娅', '瓦尔特', '杰帕德', '佩拉', '姬子', '艾丝妲', '白露', '星', '穹', '桑博', '伦纳德', '停云', '罗刹', '卡芙卡', '彦卿', '史瓦罗', '螺丝咕姆', '阿兰', '银狼', '素裳', '丹枢', '黑塔', '景元', '帕姆', '可可利亚', '半夏', '符玄', '公输师傅', '奥列格', '青雀', '大毫', '青镞', '费斯曼', '绿芙蓉', '镜流', '信使', '丽塔', '失落迷迭', '缭乱星棘', '伊甸', '伏特加女孩', '狂热蓝调', '莉莉娅', '萝莎莉娅', '八重樱', '八重霞', '卡莲', '第六夜想曲', '卡萝尔', '姬子', '极地战刃', '布洛妮娅', '次生银翼', '理之律者', '真理之律者', '迷城骇兔', '希儿', '魇夜星渊', '黑希儿', '帕朵菲莉丝', '天元骑英', '幽兰黛尔', '德丽莎', '月下初拥', '朔夜观星', '暮光骑士', '明日香', '李素裳', '格蕾修', '梅比乌斯', '渡鸦', '人之律者', '爱莉希雅', '爱衣', '天穹游侠', '琪亚娜', '空之律者', '终焉之律者', '薪炎之律者', '云墨丹心', '符华', '识之律者', '维尔薇', '始源之律者', '芽衣', '雷之律者', '苏莎娜', '阿波尼亚', '陆景和', '莫弈', '夏彦', '左然', '标贝']


# VITS 原神语音合成


此外，也可以尝试使用公开的api：http://genshinvoice.top/api 来进行尝试，此API可用于二创等用途，但禁止用于任何商业用途。
请注意多次生成的效果不会一致，可以多次尝试来选择一次较好的效果。
同时支持可视化合成：http://genshinvoice.top
感谢星尘以及国家超级计算广州中心提供的算力支持，感谢VITS模型作者Jaehyeon Kim, Jungil Kong, and Juhee Son，本模型训练时使用的所有音频文件版权属于米哈游科技（上海）有限公司。

Query String 参数：

| 参数 | 类型 | 值 |
| ------------- | ------------- | ------------  |
| text | 字符串 |  生成的文本，支持常见标点符号。英文可能无法正常生成，数字请转换为对应的汉字再进行生成。 |
| speaker | 字符串 |  说话者名称。必须是上面的名称之一。 |
| noise | 浮点数 |  生成时使用的 noise_factor，可用于控制感情等变化程度。默认为0.667。 |
| noisew | 浮点数 |  生成时使用的 noise_factor_w，可用于控制音素发音长度变化程度。默认为0.8。 |
| length | 浮点数 |  生成时使用的 length_factor，可用于控制整体语速。默认为1.2。 |
| format | 字符串 |  生成语音的格式，必须为mp3或者wav。默认为mp3。 |

示例：http://genshinvoice.top/api?text=你好&speaker=派蒙




# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)
