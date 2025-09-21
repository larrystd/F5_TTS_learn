import sys
import gc
import os
import re
import shutil
import time
import math
import torch
import torchaudio
import jieba
from pypinyin import lazy_pinyin, Style
import numpy as np
from vocos import Vocos
import onnxruntime
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

from dotenv import load_dotenv
load_dotenv()


F5_project_path      = os.getenv("F5_project_path")                  # The F5-TTS Github project download path.  URL: https://github.com/SWivid/F5-TTS
F5_safetensors_path  = os.getenv("F5_safetensors_path")              # The F5-TTS model download path.           URL: https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_Base
vocos_model_path     = os.getenv("vocos_model_path")                 # The Vocos model download path.            URL: https://huggingface.co/charactr/vocos-mel-24khz/tree/main
onnx_model_A         = os.getenv("onnx_model_A")                     # The exported onnx model path.
onnx_model_B         = os.getenv("onnx_model_B")                     # The exported onnx model path.
onnx_model_C         = os.getenv("onnx_model_C")                     # The exported onnx model path.
python_package_path  = os.getenv("python_package_path")              # The Python package path.
modified_path        = os.getenv("modified_path")


reference_audio      = os.getenv("reference_audio")
ref_text             = "对，这就是我，万人敬仰的太乙真人。"
generated_audio      = os.getenv("generated_audio")
gen_text             = "对，这就是我，万人敬仰的大可奇奇。"

with open(os.getenv("vocab_path"), "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)


F5_project_path += "/src"


if F5_project_path not in sys.path:
    sys.path.append(F5_project_path)


# Replace the original source code.
# Note! please re-install the vocos after the export process.
# Note! please re-download the F5 project after the export process.
shutil.copyfile(modified_path + '/vocos/heads.py', python_package_path + '/vocos/heads.py')
shutil.copyfile(modified_path + '/vocos/models.py', python_package_path + '/vocos/models.py')
shutil.copyfile(modified_path + '/vocos/modules.py', python_package_path + '/vocos/modules.py')
shutil.copyfile(modified_path + '/vocos/pretrained.py', python_package_path + '/vocos/pretrained.py')
shutil.copyfile(modified_path + '/F5/modules.py', F5_project_path + '/f5_tts/model/modules.py')
shutil.copyfile(modified_path + '/F5/dit.py', F5_project_path + '/f5_tts/model/backbones/dit.py')
shutil.copyfile(modified_path + '/F5/utils_infer.py', F5_project_path + '/f5_tts/infer/utils_infer.py')


from f5_tts.model import CFM, DiT
from f5_tts.infer.utils_infer import load_checkpoint


# ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider'] else keep empty.
ORT_Accelerate_Providers = ["CUDAExecutionProvider"]

DYNAMIC_AXES = True                     # Default dynamic_axes is input audio length. Note, some providers only work for static axes.
N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 1024                             # Number of FFT components for the STFT process
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT
MAX_SIGNAL_LENGTH = 2048                # Max frames for audio length after STFT processed
SAMPLE_RATE = 24000                     # The generated audio sample rate
RANDOM_SEED = 9527                      # Set seed to reproduce the generated audio
NFE_STEP = 32                           # F5-TTS model setting
CFG_STRENGTH = 2.0                      # F5-TTS model setting
SWAY_COEFFICIENT = -1.0                 # F5-TTS model setting
HIDDEN_SIZE = 1024                      # F5-TTS model setting
SPEED = 1.0                             # Set for talking speed. Only works with dynamic_axes=True
TARGET_RMS = 0.15                       # The root mean square value for the audio
HEAD_DIM = 64                           # F5-TTS Transformers model head_dim
AUDIO_LENGTH = 160000                   # Set for static axes export. Length of audio input signal in samples
TEXT_IDS_LENGTH = 60                    # Set for static axes export. Text_ids from [ref_text + gen_text]
MAX_GENERATED_LENGTH = 600              # Set for static axes export. Max signal features before passing to ISTFT
TEXT_EMBED_LENGTH = 512 + N_MELS        # Set for static axes export.
WINDOW_TYPE = 'kaiser'                  # Type of window function used in the STFT
REFERENCE_SIGNAL_LENGTH = AUDIO_LENGTH // HOP_LENGTH + 1  # Reference audio length after STFT processed
MAX_DURATION = REFERENCE_SIGNAL_LENGTH + MAX_GENERATED_LENGTH  # Set for static axes export. MAX_DURATION <= MAX_SIGNAL_LENGTH
if MAX_DURATION > MAX_SIGNAL_LENGTH:
    MAX_DURATION = MAX_SIGNAL_LENGTH


class F5Preprocess(torch.nn.Module):
    def __init__(self, f5_model, custom_stft, nfft=NFFT, n_mels=N_MELS, sample_rate=SAMPLE_RATE, head_dim=HEAD_DIM, target_rms=TARGET_RMS, hidden_size=HIDDEN_SIZE):
        super(F5Preprocess, self).__init__()
        self.f5_text_embed = f5_model.transformer.text_embed
        self.custom_stft = custom_stft
        self.num_channels = n_mels
        self.nfft = nfft
        self.target_sample_rate = sample_rate
        self.head_dim = head_dim
        self.base_rescale_factor = 1.0
        self.interpolation_factor = 1.0
        self.target_rms = target_rms
        self.hidden_size = hidden_size
        base = 10000.0 * self.base_rescale_factor ** (self.head_dim / (self.head_dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(MAX_SIGNAL_LENGTH, dtype=torch.float32), inv_freq) / self.interpolation_factor
        self.freqs = freqs.repeat_interleave(2, dim=-1).unsqueeze(0)
        self.rope_cos = self.freqs.cos().half()
        self.rope_sin = self.freqs.sin().half()
        self.fbank = (torchaudio.functional.melscale_fbanks(self.nfft // 2 + 1, 0, 12000, self.num_channels, self.target_sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0)

    def forward(self,
                audio: torch.FloatTensor,
                text_ids: torch.IntTensor,
                max_duration: torch.IntTensor
                ):
        audio = audio * self.target_rms / torch.sqrt(torch.mean(torch.square(audio)))
        mel_signal = self.custom_stft(audio).abs()
        mel_signal = torch.matmul(self.fbank, mel_signal).clamp(min=1e-5).log()
        mel_signal = mel_signal.transpose(1, 2)
        ref_signal_len = mel_signal.shape[1]
        mel_signal = torch.cat((mel_signal, torch.zeros((1, max_duration - ref_signal_len, self.num_channels), dtype=torch.float32)), dim=1)
        noise = torch.randn((1, max_duration, self.num_channels), dtype=torch.float32)
        rope_cos = self.rope_cos[:, :max_duration, :].float()
        rope_sin = self.rope_sin[:, :max_duration, :].float()
        cat_mel_text = torch.cat((mel_signal, self.f5_text_embed(torch.cat((text_ids + 1, torch.zeros((1, max_duration - text_ids.shape[-1]), dtype=torch.int32)), dim=-1), max_duration)), dim=-1)
        cat_mel_text_drop = torch.cat((torch.zeros((1, max_duration, self.num_channels), dtype=torch.float32), self.f5_text_embed(torch.zeros((1, max_duration), dtype=torch.int32), max_duration)), dim=-1)
        qk_rotated_empty = torch.zeros((2, max_duration, self.head_dim), dtype=torch.float32)
        return noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, ref_signal_len


class F5Transformer(torch.nn.Module):
    def __init__(self, f5_model, cfg=CFG_STRENGTH, steps=NFE_STEP, sway_coef=SWAY_COEFFICIENT):
        super(F5Transformer, self).__init__()
        self.f5_transformer = f5_model.transformer
        self.freq_embed_dim = 256
        self.cfg_strength = cfg
        self.steps = steps
        self.sway_sampling_coef = sway_coef
        t = torch.linspace(0, 1, self.steps + 1, dtype=torch.float32)
        time_step = t + self.sway_sampling_coef * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
        self.delta_t = torch.diff(time_step)
        self.time_expand = torch.zeros((1, self.steps, self.freq_embed_dim), dtype=torch.float32)
        half_dim = self.freq_embed_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
        for i in range(self.steps):
            emb = time_step[i] * emb_factor
            self.time_expand[:, i, :] = torch.cat((emb.sin(), emb.cos()), dim=-1)

    def forward(self,
                noise: torch.FloatTensor,
                rope_cos: torch.FloatTensor,
                rope_sin: torch.FloatTensor,
                cat_mel_text: torch.FloatTensor,
                cat_mel_text_drop: torch.FloatTensor,
                qk_rotated_empty: torch.FloatTensor,
                time_step: torch.IntTensor
                ):
        pred = self.f5_transformer(x=noise, cond=cat_mel_text, cond_drop=cat_mel_text_drop, time=self.time_expand[:, time_step], rope_cos=rope_cos, rope_sin=rope_sin, qk_rotated_empty=qk_rotated_empty)
        pred, pred1 = pred.chunk(2, dim=0)
        return noise + (pred + (pred - pred1) * self.cfg_strength) * self.delta_t[time_step]



class F5Decode(torch.nn.Module):
    def __init__(self, vocos, custom_istft, target_rms=TARGET_RMS):
        super(F5Decode, self).__init__()
        self.vocos = vocos
        self.custom_istft = custom_istft
        self.target_rms = target_rms

    def forward(self,
                denoised: torch.FloatTensor,
                ref_signal_len: torch.LongTensor
                ):
        denoised = denoised[:, ref_signal_len:, :].transpose(1, 2)
        denoised = self.vocos.decode(denoised)
        denoised = self.custom_istft(*denoised)
        generated_signal = denoised * self.target_rms / torch.sqrt(torch.mean(torch.square(denoised)))
        return generated_signal


def load_model(ckpt_path):
    F5TTS_model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
    )
    model = CFM(
        transformer=DiT(
            **F5TTS_model_cfg, text_num_embeds=vocab_size, mel_dim=N_MELS
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=SAMPLE_RATE,
            n_mel_channels=N_MELS,
            hop_length=HOP_LENGTH,
        ),
        odeint_kwargs=dict(
            method='euler',
        ),
        vocab_char_map=vocab_char_map,
    ).to('cpu')
    return load_checkpoint(model, ckpt_path, 'cpu', use_ema=True)


def is_chinese_char(c):
    cp = ord(c)
    return (
        0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or    # CJK Unified Ideographs Extension A
        0x20000 <= cp <= 0x2A6DF or  # CJK Unified Ideographs Extension B
        0x2A700 <= cp <= 0x2B73F or  # CJK Unified Ideographs Extension C
        0x2B740 <= cp <= 0x2B81F or  # CJK Unified Ideographs Extension D
        0x2B820 <= cp <= 0x2CEAF or  # CJK Unified Ideographs Extension E
        0xF900 <= cp <= 0xFAFF or    # CJK Compatibility Ideographs
        0x2F800 <= cp <= 0x2FA1F     # CJK Compatibility Ideographs Supplement
    )


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    merged_trans = str.maketrans({
        '“': '"', '”': '"', '‘': "'", '’': "'",
        ';': ','
    })
    chinese_punctuations = set("。，、；：？！《》【】—…")
    for text in text_list:
        char_list = []
        text = text.translate(merged_trans)
        for seg in jieba.cut(text):
            if seg.isascii():
                if char_list and len(seg) > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and all(is_chinese_char(c) for c in seg):
                pinyin_list = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in pinyin_list:
                    if c not in chinese_punctuations:
                        char_list.append(" ")
                    char_list.append(c)
            else:
                for c in seg:
                    if c.isascii():
                        char_list.append(c)
                    elif c in chinese_punctuations:
                        char_list.append(c)
                    else:
                        char_list.append(" ")
                        pinyin = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                        char_list.extend(pinyin)
        final_text_list.append(char_list)
    return final_text_list


def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1
):
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    text = torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text



print("\n\nStart to Export the F5-TTS Preprocess Part.")
# Dummy for Export the F5_Preprocess part
audio = torch.ones((1, 1, AUDIO_LENGTH), dtype=torch.float32)
text_ids = torch.ones((1, TEXT_IDS_LENGTH), dtype=torch.int32)
max_duration = torch.tensor(MAX_DURATION, dtype=torch.long)

with torch.inference_mode():
    f5_model = load_model(F5_safetensors_path)
    print("f5_mode", f5_model.type)
    custom_stft = STFT_Process(model_type='stft_A', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    f5_preprocess = F5Preprocess(f5_model, custom_stft)
    torch.onnx.export(
        f5_preprocess,
        (audio, text_ids, max_duration),
        onnx_model_A,
        input_names=['audio', 'text_ids', 'max_duration'],
        output_names=['noise', 'rope_cos', 'rope_sin', 'cat_mel_text', 'cat_mel_text_drop', 'qk_rotated_empty', 'ref_signal_len'],
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'text_ids': {1: 'text_ids_len'},
            'noise': {1: 'max_duration'},
            'rope_cos': {1: 'max_duration'},
            'rope_sin': {1: 'max_duration'},
            'cat_mel_text': {1: 'max_duration', 2: 'text_embed_len'},
            'cat_mel_text_drop': {1: 'max_duration',  2: 'text_embed_len'},
            'qk_rotated_empty': {1: 'max_duration'}
        } if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17)
    del custom_stft
    del f5_preprocess
    del audio
    del text_ids
    del max_duration
    gc.collect()

print("\nExport Done.")


print("\n\nStart to Export the F5-TTS Transformer Part.")
# Dummy for Export the F5_Transformer part
noise = torch.ones((1, MAX_DURATION, N_MELS), dtype=torch.float32)
rope_cos = torch.ones((1, MAX_DURATION, HEAD_DIM), dtype=torch.float32)
rope_sin = torch.ones((1, MAX_DURATION, HEAD_DIM), dtype=torch.float32)
cat_mel_text = torch.ones((1, MAX_DURATION, TEXT_EMBED_LENGTH), dtype=torch.float32)
cat_mel_text_drop = torch.ones((1, MAX_DURATION, TEXT_EMBED_LENGTH), dtype=torch.float32)
qk_rotated_empty = torch.ones((2, MAX_DURATION, HEAD_DIM), dtype=torch.float32)
time_step = torch.tensor(0, dtype=torch.int32)

with torch.inference_mode():
    scale_factor = math.pow(HEAD_DIM, -0.25)
    for i in range(len(f5_model.transformer.transformer_blocks)):
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.to_q.weight.data *= scale_factor
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.to_q.bias.data *= scale_factor
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.to_k.weight.data *= scale_factor
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.to_k.bias.data *= scale_factor
    f5_transformer = F5Transformer(f5_model)
    torch.onnx.export(
        f5_transformer,
        (noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, time_step),
        onnx_model_B,
        input_names=['noise', 'rope_cos', 'rope_sin', 'cat_mel_text', 'cat_mel_text_drop', 'qk_rotated_empty', 'time_step'],
        output_names=['denoised'],
        dynamic_axes={
            'noise': {1: 'max_duration'},
            'rope_cos': {1: 'max_duration'},
            'rope_sin': {1: 'max_duration'},
            'cat_mel_text': {1: 'max_duration', 2: 'text_embed_len'},
            'cat_mel_text_drop': {1: 'max_duration',  2: 'text_embed_len'},
            'qk_rotated_empty': {1: 'max_duration'},
            'denoised': {1: 'max_duration'}
        } if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17)
    del f5_transformer
    del noise
    del rope_cos
    del rope_sin
    del cat_mel_text
    del cat_mel_text_drop
    del qk_rotated_empty
    del time_step
    gc.collect()
    print("\nExport Done.")


print("\n\nStart to Export the F5-TTS Decode Part.")
# Dummy for Export the F5_Decode part
denoised = torch.ones((1, MAX_DURATION, N_MELS), dtype=torch.float32)
ref_signal_len = torch.tensor(REFERENCE_SIGNAL_LENGTH, dtype=torch.long)

with torch.inference_mode():
    custom_istft = STFT_Process(model_type='istft_A', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    # Vocos model preprocess
    vocos = Vocos.from_pretrained(vocos_model_path)
    vocos.backbone.norm.weight.data = (vocos.backbone.norm.weight.data * torch.sqrt(torch.tensor(vocos.backbone.norm.weight.data.shape[0], dtype=torch.float32))).view(1, -1, 1)
    vocos.backbone.norm.bias.data = vocos.backbone.norm.bias.data.view(1, -1, 1)
    vocos.backbone.final_layer_norm.weight.data = (vocos.backbone.final_layer_norm.weight.data * torch.sqrt(torch.tensor(vocos.backbone.final_layer_norm.weight.data.shape[0], dtype=torch.float32))).view(1, -1, 1)
    vocos.backbone.final_layer_norm.bias.data = vocos.backbone.final_layer_norm.bias.data.view(1, -1, 1)
    vocos.head.out.bias.data = vocos.head.out.bias.data.view(1, -1, 1)
    for i in range(len(vocos.backbone.convnext)):
        block = vocos.backbone.convnext._modules[f'{i}']
        block.norm.weight.data = (block.norm.weight.data * torch.sqrt(torch.tensor(block.norm.weight.data.shape[0], dtype=torch.float32))).view(1, -1, 1)
        block.norm.bias.data = block.norm.bias.data.view(1, -1, 1)
        block.pwconv1.weight.data = block.pwconv1.weight.data.unsqueeze(0)
        block.pwconv1.bias.data = block.pwconv1.bias.data.view(1, -1, 1)
        block.pwconv2.weight.data = (block.gamma.data.unsqueeze(-1) * block.pwconv2.weight.data).unsqueeze(0)
        block.pwconv2.bias.data = (block.gamma.data * block.pwconv2.bias.data).view(1, -1, 1)

    f5_decode = F5Decode(vocos, custom_istft)
    torch.onnx.export(
        f5_decode,
        (denoised, ref_signal_len),
        onnx_model_C,
        input_names=['denoised', 'ref_signal_len'],
        output_names=['output_audio'],
        dynamic_axes={
            'denoised': {1: 'max_duration'},
            'output_audio': {2: 'generated_len'},
        } if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17)
    del f5_decode
    del denoised
    del ref_signal_len
    del vocos
    del custom_istft
    gc.collect()
    print("\nExport Done.")


# ONNX Runtime settings
onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3  # error level, it a adjustable value.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0  # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL


#ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CUDAExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
in_name_A2 = in_name_A[2].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name
out_name_A3 = out_name_A[3].name
out_name_A4 = out_name_A[4].name
out_name_A5 = out_name_A[5].name
out_name_A6 = out_name_A[6].name


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CUDAExecutionProvider'])
# For DirectML + AMD GPU,
# pip install onnxruntime-directml --upgrade
# ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['DmlExecutionProvider'])

in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
in_name_B1 = in_name_B[1].name
in_name_B2 = in_name_B[2].name
in_name_B3 = in_name_B[3].name
in_name_B4 = in_name_B[4].name
in_name_B5 = in_name_B[5].name
in_name_B6 = in_name_B[6].name
out_name_B0 = out_name_B[0].name


#ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CUDAExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name


# Run F5-TTS by ONNX Runtime
audio, sr = torchaudio.load(reference_audio)

print("[DEBUG] audio.shape: {}, sr: {}".format(audio.shape, sr))

if sr != SAMPLE_RATE:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    audio = resampler(audio)

audio = audio.unsqueeze(0).numpy()

print("[DEBUG] audio.shape: {}".format(audio.shape))

zh_pause_punc = r"。，、；：？！"
ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
ref_audio_len = audio.shape[-1] // HOP_LENGTH + 1
max_duration = np.array(ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED), dtype=np.int64)
gen_text = convert_char_to_pinyin([ref_text + gen_text])
text_ids = list_str_to_idx(gen_text, vocab_char_map).numpy()
time_step = np.array(0, dtype=np.int32)

print("[DEBUG]: audio.dtype: {}, text_ids.dtype: {}, max_duration.dtype: {}".format(audio.dtype, text_ids.dtype, max_duration.dtype))

print("\n\nRun F5-TTS by ONNX Runtime.")
start_count = time.time()
noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, ref_signal_len = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2, out_name_A3, out_name_A4, out_name_A5, out_name_A6],
        {
            in_name_A0: audio,
            in_name_A1: text_ids,
            in_name_A2: max_duration
        })
while time_step < NFE_STEP:
    # print(f"NFE_STEP: {time_step}")
    noise = ort_session_B.run(
        [out_name_B0],
        {
            in_name_B0: noise,
            in_name_B1: rope_cos,
            in_name_B2: rope_sin,
            in_name_B3: cat_mel_text,
            in_name_B4: cat_mel_text_drop,
            in_name_B5: qk_rotated_empty,
            in_name_B6: time_step
        })[0]
    time_step += 1
generated_signal = ort_session_C.run(
        [out_name_C0],
        {
            in_name_C0: noise,
            in_name_C1: ref_signal_len
        })[0]
end_count = time.time()

# Save to audio
audio_tensor = torch.tensor(generated_signal).squeeze(0)
torchaudio.save(generated_audio, audio_tensor, SAMPLE_RATE)

if F5_project_path in sys.path:
    sys.path.remove(F5_project_path)

print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{end_count - start_count:.3f}")
