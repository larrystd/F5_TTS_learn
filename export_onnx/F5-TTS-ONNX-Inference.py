import re
import sys
import time
import os
import jieba
import numpy as np
import torch
import onnxruntime
import soundfile as sf
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style

from dotenv import load_dotenv
load_dotenv()

onnx_model_A         = os.getenv("onnx_model_A")
onnx_model_B         = os.getenv("onnx_model_B")
onnx_model_C         = os.getenv("onnx_model_C")

reference_audio      = "./assets/wgs-f5tts_mono.wav"
ref_text             = "那到时候再给你打电话，麻烦你注意接听。"
generated_audio      = "./output/onnx_test3.wav"
gen_text             = "这点请您放心，估计是我的号码被标记了，请问您是沈沈吗？"

# ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider'], else keep empty.
ORT_Accelerate_Providers = ["CUDAExecutionProvider"]
# ORT_Accelerate_Providers = ["OpenVINOExecutionProvider"]

HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT
SAMPLE_RATE = 24000                     # The generated audio sample rate
RANDOM_SEED = 9527                      # Set seed to reproduce the generated audio
NFE_STEP = 32                           # F5-TTS model setting
SPEED = 1.2                             # Set for talking speed. Only works with dynamic_axes=True


with open(os.getenv("vocab_path"), "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)


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


# ONNX Runtime settings
onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3  # error level, it a adjustable value.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0  # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
# ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CUDAExecutionProvider'])
# ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['OpenVINOExecutionProvider'])
model_type = ort_session_A._inputs_meta[0].type
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


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers)
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


ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
# ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CUDAExecutionProvider'])
# ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['OpenVINOExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name


# Load the input audio
#print(f"\nReference Audio: {reference_audio}")
audio = np.array(AudioSegment.from_file(reference_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
audio_len = len(audio)
audio = audio.astype(np.float32) / 32768.0
if "float16" in model_type:
    audio = audio.astype(np.float16)
audio = audio.reshape(1, 1, -1)

zh_pause_punc = r"。，、；：？！"
ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
ref_audio_len = audio_len // HOP_LENGTH + 1
max_duration = np.array(ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED), dtype=np.int64)
gen_text = convert_char_to_pinyin([ref_text + gen_text])
text_ids = list_str_to_idx(gen_text, vocab_char_map).numpy()
time_step = np.array(0, dtype=np.int32)

#print("\n\nRun F5-TTS by ONNX Runtime.")
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
generated_signal = generated_signal.astype(np.float32).reshape(-1)
sf.write(generated_audio, generated_signal, SAMPLE_RATE, format='FLAC')
print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{end_count - start_count:.3f}")
