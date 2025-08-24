from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch, soundfile as sf, librosa
import os, glob

# 载入预训练 Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval().cuda()

def extract_w2v_feats(wav_path):
    # 强制以 16k 读取并自动重采样
    wav, sr = librosa.load(wav_path, sr=16000)  # sr=16000 会自动重采样
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.cuda())[0]  # (B, T', H)
    return outputs.squeeze(0).cpu()  # (T', H)

# 批量处理
for wav in glob.glob("../data/react_clean/*/Audio_files/**/*.wav", recursive=True):
    feats = extract_w2v_feats(wav)              # torch.Tensor [T',768]
    save_pth = wav.replace("Audio_files", "W2V_features").replace(".wav", ".pth")
    os.makedirs(os.path.dirname(save_pth), exist_ok=True)
    torch.save(feats, save_pth)

print("Feature extraction with librosa-based resampling complete!")  
