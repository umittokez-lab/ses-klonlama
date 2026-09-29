import os
import torch
import gradio as gr
from TTS.api import TTS
import re
import numpy as np
import scipy.io.wavfile as wavfile

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Ses klonlama modeli yükleniyor...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def split_text(text, max_chars=250):
    """Metni noktalardan bölerek küçük parçalara ayırır."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def clone_and_speak_long(audio_file, text, language, progress=gr.Progress()):
    if not audio_file:
        return None, "Lütfen bir referans ses kaydı yükleyin!"
    if not text.strip():
        return None, "Lütfen okutmak istediğiniz metni girin!"

    # Metni parçalara böl
    chunks = split_text(text)
    temp_files = []
    
    progress(0, desc="Seslendirme başlatılıyor...")
    
    # Her parçayı tek tek seslendir
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        temp_path = f"temp_{i}.wav"
        tts.tts_to_file(
            text=chunk,
            speaker_wav=audio_file,
            language=language,
            file_path=temp_path
        )
        temp_files.append(temp_path)
        progress((i + 1) / len(chunks), desc=f"İşleniyor: {i+1}/{len(chunks)} parça")

    # Ses parçalarını tek dosyada birleştir
    combined_audio = []
    sample_rate = 24000
    
    for file in temp_files:
        sr, data = wavfile.read(file)
        sample_rate = sr
        combined_audio.append(data)
        os.remove(file) # Geçici dosyayı sil
        
    final_audio = np.concatenate(combined_audio)
    output_path = "output_long_cloned.wav"
    wavfile.write(output_path, sample_rate, final_audio)
    
    return output_path, f"Başarılı! {len(chunks)} parça birleştirildi, uzun ses dosyası hazır."

# Web Arayüzü
demo = gr.Interface(
    fn=clone_and_speak_long,
    inputs=[
        gr.Audio(type="filepath", label="Referans Ses Kaydınız (.wav / .mp3)"),
        gr.Textbox(lines=10, placeholder="10-20 dakikalık uzun metninizi buraya yapıştırın...", label="Okunacak Uzun Metin"),
        gr.Dropdown(choices=["tr", "en", "es", "de", "fr", "it"], value="tr", label="Dil Seçimi")
    ],
    outputs=[
        gr.Audio(label="Üretilen Uzun Ses Çıktısı"),
        gr.Textbox(label="Sistem Durumu")
    ],
    title="🎙️ AI Uzun Metin Ses Klonlama Stüdyosu"
)

if __name__ == "__main__":
    demo.launch()
