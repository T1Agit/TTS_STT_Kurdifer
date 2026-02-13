import torch, soundfile as sf, os
from transformers import VitsModel, AutoTokenizer

TRAIN_DIR = "D:\\TTS_STT_Kurdifer\\training"
OUT_DIR = TRAIN_DIR + "\\test_output"
os.makedirs(OUT_DIR, exist_ok=True)

tests = [
    "Rojbaş, navê min Agît e",
    "Ez Kurdî diaxivim",
    "Kurdistan welatê min e",
    "Em ê nebin",
    "Hêvîdarim ku hûn baş in",
]

for label, path in [
    ("original", "facebook/mms-tts-kmr-script_latin"),
    ("step1000", TRAIN_DIR + "\\checkpoints\\step_1000"),
]:
    print()
    print("=== Testing: " + label + " ===")
    try:
        model = VitsModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model.eval()

        for i, text in enumerate(tests):
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs)
            wav = output.waveform.squeeze().numpy()
            fname = label + "_" + str(i) + ".wav"
            sf.write(OUT_DIR + "\\" + fname, wav, 16000)
            print("  " + fname + " | " + text)

        del model, tokenizer
    except Exception as e:
        print("  FAILED: " + str(e))

print()
print("=== WAV files saved to: " + OUT_DIR + " ===")
print("Listen and compare original vs step1000!")
