# system_listener.py
import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

# Load the Vosk model
model = Model("model")  # Make sure the Vosk model folder is named "model"
samplerate = 16000
recognizer = KaldiRecognizer(model, samplerate)
recognizer.SetWords(True)
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    q.put(bytes(indata))

def listen_and_transcribe():
    print("🎧 Listening to system audio via VoiceMeeter B1...")

    try:
        device_index = 24  # 🔁 Replace this with your actual B1 index
        sd.default.device = device_index
        device_info = sd.query_devices(device_index)
        print(f"✅ Using device #{device_index}: {device_info['name']}")

        results = []

        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print("📝 Transcribed:", text)
                        results.append(text)
                        return text  # return first meaningful sentence

    except Exception as e:
        print("❌ Error:", str(e))
        return ""
