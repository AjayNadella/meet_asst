import queue
import sounddevice as sd
import tempfile
import os
import numpy as np
import time
from faster_whisper import WhisperModel
from pydub import AudioSegment

# Load the Whisper model
model = WhisperModel("tiny.en", compute_type="int8")
samplerate = 16000
channels = 1
blocksize = 4000  # Smaller chunk size for quicker detection
q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    q.put(bytes(indata))

def get_device_index_by_name(name_substring: str):
    for i, dev in enumerate(sd.query_devices()):
        if name_substring.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
            return i
    raise Exception(f"❌ No input device found matching: {name_substring}")

def listen_and_transcribe_():
    print("🎧 Listening to system audio via Whisper")

    try:
        device_index = get_device_index_by_name("Voicemeeter Out B1")
        sd.default.device = device_index
        device_info = sd.query_devices(device_index)
        print(f"✅ Using device #{device_index}: {device_info['name']}")

        silence_threshold = 100
        max_silence_chunks = 6
        silence_count = 0
        audio_chunks = []
        start_time = time.time()

        with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize, dtype='int16',
                               channels=channels, callback=callback):
            while True:
                data = q.get()
                rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(np.float32) ** 2))

                audio_chunks.append(data)

                if rms < silence_threshold:
                    silence_count += 1
                else:
                    silence_count = 0

                # Stop recording after sustained silence and a minimum duration
                if silence_count >= max_silence_chunks and (time.time() - start_time) > 2.5:
                    # Add 2 more chunks after silence to avoid premature cutoffs
                    for _ in range(2):
                        try:
                            audio_chunks.append(q.get(timeout=0.3))
                        except queue.Empty:
                            break
                    break

        audio_bytes = b"".join(audio_chunks)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_path = f.name
            audio = AudioSegment(
                data=audio_bytes,
                sample_width=2,
                frame_rate=samplerate,
                channels=channels
            )
            audio.export(f.name, format="wav")

        segments, _ = model.transcribe(temp_path)
        result = " ".join([seg.text for seg in segments]).strip()
        print("🧠 Transcribed:", result)

        os.remove(temp_path)
        return result

    except Exception as e:
        print("❌ Error:", str(e))
        return ""
