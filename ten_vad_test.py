import numpy as np
import sounddevice as sd
import time
from ten_vad import TenVad

SAMPLE_RATE = 16000
HOP_SIZE = 256
THRESHOLD = 0.7
SILENCE_TIMEOUT = 1.5  # seconds

vad = TenVad(hop_size=HOP_SIZE, threshold=THRESHOLD)
last_speech_time = time.time()

def audio_callback(indata, frames, t, status):
    global last_speech_time
    if status:
        print("⚠️", status)

    audio_chunk = (indata[:, 0] * 32767).astype(np.int16)

    for start in range(0, len(audio_chunk), HOP_SIZE):
        frame = audio_chunk[start:start+HOP_SIZE]
        if len(frame) < HOP_SIZE:
            frame = np.pad(frame, (0, HOP_SIZE - len(frame)))

        prob, flag = vad.process(frame)

        if flag == 1:
            print(f"🟢 Speech detected (p={prob:.2f})")
            last_speech_time = time.time()
        else:
            print(f"⚪ Silence (p={prob:.2f})")

        if time.time() - last_speech_time > SILENCE_TIMEOUT:
            print("⏹️ Conversation ended due to silence.")
            raise sd.CallbackStop


if __name__ == "__main__":
    print("🎙️ TEN-VAD running in real time... speak now! (Ctrl+C to stop)")
    try:
        with sd.InputStream(callback=audio_callback,
                            channels=1,
                            samplerate=SAMPLE_RATE,
                            blocksize=HOP_SIZE):
            while True:
                time.sleep(0.1)  # keep main thread alive
    except sd.CallbackStop:
        print("✅ Stream closed, exiting program.")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")