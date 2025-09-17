import numpy as np
import sounddevice as sd
import time
import wave
import os
from ten_vad import TenVad

# Parameters
SAMPLE_RATE = 16000
HOP_SIZE = 256
THRESHOLD = 0.7
SILENCE_TIMEOUT = 1.0
OUTPUT_DIR = "recordings"
OVERRIDE_TIMEOUT = SILENCE_TIMEOUT + 2  # merge window

os.makedirs(OUTPUT_DIR, exist_ok = True)

vad = TenVad(hop_size = HOP_SIZE, threshold = THRESHOLD)

# State
last_speech_time = None
is_recording = False
current_audio = []

segment_index = 0
last_saved_file = None
last_saved_time = None

# Stats
original_files = []
merged_files = []


def save_wav(filename, audio_data):
    """Save int16 numpy array as .wav file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())


def append_to_wav(filename, audio_data):
    """Append int16 numpy array to existing .wav file"""
    with wave.open(filename, "rb") as rf:
        params = rf.getparams()
        old_frames = rf.readframes(rf.getnframes())

    # Merge old and new audio
    new_audio = np.frombuffer(old_frames, dtype = np.int16)
    combined = np.concatenate([new_audio, audio_data])

    # Overwrite with combined
    with wave.open(filename, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(combined.tobytes())


def audio_callback(indata, frames, t, status):
    global last_speech_time, is_recording, current_audio
    global segment_index, last_saved_file, last_saved_time
    global original_files, merged_files

    if status:
        print("⚠️", status)

    # Convert float32 (-1..1) → int16
    audio_chunk = (indata[:, 0] * 32767).astype(np.int16)

    # Process in hop-size frames
    for start in range(0, len(audio_chunk), HOP_SIZE):
        frame = audio_chunk[start : start + HOP_SIZE]
        if len(frame) < HOP_SIZE:
            frame = np.pad(frame, (0, HOP_SIZE - len(frame)))

        prob, flag = vad.process(frame)

        if flag == 1:  # speech detected
            print(f"🟢 Speech detected (p={prob:.2f})")
            if not is_recording:
                print("🟢 Speech started")
                is_recording = True
                current_audio = []
            current_audio.extend(frame.tolist())
            last_speech_time = time.time()

        else:  # silence detected
            print(f"⚪ Silence (p={prob:.2f})")
            if is_recording:
                current_audio.extend(frame.tolist())

            # Check if silence lasted too long
            if last_speech_time and time.time() - last_speech_time > SILENCE_TIMEOUT:
                if is_recording and len(current_audio) > 0:
                    audio_data = np.array(current_audio, dtype=np.int16)

                    # Check override merge condition
                    now = time.time()
                    if last_saved_file and last_saved_time and (now - last_saved_time) <= OVERRIDE_TIMEOUT:
                        print(f"🔗 Merging into {last_saved_file}")
                        append_to_wav(last_saved_file, audio_data)
                        merged_files.append(last_saved_file)
                    else:
                        segment_index += 1
                        filename = os.path.join(OUTPUT_DIR, f"segment_{segment_index}.wav")
                        save_wav(filename, audio_data)
                        last_saved_file = filename
                        original_files.append(filename)
                        print(f"💾 Saved {filename}")

                    last_saved_time = now

                is_recording = False
                current_audio = []


if __name__ == "__main__":
    print("🎙️ TEN-VAD streaming... speak now! (Ctrl+C to stop)")
    try:
        with sd.InputStream(callback = audio_callback,
                            channels = 1,
                            samplerate = SAMPLE_RATE,
                            blocksize = HOP_SIZE):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        print("\n📊 Recording Summary:")
        print(f"  🎤 Original files saved: {len(original_files)}")
        for f in original_files:
            print(f"    - {f}")
        print(f"  🔗 Files merged into previous: {len(merged_files)}")
        for f in merged_files:
            print(f"    - {f}")