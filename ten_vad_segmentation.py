import numpy as np
import sounddevice as sd
import time
import wave
import os
from ten_vad import TenVad
from datetime import datetime

# Parameters
SAMPLE_RATE = 16000
HOP_SIZE = 256
THRESHOLD = 0.7
SILENCE_TIMEOUT = 1.0
OVERRIDE_TIMEOUT = 2.0  # merging window

RAW_DIR = "recordings"
MERGE_DIR = "merged"
os.makedirs(RAW_DIR, exist_ok = True)
os.makedirs(MERGE_DIR, exist_ok = True)

vad = TenVad(hop_size = HOP_SIZE, threshold = THRESHOLD)

# State
last_speech_time = None
is_recording = False
current_audio = []

segment_index = 0
pending_group = [] # segment filenames waiting to be merged
pending_close_time = None

start_time = time.time()   # track script runtime


def save_wav(filename, audio_data):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())


def read_wav(filename):
    with wave.open(filename, "rb") as rf:
        params = rf.getparams()
        audio = np.frombuffer(rf.readframes(rf.getnframes()), dtype = np.int16)
    return audio, params


def merge_wavs(files, out_file):
    merged_audio = []
    params = None
    for f in files:
        data, p = read_wav(f)
        merged_audio.append(data)
        if params is None:
            params = p
    merged_audio = np.concatenate(merged_audio)
    with wave.open(out_file, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(merged_audio.tobytes())


def finalize_pending():
    """Finalize pending group into a merged file (if >1 part)."""
    global pending_group
    if not pending_group:
        return
    if len(pending_group) == 1:
        # Just one file, keep it as is in RAW folder
        print(f"✅ Finalized single: {pending_group[0]}")
    else:
        merged_name = os.path.join(
            MERGE_DIR,
            "+".join([os.path.basename(p).replace(".wav", "") for p in pending_group]) + ".wav"
        )
        merge_wavs(pending_group, merged_name)
        print(f"🔗 Created merged file: {merged_name}")
        # Delete originals from RAW folder
        for p in pending_group:
            if os.path.exists(p):
                os.remove(p)
    pending_group = []

def audio_callback(indata, frames, t, status):
    global last_speech_time, is_recording, current_audio
    global segment_index, pending_group, pending_close_time

    if status:
        print("⚠️", status)

    audio_chunk = (indata[:, 0] * 32767).astype(np.int16)

    for start in range(0, len(audio_chunk), HOP_SIZE):
        frame = audio_chunk[start:start + HOP_SIZE]
        if len(frame) < HOP_SIZE:
            frame = np.pad(frame, (0, HOP_SIZE - len(frame)))

        prob, flag = vad.process(frame)

        # ✅ Always append the frame, whether speech or silence
        current_audio.extend(frame.tolist())

        if flag == 1:  # speech
            print(f"🟢 Speech detected (p={prob:.2f})")
            if not is_recording:
                print("🟢 Speech started")
                is_recording = True
                current_audio = []  # start fresh buffer
                current_audio.extend(frame.tolist())
            last_speech_time = time.time()

            if pending_close_time and (time.time() - pending_close_time) <= OVERRIDE_TIMEOUT:
                pending_close_time = None  # cancel pending finalize

        else:  # silence
            print(f"⚪ Silence (p={prob:.2f})")

            # If silence lasts longer than SILENCE_TIMEOUT
            if last_speech_time and time.time() - last_speech_time > SILENCE_TIMEOUT:
                if is_recording and len(current_audio) > 0:
                    # Save segment (includes speech + silence)
                    audio_data = np.array(current_audio, dtype = np.int16)
                    segment_index += 1
                    filename = os.path.join(RAW_DIR, f"segment_{segment_index}.wav")
                    save_wav(filename, audio_data)
                    print(f"💾 Saved {filename}")

                    pending_group.append(filename)
                    pending_close_time = time.time()

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
                # Check if pending group should be finalized
                if pending_close_time and (time.time() - pending_close_time) > OVERRIDE_TIMEOUT:
                    finalize_pending()
                    pending_close_time = None
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        finalize_pending()  # finalize leftovers

        total_runtime = time.time() - start_time
        print(f"⏱️ Total runtime: {total_runtime:.2f} seconds "
              f"({total_runtime/60:.2f} minutes)")