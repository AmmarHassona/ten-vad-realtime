import numpy as np 
import sounddevice as sd
import time
import wave
import os
from ten_vad import TenVad
import json
import threading
import queue
import websocket

from inference import predict_endpoint # inference predict_endpoint

# Parameters
SAMPLE_RATE = 16000
HOP_SIZE = 256
THRESHOLD = 0.7
SILENCE_TIMEOUT = 1.5
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
pending_group = []  # segment filenames waiting to be merged
pending_close_time = None

start_time = time.time()   # track script runtime
segment_start_time = None

# store timestamps for all segments and merged files
segment_times = {}

# json output file (only for logging not manually passed to websocket)
TIMESTAMP_FILE = "timestamps.json"

# WebSocket
WS_URL = "ws://localhost:8765"

# queue for messages to be sent over WebSocket
_ws_queue = queue.Queue()
_ws_stop_event = threading.Event()

def ws_sender_loop():
    """Background thread: connect to WS_URL and send queued messages."""
    while not _ws_stop_event.is_set():
        try:
            ws = websocket.create_connection(WS_URL, timeout = 5)
            print(f"🌐 WebSocket connected to {WS_URL}")
            while not _ws_stop_event.is_set():
                try:
                    msg = _ws_queue.get(timeout = 1)  # blocking with timeout
                except queue.Empty:
                    continue
                try:
                    ws.send(msg)
                except Exception as e:
                    print("⚠️ WS send failed:", e)
                    try: ws.close()
                    except: pass
                    break
            try: ws.close()
            except: pass
        except Exception as e:
            print("⚠️ WS connection error:", e)
            time.sleep(2)

def send_ws_event(event_type, payload):
    """Push JSON message to websocket queue (non-blocking)."""
    message = {"event": event_type, "data": payload}
    try:
        _ws_queue.put_nowait(json.dumps(message))
    except Exception as e:
        print("⚠️ Failed to enqueue WS message:", e)

_ws_thread = threading.Thread(target = ws_sender_loop, daemon = True)
_ws_thread.start()

def save_timestamps():
    with open(TIMESTAMP_FILE, "w", encoding = "utf-8") as f:
        json.dump(segment_times, f, indent = 4)

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
        print(f"✅ Finalized single: {pending_group[0]}")
    else:
        merged_name = os.path.join(
            MERGE_DIR,
            "+".join([os.path.basename(p).replace(".wav", "") for p in pending_group]) + ".wav"
        )
        merge_wavs(pending_group, merged_name)
        print(f"🔗 Created merged file: {merged_name}")

        # compute absolute start/end times
        first_seg = pending_group[0]
        last_seg = pending_group[-1]
        start_abs = segment_times.get(os.path.basename(first_seg), {}).get("start")
        end_abs = segment_times.get(os.path.basename(last_seg), {}).get("end")

        if start_abs is not None and end_abs is not None:
            duration = end_abs - start_abs
            print(f"⏱️ Merged absolute time range: {start_abs:.2f}s → {end_abs:.2f}s "
                  f"(duration {duration:.2f}s)")
            segment_times[os.path.basename(merged_name)] = {
                "start": start_abs, "end": end_abs, "duration": duration
            }
            save_timestamps()
            send_ws_event("merged", {
                "merged_file": os.path.basename(merged_name),
                "parts": [os.path.basename(p) for p in pending_group],
                "start": start_abs, "end": end_abs, "duration": duration
            })

        for p in pending_group:
            if os.path.exists(p): os.remove(p)
    pending_group = []

def audio_callback(indata, frames, t, status):
    global last_speech_time, is_recording, current_audio
    global segment_index, pending_group, pending_close_time
    global segment_start_time

    if status: print("⚠️", status)
    audio_chunk = (indata[:, 0] * 32767).astype(np.int16)

    for start in range(0, len(audio_chunk), HOP_SIZE):
        frame = audio_chunk[start:start + HOP_SIZE]
        if len(frame) < HOP_SIZE:
            frame = np.pad(frame, (0, HOP_SIZE - len(frame)))

        prob, flag = vad.process(frame)
        current_audio.extend(frame.tolist())

        if flag == 1:  # speech
            if not is_recording:
                print("🟢 Speech started")
                is_recording = True
                current_audio = []  
                current_audio.extend(frame.tolist())
                segment_start_time = time.time() - start_time
                send_ws_event("speech_start", {"timestamp": segment_start_time})
            last_speech_time = time.time()
            if pending_close_time and (time.time() - pending_close_time) <= OVERRIDE_TIMEOUT:
                pending_close_time = None

        else:  # silence
            if last_speech_time and time.time() - last_speech_time > SILENCE_TIMEOUT:
                if is_recording and len(current_audio) > 0:
                    # 🔹 Smart Turn check before saving
                    float_audio = np.array(current_audio[-SAMPLE_RATE*4:], dtype=np.float32) / 32767.0
                    result = predict_endpoint(float_audio)
                    print(f"🤖 Smart Turn raw: {result}")

                    if result["prediction"] == 0:   # incomplete
                        print("🤖 Smart Turn: Incomplete → continue listening")
                        return
                    else:
                        print("🤖 Smart Turn: Complete → finalize segment")

                    # Save segment
                    audio_data = np.array(current_audio, dtype = np.int16)
                    segment_index += 1
                    filename = os.path.join(RAW_DIR, f"segment_{segment_index}.wav")
                    save_wav(filename, audio_data)
                    print(f"💾 Saved {filename}")

                    segment_end_time = time.time() - start_time
                    duration = segment_end_time - segment_start_time
                    segment_times[os.path.basename(filename)] = {
                        "start": segment_start_time, "end": segment_end_time, "duration": duration
                    }
                    save_timestamps()
                    send_ws_event("segment_saved", {
                        "file": os.path.basename(filename),
                        "start": segment_start_time,
                        "end": segment_end_time,
                        "duration": duration
                    })
                    pending_group.append(filename)
                    pending_close_time = time.time()

                is_recording = False
                current_audio = []

if __name__ == "__main__":
    print("🎙️ TEN-VAD + Smart Turn streaming...")
    try:
        with sd.InputStream(callback = audio_callback,
                            channels = 1,
                            samplerate = SAMPLE_RATE,
                            blocksize = HOP_SIZE):
            while True:
                if pending_close_time and (time.time() - pending_close_time) > OVERRIDE_TIMEOUT:
                    finalize_pending()
                    pending_close_time = None
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        finalize_pending()
        total_runtime = time.time() - start_time
        segment_times["__summary__"] = {
            "total_runtime_seconds": total_runtime,
            "total_runtime_minutes": total_runtime / 60
        }
        save_timestamps()

    _ws_stop_event.set()
    try: _ws_queue.put_nowait(json.dumps({"event":"shutdown","data":{}}))
    except: pass
    _ws_thread.join(timeout=2)
    print("✅ Exiting.")