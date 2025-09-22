# TEN-VAD Realtime

**TEN-VAD Realtime** is a real-time Voice Activity Detection (VAD) and audio segmentation tool with WebSocket integration. It records audio, detects speech segments, saves them as WAV files, merges close segments, and sends events to a WebSocket server for real-time monitoring or integration with other systems.

## Features

- Real-time audio recording and speech detection using [TenVad](https://github.com/your-tenvad-link)
- Automatic segmentation and merging of speech segments
- Saves audio segments and merged files as WAV
- Logs segment timestamps and durations to a JSON file
- Sends real-time events (speech start, segment saved, merged) over WebSocket
- Simple WebSocket server for testing and integration

## Requirements

- Python 3.8+
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- [numpy](https://numpy.org/)
- [websocket-client](https://pypi.org/project/websocket-client/)
- [websockets](https://websockets.readthedocs.io/) (for server)
- [TenVad](https://github.com/your-tenvad-link) (custom VAD module)

Install dependencies:

```sh
pip install sounddevice numpy websocket-client websockets
```

## Usage

### 1. Start the WebSocket Server

Run the provided server to receive events:

```sh
python ws_client_test.py
```

You should see:

```
WebSocket server listening on ws://localhost:8765
```

### 2. Start the VAD Segmentation Script

In another terminal, run:

```sh
python ten_vad_segmentation.py
```

You should see:

```
🎙️ TEN-VAD streaming... speak now! (Ctrl+C to stop)
```

### 3. Speak into your microphone

- Speech segments will be detected, saved, and merged automatically.
- Events will be sent to the WebSocket server and printed in the server terminal.
- Segment and merge information is saved in `timestamps.json`.

### 4. Stop

Press `Ctrl+C` to stop recording. The script will finalize any pending segments and shut down cleanly.

## File Structure

- `ten_vad_segmentation.py` — Main VAD and segmentation script
- `ws_client_test.py` — Simple WebSocket server for testing
- `recordings/` — Saved raw speech segments (WAV)
- `merged/` — Merged speech segments (WAV)
- `timestamps.json` — Log of segment and merge times

## Customization

- Adjust VAD parameters (`THRESHOLD`, `SILENCE_TIMEOUT`, etc.) in `ten_vad_segmentation.py` as needed.
- Integrate your own WebSocket client or server for advanced workflows.

## Notes

- The `TenVad` module must be available in your Python path.
- This repo is for research and prototyping; production use may require further error handling and security.

## License

MIT License

---

**Contributions welcome!**