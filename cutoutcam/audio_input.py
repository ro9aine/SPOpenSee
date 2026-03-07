from threading import Lock
from typing import Any

import numpy as np

sd: Any = None

try:
    import sounddevice as _sd
except ImportError:
    pass
else:
    sd = _sd


class MicSpeechInput:
    def __init__(self, sample_rate: int = 16000, block_ms: int = 20):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self._lock = Lock()
        self._stream: Any | None = None
        self._last_rms = 0.0
        self._smooth_rms = 0.0
        self._hangover = 0
        self._running = False
        self._enabled = sd is not None

    @property
    def available(self) -> bool:
        return self._enabled

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        if not self._enabled:
            print("sounddevice is not installed. Microphone mouth control is disabled.")
            return False
        if sd is None:
            return False

        blocksize = max(1, int(self.sample_rate * self.block_ms / 1000))
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=self._on_audio,
            )
            self._stream.start()
            self._running = True
            return True
        except Exception as exc:
            print(f"Cannot start microphone input: {exc}")
            self._stream = None
            self._running = False
            return False

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            return
        samples = indata[:, 0]
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float32)))
        with self._lock:
            self._last_rms = rms

    def read_state(self, threshold: float = 0.08, gain: float = 20.0) -> tuple[bool, float]:
        with self._lock:
            rms = self._last_rms

        self._smooth_rms = self._smooth_rms * 0.75 + rms * 0.25
        energy = max(0.0, min(1.0, self._smooth_rms * max(0.1, gain)))

        speaking_now = energy >= max(0.01, threshold)
        if speaking_now:
            self._hangover = 6
        else:
            self._hangover = max(0, self._hangover - 1)

        is_speaking = speaking_now or self._hangover > 0
        return is_speaking, energy

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._running = False
