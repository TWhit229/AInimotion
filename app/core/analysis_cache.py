"""
Background analysis cache — pre-analyzes queued videos on CPU
while the GPU is busy interpolating.
"""

from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue, Empty

from .frame_analysis import AnalysisResult, analyze_video


class AnalysisCache:
    """
    Background worker that pre-analyzes videos for duplicate/scene-cut detection.

    While a video is being interpolated on the GPU, this runs analyze_video()
    on the next queued videos using CPU only. When the GPU finishes and picks
    up the next video, the analysis is already done — no idle gap.

    Thread-safe: submit/get/cancel can be called from the UI thread while
    the worker runs in the background.
    """

    MAX_CACHED = 5  # max results to keep in memory (~13MB each worst case)

    def __init__(self, max_height: int | None = 720):
        self.max_height = max_height
        self._cache: dict[str, AnalysisResult] = {}
        self._lock = threading.Lock()
        self._work_queue: Queue[str | None] = Queue()
        self._cancel_events: dict[str, threading.Event] = {}
        self._shutdown = False

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def submit(self, path: str | Path):
        """Queue a video for background analysis."""
        key = str(path)
        with self._lock:
            if key in self._cache:
                return  # already analyzed
            if key in self._cancel_events:
                return  # already queued
            self._cancel_events[key] = threading.Event()
        self._work_queue.put(key)

    def get(self, path: str | Path) -> AnalysisResult | None:
        """Get cached analysis result, or None if not ready yet."""
        key = str(path)
        with self._lock:
            return self._cache.get(key)

    def cancel(self, path: str | Path):
        """Cancel pending analysis for a video."""
        key = str(path)
        with self._lock:
            event = self._cancel_events.get(key)
            if event:
                event.set()

    def evict(self, path: str | Path):
        """Remove cached result and cancel any pending analysis."""
        key = str(path)
        with self._lock:
            self._cache.pop(key, None)
            event = self._cancel_events.pop(key, None)
            if event:
                event.set()  # stop running analysis from storing result

    def update_max_height(self, max_height: int | None):
        """Clear cache and cancel running analyses if resolution changes."""
        if max_height != self.max_height:
            self.max_height = max_height
            with self._lock:
                self._cache.clear()
                for event in self._cancel_events.values():
                    event.set()
                self._cancel_events.clear()

    def shutdown(self):
        """Stop the worker thread."""
        self._shutdown = True
        # Cancel all pending
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
        self._work_queue.put(None)  # sentinel to unblock worker
        self._thread.join(timeout=5)

    def _worker_loop(self):
        while not self._shutdown:
            try:
                key = self._work_queue.get(timeout=1)
            except Empty:
                continue

            if key is None:  # shutdown sentinel
                break

            with self._lock:
                cancel_event = self._cancel_events.get(key)
                if cancel_event is None:
                    continue  # evicted before worker picked it up
                if cancel_event.is_set():
                    self._cancel_events.pop(key, None)
                    continue
                if key in self._cache:
                    continue

            # Run analysis (CPU-only, can take seconds to minutes)
            try:
                result = analyze_video(
                    key,
                    max_height=self.max_height,
                    cancel_event=cancel_event,
                )

                with self._lock:
                    if not (cancel_event and cancel_event.is_set()):
                        self._cache[key] = result

                        # Evict oldest if over limit
                        while len(self._cache) > self.MAX_CACHED:
                            oldest = next(iter(self._cache))
                            del self._cache[oldest]

                    self._cancel_events.pop(key, None)

            except Exception as e:
                print(f"  [!] Background analysis failed for {Path(key).name}: {e}")
                with self._lock:
                    self._cancel_events.pop(key, None)
