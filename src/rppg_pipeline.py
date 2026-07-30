import cv2
import mediapipe.python.solutions as mp_solutions
import numpy as np
from scipy.signal import butter, filtfilt
import threading
import time
import random
from collections import deque

class RPPGPipeline:
    """
    Real-time rPPG pipeline using OpenCV, MediaPipe Face Mesh, and SciPy.
    Runs in a background thread.
    """
    def __init__(self, fps=30, buffer_seconds=10):
        self.fps = fps
        self.buffer_size = fps * buffer_seconds
        
        # Buffers
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.green_signal = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # Face Mesh
        self.mp_face_mesh = mp_solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Forehead landmarks
        self.forehead_indices = [10, 338, 297, 332, 284]
        
        # HR tracking
        self.current_hr = None
        self.baseline_hr = None
        self.hr_history = deque(maxlen=10)
        self.quality = "Poor"
        
        # Thread control
        self.running = False
        self.thread = None
        self.cap = None
        
        # UI Display
        self.latest_frame = None

    def start(self):
        """Starts the webcam and processing thread."""
        if self.running:
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
            
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the pipeline and releases resources."""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        self.face_mesh.close()

    def _run_loop(self):
        """Main capture and processing loop."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            face_detected = False
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                # Extract forehead ROI
                forehead_pts = []
                for idx in self.forehead_indices:
                    pt = landmarks.landmark[idx]
                    forehead_pts.append((int(pt.x * w), int(pt.y * h)))
                
                # Create mask and extract mean green channel
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.array(forehead_pts), 255)
                mean_val = cv2.mean(frame_rgb, mask=mask)
                green_val = mean_val[1]
                
                self.green_signal.append(green_val)
                self.timestamps.append(time.time())
                face_detected = True
                
                # Draw ROI on frame for display
                cv2.polylines(frame_rgb, [np.array(forehead_pts)], True, (0, 255, 0), 2)
                
            if not face_detected:
                # Append 0 or previous to keep timing?
                if self.green_signal:
                    self.green_signal.append(self.green_signal[-1])
                else:
                    self.green_signal.append(0)
                self.timestamps.append(time.time())

            self._process_signal()
            
            # Save frame for UI
            self.latest_frame = frame_rgb
            time.sleep(1.0 / self.fps)

    def _process_signal(self):
        """Processes the buffer to estimate HR."""
        if len(self.green_signal) < self.buffer_size:
            self.quality = "Initializing"
            return
            
        # Signal quality check (e.g., too many static/zero values if face lost)
        signal = np.array(self.green_signal)
        if np.std(signal) < 1.0:
            self.quality = "Poor"
            return
            
        # Bandpass filter (0.7 - 4.0 Hz, corresponding to 42 - 240 BPM)
        nyq = 0.5 * self.fps
        low = 0.7 / nyq
        high = 4.0 / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        # FFT
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0/self.fps)
        fft_mag = np.abs(np.fft.rfft(filtered))
        
        # Find peak
        valid_idx = np.where((freqs >= 0.7) & (freqs <= 4.0))[0]
        if len(valid_idx) == 0:
            self.quality = "Poor"
            return
            
        peak_idx = valid_idx[np.argmax(fft_mag[valid_idx])]
        hr_hz = freqs[peak_idx]
        hr_bpm = hr_hz * 60.0
        
        # SNR check (simplified)
        peak_power = fft_mag[peak_idx]**2
        total_power = np.sum(fft_mag[valid_idx]**2)
        snr = peak_power / total_power if total_power > 0 else 0
        
        if snr > 0.3:
            self.quality = "Good"
            self.current_hr = hr_bpm
            self.hr_history.append(hr_bpm)
            if self.baseline_hr is None:
                self.baseline_hr = hr_bpm
        else:
            self.quality = "Poor"

    def get_current_hr(self):
        return self.current_hr
        
    def get_feature_vector(self):
        """Returns the dictionary format required by integration layer."""
        if self.current_hr is None:
            return {
                "hr_bpm": None,
                "hr_baseline_delta": None,
                "hr_trend_slope": None,
                "signal_quality": self.quality,
                "sbp_estimated": None,
                "dbp_estimated": None
            }
            
        # Calculate trend
        trend = 0.0
        if len(self.hr_history) > 1:
            x = np.arange(len(self.hr_history))
            y = np.array(self.hr_history)
            slope, _ = np.polyfit(x, y, 1)
            trend = slope
            
        delta = self.current_hr - self.baseline_hr if self.baseline_hr else 0
        
        # Simulated BP (just for placeholder mapping)
        sbp = 110 + (self.current_hr - 60) * 0.5
        dbp = 70 + (self.current_hr - 60) * 0.3
        
        return {
            "hr_bpm": float(self.current_hr),
            "hr_baseline_delta": float(delta),
            "hr_trend_slope": float(trend),
            "signal_quality": self.quality,
            "sbp_estimated": float(sbp),
            "dbp_estimated": float(dbp)
        }


class SimulatedRPPG:
    """Fake rPPG generator for demo mode without webcam."""
    def __init__(self, base_hr=75):
        self.base_hr = base_hr
        self.current_hr = base_hr
        self.running = False
        self.baseline_hr = base_hr
        self.history = deque(maxlen=10)
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run_loop(self):
        while self.running:
            # Random walk
            self.current_hr += random.uniform(-1.0, 1.0)
            self.history.append(self.current_hr)
            time.sleep(1)

    def get_current_hr(self):
        return self.current_hr

    def get_feature_vector(self):
        trend = 0.0
        if len(self.history) > 1:
            x = np.arange(len(self.history))
            y = np.array(self.history)
            slope, _ = np.polyfit(x, y, 1)
            trend = slope
            
        delta = self.current_hr - self.baseline_hr
        sbp = 110 + (self.current_hr - 60) * 0.5
        dbp = 70 + (self.current_hr - 60) * 0.3
        
        return {
            "hr_bpm": float(self.current_hr),
            "hr_baseline_delta": float(delta),
            "hr_trend_slope": float(trend),
            "signal_quality": "Good (Simulated)",
            "sbp_estimated": float(sbp),
            "dbp_estimated": float(dbp)
        }
