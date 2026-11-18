"""SignBridge - lõi nhận diện của máy chủ web (không phụ thuộc FastAPI, để thử riêng được). Điện thoại/máy tính mở trang, gửi hình camera lên đây;
máy chủ chạy ĐÚNG engine nhận diện đang dùng trên laptop (train_engine.py + main_launcher.py) rồi trả chữ về."""
import os
import sys
import json
import time
import queue
import zipfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "stub_streamlit"))   # Streamlit giả (chỉ dùng trên máy chủ, nằm cuối đường tìm để không che Streamlit thật)
os.environ["VSL_HEADLESS"] = "1"

# Dữ liệu mẫu được nén thành 1 file zip (dễ tải lên) -> giải nén 1 lần trước khi nạp engine
_zip = os.path.join(HERE, "dataset_vsl.zip")
if os.path.isfile(_zip) and not os.path.isdir(os.path.join(HERE, "dataset_vsl")):
    with zipfile.ZipFile(_zip) as z:
        z.extractall(HERE)

import numpy as np
import cv2

import main_launcher as ml          # chỉ nạp hàm (HEADLESS=1: không dựng giao diện)
te = ml.te

CONF_THRESHOLD = ml.CONF_THRESHOLD
SENTENCE_TIMEOUT = ml.SENTENCE_TIMEOUT
LANG_VI = next(lg for lg in ml.LANGS if lg["key"] == "vi")
GRID_DT = 1.0 / 30.0                # engine học trên video ~30 khung/giây -> mọi luồng được quy về 30 khung/giây đều
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "3"))
TOLERANCE = float(os.environ.get("VSL_TOLERANCE", "1.0"))

ENGINE = {"bundle": None, "err": "", "ready": False}


def load_engine():
    try:
        ENGINE["bundle"] = te.load_templates_cached()
        ENGINE["ready"] = bool(ENGINE["bundle"]["actions"])
        if not ENGINE["ready"]:
            ENGINE["err"] = "Chưa có bộ mẫu từ nào (thiếu dataset_vsl hoặc vsl_embed.npz)."
    except Exception as e:
        ENGINE["err"] = f"Không nạp được bộ nhận diện: {e}"


threading.Thread(target=load_engine, daemon=True).start()


class Resampler:
    """Quy dòng khung hình tới không đều (mạng) về lưới 30 khung/giây bằng nội suy tuyến tính,
    để các ngưỡng 'số khung' của SegmentDetector giữ đúng ý nghĩa như trên laptop."""

    def __init__(self):
        self.t0 = None
        self.f0 = None
        self.p0 = False
        self.last_g = None

    def push(self, t, feat, present):
        if self.t0 is None or t - self.t0 > 0.5 or t <= self.t0:
            first = self.t0 is None or t - self.t0 > 0.5
            self.t0, self.f0, self.p0, self.last_g = t, feat, present, t
            return [(feat, present)] if first else []
        out = []
        g = self.last_g + GRID_DT
        while g <= t + 1e-9:
            a = (g - self.t0) / (t - self.t0)
            if present and self.p0:
                both = (self.f0 != 0) & (feat != 0)
                near = feat if a >= 0.5 else self.f0
                f = np.where(both, self.f0 + (feat - self.f0) * a, near).astype(np.float32)
                out.append((f, True))
            else:
                out.append((feat, present) if a >= 0.5 else (self.f0, self.p0))
            self.last_g = g
            g += GRID_DT
        self.t0, self.f0, self.p0 = t, feat, present
        return out


class Session:
    """Một người dùng = một Session: giữ bộ tách ký hiệu, bộ đệm từ, đếm giờ kết câu - giống vòng lặp camera trên laptop."""

    def __init__(self, emit):
        self.emit = emit                       # emit(dict) -> gửi về trình duyệt
        self.detector = te.SegmentDetector()
        self.shoulder = te.ShoulderRef()
        self.rs = Resampler()
        self.buf, self.alts = [], []
        self.last_speak = time.time()
        self.paused = False
        self.no_shoulder = 0
        self.out_q = queue.Queue()
        self.pending = 0
        self.last_state = None
        self.stamps = []          # thời điểm các khung đã xử lý (để báo tốc độ thật)

    # ---- lệnh từ giao diện
    def command(self, cmd):
        if cmd == "clear":
            self.buf, self.alts = [], []
            te.clear_recent()
        elif cmd == "pause":
            self.paused = not self.paused
            self.detector.reset()

    def fps(self):
        now = time.time()
        self.stamps = [x for x in self.stamps if now - x < 3.0]
        return int(round(len(self.stamps) / 3.0)) if len(self.stamps) > 5 else 0

    # ---- một khung hình đã trích đặc trưng (giữ tách riêng để thử không cần camera)
    def feed_features(self, t, feat, present):
        for f, p in self.rs.push(t, feat, present):
            self._step(f, p)

    def _step(self, feat, present):
        segment = None if self.paused else self.detector.push(feat, present)
        if self.paused:
            self.detector.reset()
        if self.detector.recording:
            self.last_speak = time.time()
        if segment is not None:
            self._classify(segment)

    def _classify(self, segment):
        bundle = ENGINE["bundle"]
        if not bundle:
            return
        try:
            word, info = te.classify_segment(None, [], bundle, segment, tolerance=TOLERANCE, min_conf=CONF_THRESHOLD)
        except Exception as e:
            word, info = None, {"reason": f"lỗi nhận diện: {e}", "candidate": None, "ratio": None}
        try:
            ml.log_recognition(segment, word, info)
        except Exception:
            pass
        if word:
            self.buf.append(word)
            self.alts.append([x for x in info.get("suggest", []) if x != word][:2])
            te.note_recent(word)
            self.last_speak = time.time()
            self.emit({"t": "word", "w": word})
        else:
            self.emit({"t": "skip", "why": info.get("reason") or "không chắc"})

    # ---- gọi định kỳ (kể cả khi không có khung hình): kết câu, nhận câu đã dịch, báo trạng thái
    def tick(self):
        idle = time.time() - self.last_speak
        if self.buf and not self.detector.recording and idle > SENTENCE_TIMEOUT:
            words, alts = list(self.buf), list(self.alts)
            self.buf, self.alts = [], []
            te.clear_recent()
            vi = ml.build_vietnamese_sentence(words)
            threading.Thread(target=ml._translate_worker, args=(words, vi, LANG_VI, self.out_q, None, alts), daemon=True).start()
            self.pending += 1
        while True:
            try:
                text, _tts, source, err = self.out_q.get_nowait()
            except queue.Empty:
                break
            self.pending = max(0, self.pending - 1)
            self.emit({"t": "sentence", "text": text, "src": source})
        idle = time.time() - self.last_speak
        left = None if (not self.buf or self.detector.recording) else max(0.0, SENTENCE_TIMEOUT - idle)
        st = {"t": "st", "rec": bool(self.detector.recording), "words": list(self.buf),
              "left": None if left is None else round(left, 1), "paused": self.paused,
              "busy": self.pending > 0, "ready": ENGINE["ready"], "err": ENGINE["err"], "fps": self.fps()}
        key = json.dumps({k: v for k, v in st.items()}, sort_keys=True)
        if key != self.last_state:
            self.last_state = key
            self.emit(st)


def worker(sess, frames, stop, emit):
    """Luồng xử lý riêng của 1 người dùng (MediaPipe Holistic không dùng chung giữa các luồng)."""
    holistic = te.create_holistic()
    try:
        while not stop.is_set():
            try:
                t, data = frames.get(timeout=0.1)
            except queue.Empty:
                t = None
            if t is not None:
                try:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        h, w = img.shape[:2]
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        results = holistic.process(rgb)
                        feat = te.extract_keypoints(results, w / h, sess.shoulder)
                        present = te.keypoints_present(feat)
                        if te.has_hand(results) and not present:
                            sess.no_shoulder += 1
                            if sess.no_shoulder == 20:
                                emit({"t": "hint", "text": "Thấy tay nhưng chưa thấy hai vai. Hãy lùi ra để camera thấy nửa thân trên."})
                        else:
                            sess.no_shoulder = 0
                        sess.stamps.append(time.time())
                        sess.feed_features(t, feat, present)
                except Exception as e:
                    emit({"t": "hint", "text": f"Lỗi xử lý hình: {e}"})
            try:
                sess.tick()
            except Exception as e:
                emit({"t": "hint", "text": f"Lỗi: {e}"})
    finally:
        try:
            holistic.close()
        except Exception:
            pass


