"""
train_engine.py - Engine nhận diện ngôn ngữ ký hiệu (VSL) cho HÀNG NGHÌN TỪ, chỉ cần 1 video/từ.

Ý tưởng (đã đo thử trên dữ liệu thật, xem danh_gia.py):
  1. Đặc trưng mỗi khung hình: HÌNH DẠNG bàn tay (chia cho cỡ bàn tay) + QUỸ ĐẠO cổ tay (trừ vị trí trung bình)
     + chút vị trí tuyệt đối. Nên đứng/ngồi, xa/gần, đặt tay lệch vẫn cho cùng đặc trưng. Khung mất tay được nội suy;
     ký hiệu một tay luôn quy về một tay (người thuận tay trái/phải đều khớp).
  2. Từ 1 video, sinh hàng chục biến thể "người khác làm" (đổi tốc độ, xoay 3 chiều, tay to/nhỏ, ngón dài/ngắn,
     biên độ, rung, mất khung hình, lật tay...). Học một phép chiếu LDA: giữ lại những hướng KHÁC NHAU GIỮA CÁC TỪ
     và bỏ những hướng chỉ khác nhau giữa các lần/người làm. Mỗi video thành 1 điểm "nguyên mẫu" trong không gian nhúng.
  3. Nhận diện = nhúng đoạn ký hiệu rồi so cosine với các nguyên mẫu (1 phép nhân ma trận, tức thì kể cả 5000 video).
     Từ có nhiều video (Bắc/Trung/Nam...) có nhiều nguyên mẫu, lấy điểm cao nhất.
  4. Thêm từ/video mới: tính nguyên mẫu ngay (không cần học lại); học lại toàn bộ chỉ mất vài phút (không có LSTM).
"""
import glob
import json
import os
import re
import shutil
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------ Cấu hình ------------------------------ #
SEQ_LEN = 30                   # số khung hình mỗi chuỗi
NUM_FEATURES = 126             # 2 tay x 21 điểm x 3 tọa độ (dữ liệu thô lưu trong dataset_vsl)
REPR_DIM = 128                 # đặc trưng mỗi khung: hình dạng + quỹ đạo + vị trí
REPR_VERSION = 6               # đổi khi cách tính đặc trưng/mô hình thay đổi (mô hình cũ tự bị bỏ, dữ liệu thô vẫn dùng)
FEATURE_VERSION = 3            # định dạng dữ liệu thô .npy trong dataset_vsl
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
MIN_SHOULDER_WIDTH = 0.05
ABS_W = 0.3                    # trọng số vị trí tuyệt đối cổ tay
TRAJ_W = 1.5                   # trọng số quỹ đạo cổ tay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_vsl")
EMBED_PATH = os.path.join(BASE_DIR, "vsl_embed.npz")        # mô hình nhúng + nguyên mẫu
DTW_CACHE_PATH = os.path.join(BASE_DIR, "vsl_dtw.npz")      # đặc trưng DTW đã tính sẵn: để chạy trên máy chủ KHÔNG cần thư mục dataset_vsl
OLD_FILES = ("vsl_model.h5", "vsl_labels.json", "vsl_meta.json", "vsl_templates.pkl")   # file của bản LSTM cũ
ACTIVE_PATH = os.path.join(BASE_DIR, "vsl_active_words.txt")   # (tùy chọn) danh sách từ đang dùng, mỗi dòng 1 từ
DATA_MARKER = os.path.join(DATA_DIR, "_format.json")
LEGACY_MOVED = None

MIN_CLASSES = 2                # số từ tối thiểu để nhận diện
MIN_HAND_FRAMES = 8            # đoạn ký hiệu phải có tối thiểu chừng này khung thấy tay (và thấy vai)

AUG_K = 24                     # số biến thể sinh từ mỗi video khi học (ít hơn nếu có rất nhiều video)
AUG_K_INC = 12                 # số biến thể khi thêm 1 video mới (không học lại)
PCA_MAX = 600
EMB_MAX = 500
LDA_REG = 1e-2
MIN_SCORE = 0.30               # sàn tuyệt đối của độ tương đồng cosine (an toàn khi chưa hiệu chỉnh được)

_rng = np.random.default_rng()
_train_lock = threading.Lock()
_mp_holistic = mp.solutions.holistic



def ensure_dataset_format():
    """Dữ liệu .npy cũ được lưu theo cách chuẩn hóa khác (không có vai) nên KHÔNG được trộn với dữ liệu mới.
    Nếu phát hiện, cất cả thư mục cũ sang dataset_vsl_old_<thời gian> (không xóa) rồi tạo thư mục mới."""
    global LEGACY_MOVED
    current = False
    try:
        with open(DATA_MARKER, "r", encoding="utf-8") as f:
            current = json.load(f).get("format") == FEATURE_VERSION
    except Exception:
        current = False
    if os.path.isdir(DATA_DIR) and not current:
        has_data = any(os.path.isdir(os.path.join(DATA_DIR, d)) for d in os.listdir(DATA_DIR))
        if has_data:
            backup = f"{DATA_DIR}_old_{int(time.time())}"
            os.rename(DATA_DIR, backup)
            LEGACY_MOVED = os.path.basename(backup)
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_MARKER):
        with open(DATA_MARKER, "w", encoding="utf-8") as f:
            json.dump({"format": FEATURE_VERSION}, f)


ensure_dataset_format()


# ------------------------------ Trích xuất đặc trưng (tương đối so với vai) ------------------------------ #
def create_holistic():
    """Dùng CHUNG cho cả lúc train lẫn lúc chạy camera để đặc trưng đồng nhất."""
    return _mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


class ShoulderRef:
    """Gốc tọa độ = trung tâm hai vai: shoulder_center = (left_shoulder + right_shoulder) / 2,
    và độ rộng vai (để quy đổi xa/gần). Làm mượt bằng trung bình trượt để vai rung không làm tay rung theo.
    Mất Pose trong vài khung hình thì dùng lại vị trí vai gần nhất. Dùng 1 đối tượng cho 1 luồng video."""
    ALPHA = 0.4

    def __init__(self):
        self.reset()

    def reset(self):
        self.center = None   # (x, y) đã nhân tỉ lệ khung hình
        self.width = None

    def update(self, results, aspect=1.0):
        pose = results.pose_landmarks
        if pose is not None:
            ls, rs = pose.landmark[LEFT_SHOULDER], pose.landmark[RIGHT_SHOULDER]
            left = np.array([ls.x * aspect, ls.y], dtype=np.float64)
            right = np.array([rs.x * aspect, rs.y], dtype=np.float64)
            width = float(np.linalg.norm(left - right))
            if width >= MIN_SHOULDER_WIDTH:
                center = (left + right) / 2.0
                if self.center is None:
                    self.center, self.width = center, width
                else:
                    a = self.ALPHA
                    self.center = a * center + (1 - a) * self.center
                    self.width = a * width + (1 - a) * self.width
        return self.center is not None


def _hand_to_vec(hand, aspect, ref):
    if hand is None:
        return np.zeros(63, dtype=np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float64)
    pts[:, 0] *= aspect                 # đưa x (và z) về cùng đơn vị với y -> không méo khi video dọc/ngang khác nhau
    pts[:, 2] *= aspect
    pts[:, :2] -= ref.center            # trừ trung tâm hai vai (x, y)
    pts /= ref.width                    # đơn vị mới: "độ rộng vai"
    return pts.astype(np.float32).flatten()


def extract_keypoints(results, aspect=1.0, ref=None):
    """Kết quả MediaPipe Holistic -> vector 126 số (tay trái 63 + tay phải 63, thiếu tay = 0).
    Mọi điểm ngón tay được trừ đi shoulder_center của khung hình hiện tại rồi chia cho độ rộng vai.
    (Chỉ trừ x, y: z của bàn tay MediaPipe vốn đã tính so với cổ tay, còn z của vai lấy mốc ở hông, trừ vào sẽ
    làm lệch đúng sự khác biệt ngồi/đứng mà ta muốn loại bỏ.)
    aspect = rộng/cao của khung hình; ref = ShoulderRef dùng chung cho cả luồng. Không thấy vai -> vector 0."""
    ref = ref if ref is not None else ShoulderRef()
    if not ref.update(results, aspect):
        return np.zeros(NUM_FEATURES, dtype=np.float32)
    return np.concatenate([_hand_to_vec(results.left_hand_landmarks, aspect, ref),
                           _hand_to_vec(results.right_hand_landmarks, aspect, ref)])


def keypoints_present(feat):
    """Khung hình này có dữ liệu tay hợp lệ (thấy tay VÀ thấy vai) không."""
    return bool(np.any(feat))


def has_hand(results):
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def extract_video_features(video_path, max_frames=900):
    """Đọc video -> (mảng (T,126), mảng bool T: khung có tay, số khung thấy tay nhưng KHÔNG thấy vai)."""
    cap = cv2.VideoCapture(video_path)
    feats, present, no_shoulder = [], [], 0
    ref = ShoulderRef()
    try:
        with create_holistic() as holistic:
            while cap.isOpened() and len(feats) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * 640 / w)))
                    h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                feat = extract_keypoints(results, w / h, ref)
                feats.append(feat)
                present.append(keypoints_present(feat))
                if has_hand(results) and not keypoints_present(feat):
                    no_shoulder += 1
    finally:
        cap.release()
    if not feats:
        return np.zeros((0, NUM_FEATURES), np.float32), np.zeros(0, bool), 0
    return np.stack(feats), np.array(present, dtype=bool), no_shoulder


def extract_active_segment(feats, present):
    """Cắt từ lúc tay xuất hiện đầu tiên đến lần cuối. Trả về None nếu quá ít."""
    idx = np.where(present)[0]
    if len(idx) < MIN_HAND_FRAMES:
        return None
    return feats[idx[0]: idx[-1] + 1].astype(np.float32)


def _resample(seq, n, pos=None):
    """Nội suy tuyến tính chuỗi (T,126) về n khung hình (pos: vị trí lấy mẫu, mặc định đều)."""
    if len(seq) == 1:
        return np.repeat(seq, n, axis=0).astype(np.float32)
    if pos is None:
        pos = np.linspace(0, len(seq) - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(seq) - 1)
    w = (pos - i0)[:, None]
    return (seq[i0] * (1 - w) + seq[i1] * w).astype(np.float32)


def to_invariant(seq):
    """Chuỗi thô (T,126: tọa độ 2 tay, tương đối so với vai) -> chuỗi (T,128) bất biến với cách đứng/ngồi, xa/gần
    và khoảng cách đặt tay. Mỗi tay gồm:
      - 60 số HÌNH DẠNG: 20 điểm ngón tay trừ cổ tay, chia cho cỡ bàn tay (trung vị cả đoạn ký hiệu)
      -  2 số QUỸ ĐẠO: vị trí cổ tay trừ vị trí cổ tay trung bình của cả đoạn (tay đặt cao/thấp/lệch vẫn giống nhau)
      -  2 số VỊ TRÍ tuyệt đối (trọng số nhỏ) để vẫn phân biệt ký hiệu chỉ khác chỗ đặt tay."""
    x = np.asarray(seq, dtype=np.float64).reshape(-1, 2, 21, 3)
    T = x.shape[0]
    out = np.zeros((T, 2, 64), dtype=np.float64)
    for h in range(2):
        pts = x[:, h]
        present = np.abs(pts).sum(axis=(1, 2)) > 0
        if not present.any():
            continue
        rel = pts - pts[:, :1, :]
        r = rel[present]
        size_f = 0.5 * (np.linalg.norm(r[:, 9], axis=1) + np.linalg.norm(r[:, 5] - r[:, 17], axis=1))
        size = max(float(np.median(size_f)), 1e-3)
        shape = np.clip(rel[:, 1:, :] / size, -4.0, 4.0).reshape(T, 60)
        wrist = pts[:, 0, :2]
        traj = (wrist - wrist[present].mean(axis=0)) * TRAJ_W
        feat = np.concatenate([shape, traj, wrist * ABS_W], axis=1)
        feat[~present] = 0.0
        out[:, h] = feat
    return out.reshape(T, REPR_DIM).astype(np.float32)




# ------------------------------ Danh sách từ ------------------------------ #
def safe_label(name):
    name = re.sub(r'[\\/:*?"<>|]', "", name).strip().strip(".")
    return re.sub(r"\s+", " ", name)


def list_actions():
    """Danh sách từ đã có dữ liệu (quét thư mục bằng os.listdir, đã sắp xếp)."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d))
                  and any(f.endswith(".npy") for f in os.listdir(os.path.join(DATA_DIR, d))))


def load_active_words():
    """Danh sách từ ĐANG DÙNG (file vsl_active_words.txt). Không có file = dùng tất cả từ đã có dữ liệu.
    Nhiều từ mà mỗi từ chỉ 1-2 video thì nhận diện rất dễ nhầm; chỉ bật những từ cần thiết sẽ chính xác hơn hẳn."""
    try:
        with open(ACTIVE_PATH, "r", encoding="utf-8-sig") as f:
            words = [safe_label(l) for l in f if l.strip() and not l.strip().startswith("#")]
    except OSError:
        return None
    return {w.lower() for w in words if w} or None


def active_actions():
    allw = list_actions()
    act = load_active_words()
    return allw if act is None else [a for a in allw if a.lower() in act]


def set_active_words(words):
    with open(ACTIVE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(words) + "\n")


def add_active_word(label):
    """Dạy từ mới trong app: nếu đang dùng danh sách từ thì tự thêm từ này vào để nó được nhận diện."""
    act = load_active_words()
    if act is not None and label.lower() not in act:
        with open(ACTIVE_PATH, "a", encoding="utf-8") as f:
            f.write(label + "\n")


def count_samples(label):
    d = os.path.join(DATA_DIR, label)
    return len([f for f in os.listdir(d) if f.endswith(".npy")]) if os.path.isdir(d) else 0


def _load_segments(action):
    return [np.load(f) for f in sorted(glob.glob(os.path.join(DATA_DIR, action, "*.npy")))]




# ------------------------------ Đặc trưng ------------------------------ #
def fill_gaps(raw):
    """Nội suy các khung mất tay (toàn số 0) nằm giữa các khung có tay; đầu/đuôi lấy khung gần nhất."""
    x = np.asarray(raw, dtype=np.float32).reshape(-1, 2, 63).copy()
    T = len(x)
    pos = np.arange(T)
    for h in range(2):
        pres = np.abs(x[:, h]).sum(axis=1) > 0
        n = int(pres.sum())
        if n == 0 or n == T:
            continue
        idx = np.where(pres)[0]
        r = np.searchsorted(idx, pos)
        right = idx[np.clip(r, 0, n - 1)]
        left = idx[np.clip(r - 1, 0, n - 1)]
        span = (right - left).astype(np.float32)
        w = np.where(span > 0, (pos - left) / np.maximum(span, 1.0), 0.0).astype(np.float32)
        x[:, h] = x[left, h] * (1 - w[:, None]) + x[right, h] * w[:, None]
    return x.reshape(T, NUM_FEATURES)


def mirror_raw(raw):
    """Lật gương: đổi tay trái <-> phải và đảo trục x (quanh trung tâm vai)."""
    x = np.asarray(raw, dtype=np.float32).reshape(-1, 2, 21, 3)[:, ::-1].copy()
    pres = np.abs(np.asarray(raw).reshape(-1, 2, 63)).sum(axis=2)[:, ::-1] > 0
    x[..., 0] = -x[..., 0]
    x = np.where(pres[:, :, None, None], x, 0.0)
    return x.reshape(-1, NUM_FEATURES).astype(np.float32)


def canon(raw):
    """Ký hiệu chủ yếu bằng tay trái -> lật về tay phải: người thuận tay trái/phải đều khớp nhau."""
    p = (np.abs(np.asarray(raw).reshape(-1, 2, 63)).sum(axis=2) > 0).mean(axis=0)
    if p[0] > p[1] * 1.5 and p[0] > 0.3:
        return mirror_raw(raw)
    return raw


TRIM_REST = True               # cắt đoạn tay còn nghỉ/hạ xuống ở đầu-đuôi (cả lúc học lẫn lúc nhận diện) để mọi video cùng "cắt" như nhau
TRIM_DY = 0.6                  # khung nào có tay cao nhất thấp hơn mức cao nhất của đoạn quá chừng này (độ rộng vai) thì coi là tay nghỉ


def trim_rest(seg):
    """Bỏ các khung tay hạ xuống/nghỉ ở đầu và đuôi đoạn ký hiệu. Video bộ giáo dục và webcam bắt đầu/kết thúc ở
    những chỗ khác nhau (webcam mờ nên mất tay sớm hơn); cắt cùng một quy tắc cho cả hai để chúng khớp nhau.
    Ký hiệu làm ở thấp (ít chênh độ cao) thì không bị cắt gì."""
    if not TRIM_REST:
        return seg
    x = np.asarray(seg, dtype=np.float32)
    T = len(x)
    if T <= 2 * MIN_HAND_FRAMES:
        return seg
    w = x.reshape(T, 2, 21, 3)[:, :, 0, 1]                    # độ cao (y) cổ tay mỗi tay
    pres = np.abs(x.reshape(T, 2, 63)).sum(axis=2) > 0
    ym = np.where(pres, w, np.inf).min(axis=1)                # tay nào cao hơn (y nhỏ hơn)
    ok = np.isfinite(ym)
    if ok.sum() < MIN_HAND_FRAMES:
        return seg
    top = float(np.percentile(ym[ok], 10))
    act = np.where(ok & (ym < top + TRIM_DY))[0]
    if len(act) < MIN_HAND_FRAMES or len(act) < 0.4 * T:
        return seg
    s, e = max(0, int(act[0]) - 1), min(T, int(act[-1]) + 2)
    return seg[s:e] if e - s >= MIN_HAND_FRAMES else seg


def make_features(seg):
    """Đoạn ký hiệu thô (T,126) -> (30,128). Dùng cho CẢ lúc học lẫn lúc chạy camera."""
    r = canon(fill_gaps(trim_rest(np.asarray(seg, dtype=np.float32))))
    return to_invariant(_resample(np.asarray(r, dtype=np.float32), SEQ_LEN))


prepare_sequence = make_features


# ------------------------------ Tăng cường: giả lập "người khác làm" ------------------------------ #
def _rot(ax, a):
    c, s = np.cos(a), np.sin(a)
    if ax == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _warp_time(x, rng, lo, hi, crop_lo):
    T = len(x)
    if T > MIN_HAND_FRAMES and crop_lo < 1:
        crop = max(MIN_HAND_FRAMES, int(rng.uniform(crop_lo, 1.0) * T))
        st = int(rng.integers(0, T - crop + 1))
        x = x[st:st + crop]
        T = len(x)
    sp = np.interp(np.linspace(0, 1, T - 1), np.linspace(0, 1, 5), rng.uniform(lo, hi, 5))
    pos = np.concatenate([[0.0], np.cumsum(sp)])
    pos = pos / pos[-1] * (T - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (pos - i0)[:, None, None, None]
    return x[i0] * (1 - w) + x[i1] * w


def perturb_raw(raw, rng, level=1.0, mirror_p=0.25):
    """Biến thể 'người khác + camera khác': đổi tốc độ, cắt đầu/đuôi, đứng hình/rớt khung, ngón dài/ngắn, tay to/nhỏ,
    méo/xiên bàn tay, xoay 3 chiều, biên độ cử động, lệch vị trí, rung, nhiễu chậm, che đầu ngón, mất khung hình,
    lật tay. level 1.0 = mức vừa; 1.5 = khó."""
    x = np.asarray(raw, dtype=np.float64).reshape(-1, 2, 21, 3).copy()
    x = _warp_time(x, rng, 1 - 0.45 * level, 1 + 0.45 * level, 1 - 0.25 * level)
    T = len(x)
    pres = np.abs(x).sum(axis=(2, 3)) > 1e-6
    stut = np.where(rng.random(T) < 0.08 * level)[0]          # đứng hình: lặp lại khung trước
    for i in stut:
        if i > 0:
            x[i], pres[i] = x[i - 1], pres[i - 1]
    gs = 1 + rng.uniform(-0.2, 0.25) * min(level, 1.2)         # tay dài/ngắn, người to/nhỏ
    goff = rng.uniform(-0.25, 0.25, 2) * min(level, 1.2)
    out = np.zeros_like(x)
    for h in range(2):
        if not pres[:, h].any():
            continue
        pts = x[:, h].copy()
        wr = pts[:, 0:1, :].copy()
        rel = pts - wr
        for f in range(5):
            idx = [1 + f * 4 + j for j in range(4)]
            rel[:, idx] = rel[:, idx] * (1 + rng.uniform(-0.12, 0.12) * level)
        rel *= 1 + rng.uniform(-0.15, 0.15) * level
        A = np.eye(2) + rng.uniform(-0.15, 0.15, (2, 2)) * level   # bàn tay bị méo / xiên
        rel[..., :2] = rel[..., :2] @ A.T
        R = (_rot(2, rng.uniform(-0.35, 0.35) * level) @ _rot(0, rng.uniform(-0.3, 0.3) * level)
             @ _rot(1, rng.uniform(-0.3, 0.3) * level))
        rel = rel @ R.T
        nz = np.cumsum(rng.normal(0, 0.006 * level, (T, 21, 3)), axis=0)   # nhiễu phi cứng biến thiên chậm
        rel += nz - nz.mean(axis=0)
        for tip in (4, 8, 12, 16, 20):                              # đầu ngón bị che -> nhận sai về khớp sát
            if rng.random() < 0.25 * min(level, 1.2):
                rel[:, tip] = rel[:, tip - 1] + rng.normal(0, 0.01, (T, 3))
        center = wr[pres[:, h], 0, :2].mean(axis=0)
        amp = 1 + rng.uniform(-0.3, 0.3) * level
        wr2 = wr.copy()
        wr2[:, 0, :2] = center + (wr[:, 0, :2] - center) * amp + rng.uniform(-0.2, 0.2, 2) * level
        jit = np.cumsum(rng.normal(0, 0.004 * level, (T, 2)), axis=0)
        wr2[:, 0, :2] += jit - jit.mean(axis=0)
        o = (rel + wr2 + rng.normal(0, 0.012 * level, rel.shape)) * gs
        o[..., :2] += goff
        out[:, h] = o
    for h in range(2):
        if pres[:, h].any() and rng.random() < 0.7:
            n = int(rng.uniform(0.05, 0.25 * level + 0.05) * T)
            st = int(rng.integers(0, max(1, T - n)))
            out[st:st + n, h] = 0
    out = np.where(pres[:, :, None, None], out, 0.0)
    if rng.random() < mirror_p:
        out = out[:, ::-1].copy()
        out[..., 0] = -out[..., 0]
    return out.reshape(T, NUM_FEATURES).astype(np.float32)


def aug_basic(seg, rng):
    """Kiểu tăng cường đơn giản (đổi tốc độ, xoay phẳng, co giãn, dịch, nhiễu, lật tay) -> (30,126)."""
    seg = np.asarray(seg, dtype=np.float32)
    L = len(seg)
    if L > MIN_HAND_FRAMES:
        crop = max(MIN_HAND_FRAMES, int(rng.uniform(0.75, 1.0) * L))
        st = int(rng.integers(0, L - crop + 1))
        seg = seg[st:st + crop]
    sp = np.interp(np.linspace(0, 1, SEQ_LEN - 1), np.linspace(0, 1, 5), rng.uniform(0.6, 1.4, 5))
    pos = np.concatenate([[0.0], np.cumsum(sp)])
    pos = pos / pos[-1] * (len(seg) - 1)
    x = _resample(seg, SEQ_LEN, pos).reshape(SEQ_LEN, 2, 21, 3).astype(np.float64)
    pres = np.abs(x).sum(axis=(2, 3)) > 0
    a = rng.uniform(-0.25, 0.25)
    c, s = np.cos(a), np.sin(a)
    px, py = x[..., 0].copy(), x[..., 1].copy()
    x[..., 0], x[..., 1] = c * px - s * py, s * px + c * py
    x = x * rng.uniform(0.9, 1.1)
    x[..., :2] += rng.uniform(-0.12, 0.12, 2)
    x = x + rng.normal(0, 0.01, x.shape)
    if rng.random() < 0.25:
        x = x[:, ::-1].copy()
        x[..., 0] = -x[..., 0]
        pres = pres[:, ::-1]
    x = np.where(pres[:, :, None, None], x, 0.0)
    return x.reshape(SEQ_LEN, NUM_FEATURES).astype(np.float32)


def augment_raw(seg, rng):
    """Một biến thể ngẫu nhiên của đoạn ký hiệu thô (trộn 2 kiểu tăng cường để bền hơn)."""
    if rng.random() < 0.4:
        return aug_basic(seg, rng)
    return perturb_raw(seg, rng, level=float(rng.uniform(0.3, 1.3)), mirror_p=0.25)


# ------------------------------ Dữ liệu ------------------------------ #
# ------------------------------ Quét dữ liệu ------------------------------ #
def _scan_dataset():
    """{từ: [tên file .npy đã sắp xếp]} (chỉ từ có ít nhất 1 file)."""
    out = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for d in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, d)
        if not os.path.isdir(p):
            continue
        fs = sorted(f for f in os.listdir(p) if f.endswith(".npy"))
        if fs:
            out[d] = fs
    return out


def _load_seg(word, fname):
    try:
        seg = np.load(os.path.join(DATA_DIR, word, fname))
    except Exception:
        return None
    if seg.ndim != 2 or seg.shape[1] != NUM_FEATURES or len(seg) < 2:
        return None
    return seg.astype(np.float32)


def save_sample(label, seg):
    """Lưu thêm 1 đoạn ký hiệu thô (T,126) cho 1 từ (dùng khi người dùng xác nhận 'đúng là từ này' trên camera)."""
    label = safe_label(label)
    if not label:
        raise ValueError("Tên từ không hợp lệ.")
    d = os.path.join(DATA_DIR, label)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"cam_{int(time.time() * 1000)}.npy"), np.asarray(seg, dtype=np.float32))
    add_active_word(label)


# ------------------------------ Mô hình nhúng ------------------------------ #
class EmbedModel:
    """mu, P: PCA (3840 -> p). W: LDA (p -> m). protos: nguyên mẫu đã chuẩn hóa (n,m), xếp theo từ."""

    def __init__(self, mu, P, W, protos, proto_word, proto_key, cal):
        self.mu, self.P, self.W = mu, P, W
        self.protos = protos
        self.proto_word = [str(w) for w in proto_word]
        self.proto_key = [str(k) for k in proto_key]
        self.cal = cal

    def embed_feats(self, F):
        Z = ((F - self.mu) @ self.P) @ self.W
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def embed(self, seg):
        return self.embed_feats(make_features(seg).reshape(1, -1))[0]

    def save(self, path=EMBED_PATH):
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, mu=self.mu, P=self.P, W=self.W, protos=self.protos,
                 proto_word=np.array(self.proto_word, dtype=str), proto_key=np.array(self.proto_key, dtype=str),
                 cal=np.array(json.dumps(self.cal)), version=np.array(REPR_VERSION))
        os.replace(tmp, path)


_EMBED_CACHE = {"stamp": None, "model": None}


def _load_embed():
    """Đọc vsl_embed.npz (có nhớ tạm theo thời gian sửa file). Không có / sai phiên bản -> None."""
    try:
        st = os.stat(EMBED_PATH)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    if _EMBED_CACHE["stamp"] == stamp:
        return _EMBED_CACHE["model"]
    try:
        with np.load(EMBED_PATH, allow_pickle=False) as d:
            if int(d["version"]) != REPR_VERSION:
                return None
            model = EmbedModel(d["mu"], d["P"], d["W"], d["protos"], d["proto_word"], d["proto_key"],
                               json.loads(str(d["cal"])))
    except Exception:
        return None
    _EMBED_CACHE["stamp"], _EMBED_CACHE["model"] = stamp, model
    return model


def _word_starts(words):
    """Với danh sách từ đã xếp liền nhau: (danh sách từ duy nhất, chỉ số bắt đầu từng từ)."""
    uniq, starts = [], []
    for i, w in enumerate(words):
        if not uniq or uniq[-1] != w:
            uniq.append(w)
            starts.append(i)
    return uniq, np.array(starts, dtype=np.int64)


def _calibrate(model, protos, rng, n_check=1200):
    """Đo điểm cosine của mẫu 'đúng từ' và 'từ khác' trên các biến thể mới -> ngưỡng chấp nhận."""
    words = model.proto_word
    uniq, starts = _word_starts(words)
    widx = {w: i for i, w in enumerate(uniq)}
    pick = rng.permutation(len(protos))[:min(len(protos), n_check)]
    gen, imp = [], []
    for i in pick:
        w, _key, seg = protos[i]
        F = np.stack([make_features(augment_raw(seg, rng)).reshape(-1) for _ in range(2)])
        Q = model.embed_feats(F)
        ws = np.maximum.reduceat(Q @ model.protos.T, starts, axis=1)
        j = widx[w]
        for r in range(len(Q)):
            gen.append(ws[r, j])
            other = np.delete(ws[r], j) if len(uniq) > 1 else np.array([0.0])
            imp.append(other.max())
    gen, imp = np.array(gen), np.array(imp)
    q = lambda a, p: float(np.percentile(a, p))
    return {"gen_p05": q(gen, 5), "gen_p10": q(gen, 10), "gen_p50": q(gen, 50),
            "imp_p50": q(imp, 50), "imp_p90": q(imp, 90), "imp_p95": q(imp, 95), "n_words": len(uniq)}


def _tau(cal):
    """Ngưỡng điểm tối thiểu để chấp nhận một từ (độ dung sai trên giao diện dịch ngưỡng này)."""
    lo = cal.get("imp_p50", 0.0)
    hi = cal.get("gen_p10", 0.5)
    return float(max(MIN_SCORE, min(hi, max(lo, 0.5 * (lo + hi)))))


def fit_embedding(report=None, seed=0):
    """Học lại toàn bộ phép nhúng từ dataset_vsl (vài phút, kể cả vài nghìn từ). Trả về EmbedModel đã lưu."""
    rep = report or (lambda msg, frac: None)
    rng = np.random.default_rng(seed)
    rep("Đang đọc dữ liệu...", 0.02)
    protos = []
    for w, fs in _scan_dataset().items():
        for f in fs:
            seg = _load_seg(w, f)
            if seg is not None:
                protos.append((w, f"{w}/{f}", seg))
    n = len(protos)
    n_words = len({p[0] for p in protos})
    if n_words < MIN_CLASSES:
        raise ValueError(f"Cần ít nhất {MIN_CLASSES} từ để học (đang có {n_words}).")
    K = int(np.clip(36000 // n, 12, AUG_K))
    DIM = SEQ_LEN * REPR_DIM
    p = int(min(PCA_MAX, DIM, max(24, n)))
    m = int(max(2, min(EMB_MAX, p - 1, n_words - 1)))

    # --- bước 1: PCA trên mẫu con ---
    rep("Đang học phép chiếu (1/3)...", 0.05)
    n_pick = min(n, 4500)
    per = max(2, min(K, 600 // n_pick + 1))
    take = rng.permutation(n)[:n_pick]
    rows = []
    for i in take:
        seg = protos[i][2]
        rows.append(make_features(seg).reshape(-1))
        for _ in range(per):
            rows.append(make_features(augment_raw(seg, rng)).reshape(-1))
    X = np.array(rows, dtype=np.float32)
    mu = X.mean(axis=0)
    X -= mu
    C = (X.T @ X) / len(X)
    del X
    ev, evec = np.linalg.eigh(C.astype(np.float64))
    p = max(2, min(p, int((ev > ev.max() * 1e-7).sum())))
    P = evec[:, ::-1][:, :p].astype(np.float32)
    del C, evec

    # --- bước 2: gom thống kê nội-từ / liên-từ (tạo biến thể rồi chiếu ngay, không giữ lại) ---
    ZZ = np.zeros((p, p))
    M = np.zeros((n, p))
    cnt = K + 1
    for i, (_w, _key, seg) in enumerate(protos):
        F = [make_features(seg).reshape(-1)] + [make_features(augment_raw(seg, rng)).reshape(-1) for _ in range(K)]
        Z = ((np.array(F, dtype=np.float32) - mu) @ P).astype(np.float64)
        ZZ += Z.T @ Z
        M[i] = Z.mean(axis=0)
        if i % 100 == 0:
            rep(f"Đang học phép chiếu (2/3)... {i}/{n} video", 0.10 + 0.70 * i / n)
    Sw = (ZZ - cnt * (M.T @ M)) / (n * cnt)               # biến thiên do "người khác làm" (tăng cường)
    uniq, starts = _word_starts([w for w, _k, _s in protos])
    ends = list(starts[1:]) + [n]
    Wm = np.stack([M[a:b].mean(axis=0) for a, b in zip(starts, ends)])   # trung bình mỗi từ
    # các video khác nhau của CÙNG một từ (giọng Bắc/Trung/Nam, nhiều người) cũng là biến thiên thật cần bỏ qua
    dv = np.zeros((p, p))
    nv = 0
    for (a, b), mw in zip(zip(starts, ends), Wm):
        if b - a > 1:
            d = M[a:b] - mw
            dv += d.T @ d
            nv += b - a
    if nv:
        Sw = Sw + 0.3 * dv / nv
    Sb = np.cov(Wm.T) if len(Wm) > 2 else np.outer(Wm[0] - Wm[1], Wm[0] - Wm[1])
    Sw = Sw + LDA_REG * np.trace(Sw) / p * np.eye(p)

    # --- bước 3: LDA ---
    rep("Đang học phép chiếu (3/3)...", 0.85)
    L = np.linalg.cholesky(Sw)
    Linv = np.linalg.inv(L)
    ev2, evec2 = np.linalg.eigh(Linv @ Sb @ Linv.T)
    W = (Linv.T @ evec2[:, ::-1][:, :m]).astype(np.float32)
    E = M.astype(np.float32) @ W
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    model = EmbedModel(mu.astype(np.float32), P, W, E, [w for w, _k, _s in protos], [k for _w, k, _s in protos], {})
    rep("Đang hiệu chỉnh ngưỡng nhận diện...", 0.92)
    model.cal = _calibrate(model, protos, rng)
    model.save()
    _EMBED_CACHE["stamp"] = None
    rep(f"Đã học xong {n_words} từ ({n} video).", 1.0)
    return model


def sync_embedding(model):
    """Đồng bộ nguyên mẫu với dataset_vsl mà KHÔNG học lại phép chiếu: thêm video mới, bỏ video đã xóa. (model, đổi?)"""
    scan = _scan_dataset()
    now = {f"{w}/{f}": w for w, fs in scan.items() for f in fs}
    have = set(model.proto_key)
    new_keys = [k for k in now if k not in have]
    gone = have - set(now)
    if not new_keys and not gone:
        return model, False
    rng = np.random.default_rng(1)
    items = [(w, k, e) for w, k, e in zip(model.proto_word, model.proto_key, model.protos) if k in now]
    for k in new_keys:
        w, f = k.split("/", 1)
        seg = _load_seg(w, f)
        if seg is None:
            continue
        F = [make_features(seg).reshape(-1)] + [make_features(augment_raw(seg, rng)).reshape(-1) for _ in range(AUG_K_INC)]
        e = model.embed_feats(np.array(F, dtype=np.float32)).mean(axis=0)
        items.append((w, k, e / (np.linalg.norm(e) + 1e-9)))
    items.sort(key=lambda t: (t[0], t[1]))
    model = EmbedModel(model.mu, model.P, model.W, np.stack([t[2] for t in items]).astype(np.float32),
                       [t[0] for t in items], [t[1] for t in items], model.cal)
    model.save()
    _EMBED_CACHE["stamp"] = None
    return model, True


# ------------------------------ Xếp hạng lại bằng DTW (góc ngón + lật gương) ------------------------------ #
_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
_FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
DTW_BAND = 8
MIRROR_PEN = 1.03
PART_PEN = 1.04          # phạt nhẹ khi khớp bằng nửa đầu / nửa sau (ký hiệu lặp lại nhiều lần)
CAM_BONUS = 0.88         # mẫu quay từ chính camera của bạn (cam_*.npy) được ưu tiên nhẹ
RERANK_K = 40            # số từ ứng viên từ bước nhúng được đem đi so DTW
FUSE_W = 1.5
MIN_MARGIN = 0.04        # độ chênh tối thiểu giữa từ giống nhất và từ giống thứ hai (cosine); thấp hơn = máy phân vân, từ chối
MARGIN_TOL = 0.03        # thanh dung sai hạ mức chênh này xuống (mỗi 1.0 dung sai)
REGION_PREF = None       # 'B' / 'T' / 'N' = người dùng ký theo vùng Bắc / Trung / Nam; None = không ưu tiên
REGION_BONUS = 0.03      # cộng nhẹ vào độ giống của mẫu đúng vùng (mẫu vùng khác không bị loại)
REGION_PATH = os.path.join(BASE_DIR, "vsl_region.txt")
AUTO_VOCAB = True        # luôn tìm trong TOÀN BỘ kho; các từ trong danh sách thường dùng (vsl_active_words.txt) chỉ được ưu tiên nhẹ
PRIOR_COMMON = 0.05      # cộng điểm cho từ thường dùng (không loại từ nào)
PRIOR_TOPIC = 0.03       # cộng điểm cho từ cùng chủ đề với các từ vừa nhận diện trong phiên
RECENT_N = 6
_RECENT = []             # các từ vừa được chấp nhận gần đây
ACCEPT_COS = 0.33        # trần ngưỡng cosine chấp nhận (ngưỡng hiệu chỉnh cao hơn sẽ bị hạ xuống mức này)


def region_of_key(key):
    """'từ/W00032B.npy' -> 'B' (Bắc) / 'T' (Trung) / 'N' (Nam); không có hậu tố vùng -> ''."""
    stem = os.path.splitext(key.split("/", 1)[-1])[0]
    return stem[-1] if len(stem) >= 2 and stem[-1] in "BNT" and stem[-2].isdigit() else ""


def load_region():
    global REGION_PREF
    try:
        with open(REGION_PATH, "r", encoding="utf-8") as f:
            v = f.read().strip().upper()
    except OSError:
        v = ""
    REGION_PREF = v if v in ("B", "T", "N") else None
    return REGION_PREF


def save_region(v):
    global REGION_PREF
    REGION_PREF = v if v in ("B", "T", "N") else None
    try:
        with open(REGION_PATH, "w", encoding="utf-8") as f:
            f.write(REGION_PREF or "")
    except OSError:
        pass


def note_recent(word):
    """App gọi khi một từ được chấp nhận, để máy tự thu hẹp theo chủ đề đang nói."""
    if word:
        _RECENT.append(word)
        del _RECENT[:-RECENT_N]


def clear_recent():
    del _RECENT[:]


def _topic_maps(uniq):
    """Đọc các file words_*.txt (bộ từ theo chủ đề) -> với mỗi từ, danh sách chủ đề của nó. Bỏ từ nằm trong >=4 bộ (từ giao tiếp chung)."""
    import glob
    sets = {}
    for pth in glob.glob(os.path.join(BASE_DIR, "words_*.txt")):
        try:
            with open(pth, "r", encoding="utf-8-sig") as f:
                sets[os.path.basename(pth)] = {safe_label(l).lower() for l in f if l.strip() and not l.strip().startswith("#")}
        except OSError:
            pass
    cnt = {}
    for ws_ in sets.values():
        for w in ws_:
            cnt[w] = cnt.get(w, 0) + 1
    names = sorted(sets)
    wt = []
    for w in uniq:
        wl = w.lower()
        wt.append([i for i, n in enumerate(names) if wl in sets[n]] if cnt.get(wl, 0) < 4 else [])
    return names, wt


def _word_bonus(bundle):
    n = len(bundle["actions"])
    b = np.zeros(n, dtype=np.float64)
    pr = bundle.get("prior")
    if pr is not None and len(pr) == n:
        b += PRIOR_COMMON * pr
    if _RECENT and bundle.get("word_topics"):
        w2i = bundle.get("word_index") or {}
        share = {}
        for w in _RECENT:
            i = w2i.get(w)
            if i is None:
                continue
            for t in bundle["word_topics"][i]:
                share[t] = share.get(t, 0) + 1.0
        if share:
            tot = float(len(_RECENT))
            for i, tl in enumerate(bundle["word_topics"]):
                if tl:
                    b[i] += PRIOR_TOPIC * max(share.get(t, 0.0) for t in tl) / tot
    return b


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def dtw_feat(seg):
    """(T,126) -> (30,182): mỗi tay gồm hướng xương, góc gập ngón, độ duỗi, khép ngón, pháp tuyến lòng bàn tay,
    quỹ đạo và vị trí cổ tay. Không phụ thuộc kích cỡ bàn tay / khoảng cách tay."""
    x = _resample(fill_gaps(trim_rest(np.asarray(seg, dtype=np.float32))), SEQ_LEN).astype(np.float64).reshape(-1, 2, 21, 3)
    T = len(x)
    outs = []
    for h in range(2):
        pts = x[:, h]
        present = np.abs(pts).sum((1, 2)) > 0
        if not present.any():
            outs.append(None)
            continue
        rel = pts - pts[:, :1, :]
        bones = np.stack([rel[:, k] - rel[:, _PARENT[k]] for k in range(1, 21)], axis=1)
        d = [_unit(bones).reshape(T, -1)]
        angs = []
        for f in _FINGERS:
            idx = [0] + f
            b = [_unit(rel[:, idx[i + 1]] - rel[:, idx[i]]) for i in range(4)]
            for i in range(3):
                angs.append(np.arccos(np.clip((b[i] * b[i + 1]).sum(-1), -1, 1)))
        d.append(np.stack(angs, 1) / np.pi * 2)
        size = np.linalg.norm(rel[:, 9], axis=-1, keepdims=True) + 1e-6
        d.append(np.stack([np.linalg.norm(rel[:, t], axis=-1) for t in (4, 8, 12, 16, 20)], 1) / size)
        d.append(np.stack([np.linalg.norm(rel[:, 4] - rel[:, t], axis=-1) for t in (8, 12, 16, 20)], 1) / size)
        d.append(_unit(np.cross(rel[:, 5], rel[:, 17])))
        wrist = pts[:, 0, :2]
        f = np.concatenate(d + [(wrist - wrist[present].mean(0)) * TRAJ_W, wrist * ABS_W], 1)
        f[~present] = 0
        outs.append(f)
    if all(o is None for o in outs):          # nửa đoạn không có tay nào -> toàn số 0 (tránh lỗi làm app dừng)
        return np.zeros((T, 182), np.float32)
    D = max(o.shape[1] for o in outs if o is not None)
    outs = [o if o is not None else np.zeros((T, D)) for o in outs]
    return np.concatenate(outs, 1).astype(np.float32)


def dtw_pairs(A, B, band=DTW_BAND):
    """DTW (dải hẹp) cho nhiều cặp cùng lúc bằng numpy. A:(n,d) hoặc (P,n,d), B:(P,m,d). Trả về (P,)."""
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    if A.ndim == 2:
        A = A[None]
    n, m = A.shape[1], B.shape[1]
    P = len(B)
    ab = np.einsum('xid,pjd->pij', A, B) if len(A) == 1 else np.einsum('pid,pjd->pij', A, B)
    a2 = (A * A).sum(-1)
    b2 = (B * B).sum(-1)
    cost = np.sqrt(np.maximum(a2[:, :, None] + b2[:, None, :] - 2.0 * ab, 0.0))
    w = max(band, abs(n - m))
    D = np.full((P, n + 1, m + 1), np.inf)
    D[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            D[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return D[:, n, m] / (n + m)


def _query_variants(seg):
    """Đoạn đầy đủ + nửa đầu + nửa sau (khi làm lặp nhiều lần hoặc dư động tác) -> [(feat, feat_gương, phạt)]."""
    seg = np.asarray(seg, dtype=np.float32)
    parts = [(seg, 1.0)]
    T = len(seg)
    if T >= 24:
        c = int(T * 0.6)
        parts += [(seg[:c], PART_PEN), (seg[T - c:], PART_PEN)]
    return [(dtw_feat(p), dtw_feat(mirror_raw(p)), pen) for p, pen in parts]


def _load_dtw_cache():
    """Đọc vsl_dtw.npz -> {khóa 'từ/tệp': đặc trưng DTW}. Không có/lỗi -> {}."""
    try:
        with np.load(DTW_CACHE_PATH, allow_pickle=False) as d:
            if int(d["version"]) != REPR_VERSION:
                return {}
            D, keys = d["D"], [str(k) for k in d["keys"]]      # đọc mảng 1 lần (np.load là tải lười: gọi d["D"] nhiều lần sẽ giải nén lại mỗi lần)
            return {k: D[i] for i, k in enumerate(keys)}
    except Exception:
        return {}


def build_dtw_cache():
    """Chạy trên máy có dataset_vsl: tính sẵn đặc trưng DTW của mọi nguyên mẫu và lưu vsl_dtw.npz (dùng cho bản chạy trên máy chủ)."""
    model = _load_embed()
    if model is None:
        return 0
    keys, feats = [], []
    for k in model.proto_key:
        w, f = k.split("/", 1)
        seg = _load_seg(w, f)
        if seg is None:
            continue
        keys.append(k)
        feats.append(dtw_feat(seg))
    if not feats:
        return 0
    tmp = DTW_CACHE_PATH[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, keys=np.array(keys, dtype=str), D=np.stack(feats).astype(np.float32), version=np.array(REPR_VERSION))
    os.replace(tmp, DTW_CACHE_PATH)
    return len(keys)


def load_templates_cached():
    """Bộ nhận diện cho các từ ĐANG DÙNG. Lần đầu (chưa có mô hình) sẽ học từ dataset_vsl; sau đó chỉ đồng bộ nhanh.
    Chế độ ĐỘC LẬP (máy chủ): không có dataset_vsl nhưng có vsl_embed.npz -> nạp thẳng từ vsl_embed.npz (+ vsl_dtw.npz nếu có),
    không quét/đồng bộ thư mục dữ liệu (tránh xóa sạch nguyên mẫu và lỗi np.stack rỗng)."""
    all_words = list_actions()
    model = _load_embed()
    standalone = (not all_words) and model is not None
    if standalone:
        all_words = sorted(set(model.proto_word))
    elif model is None:
        if len(all_words) >= MIN_CLASSES:
            model = fit_embedding()
    else:
        model, _ = sync_embedding(model)
    act_words = load_active_words()
    if (AUTO_VOCAB and act_words is not None) or act_words is None:
        active = set(all_words)
    else:
        active = {a for a in all_words if a.lower() in act_words}
    if model is None or not active:
        return {"actions": [], "templates": {}, "thresholds": {}, "embed": None}
    keep = [i for i, w in enumerate(model.proto_word) if w in active]
    words = [model.proto_word[i] for i in keep]
    uniq, starts = _word_starts(words)
    feats, ok_idx = [], []
    dtw_cache = None
    for n_i, i in enumerate(keep):
        w, f = model.proto_key[i].split("/", 1)
        seg = _load_seg(w, f)
        if seg is not None:
            feats.append(dtw_feat(seg))
            ok_idx.append(n_i)
            continue
        if standalone:
            if dtw_cache is None:
                dtw_cache = _load_dtw_cache()
            cf = dtw_cache.get(model.proto_key[i])
            if cf is not None:
                feats.append(cf)
                ok_idx.append(n_i)
    D = np.stack(feats).astype(np.float32) if feats else np.zeros((0, SEQ_LEN, 1), np.float32)
    ok = np.zeros(len(keep), dtype=bool)
    ok[ok_idx] = True
    is_cam = np.array([model.proto_key[i].split("/", 1)[1].startswith("cam_") for i in keep])
    region = np.array([region_of_key(model.proto_key[i]) for i in keep])
    prior = None
    word_topics = None
    if AUTO_VOCAB and act_words is not None:
        prior = np.array([1.0 if w.lower() in act_words else 0.0 for w in uniq])
    _names, word_topics = _topic_maps(uniq)
    if not any(word_topics):
        word_topics = None
    return {"prior": prior, "word_topics": word_topics, "word_index": {w: i for i, w in enumerate(uniq)},
            "region": region, "actions": uniq, "templates": {}, "thresholds": {}, "embed": model,
            "E": model.protos[keep], "starts": starts, "tau": _tau(model.cal),
            "D": D, "dtw_of": np.cumsum(ok) - 1, "dtw_ok": ok, "is_cam": is_cam}


def classify_segment(model, model_actions, bundle, seg, tolerance=1.0, min_conf=0.6):
    """Phân loại 1 đoạn ký hiệu thô (T,126). Trả về (từ hoặc None, info).
    Bước 1: nhúng LDA chọn ~40 từ gần nhất. Bước 2: DTW (góc ngón, lật gương, nửa đầu/nửa sau) xếp hạng lại.
    tolerance: hệ số dung sai (lớn = dễ chấp nhận hơn). model/model_actions/min_conf chỉ để giữ tương thích."""
    actions = bundle["actions"]
    info = {"candidate": None, "ratio": None, "conf": None, "reason": "", "top": [], "suggest": []}
    emb = bundle.get("embed")
    if not actions or emb is None:
        info["reason"] = "chưa dạy từ nào" if not actions else "chưa học xong mô hình"
        return None, info
    q = emb.embed(seg)
    starts = bundle["starts"]
    ends = np.append(starts[1:], len(bundle["E"]))
    cs = bundle["E"] @ q
    ws_raw = np.maximum.reduceat(cs, starts)
    reg = bundle.get("region")
    pref = REGION_PREF if (reg is not None and len(reg) == len(cs)) else None
    if pref:
        cs = cs + REGION_BONUS * (reg == pref)          # chỉ để xếp hạng; ngưỡng chấp nhận vẫn dùng độ giống thật
    ws = np.maximum.reduceat(cs, starts) + _word_bonus(bundle)
    K = min(len(actions), RERANK_K)
    cand = np.argsort(-ws)[:K]
    dist = np.full(K, np.inf)
    D = bundle["D"]
    if len(D):
        pidx, owner = [], []
        for ci, wi in enumerate(cand):
            for pi in range(starts[wi], ends[wi]):
                if bundle["dtw_ok"][pi]:
                    pidx.append(bundle["dtw_of"][pi])
                    owner.append((ci, pi))
        if pidx:
            T = D[np.array(pidx)]
            best = np.full(len(pidx), np.inf)
            for f, fm, pen in _query_variants(seg):
                d = np.minimum(dtw_pairs(f, T), MIRROR_PEN * dtw_pairs(fm, T)) * pen
                best = np.minimum(best, d)
            for (ci, pi), d in zip(owner, best):
                if bundle["is_cam"][pi]:
                    d *= CAM_BONUS
                if pref and reg[pi] == pref:
                    d *= 1.0 - REGION_BONUS
                dist[ci] = min(dist[ci], d)
    if np.isfinite(dist).any():
        dz = np.where(np.isfinite(dist), dist, np.nanmax(dist[np.isfinite(dist)]) + 1.0)
        cz = ws[cand]
        fused = (cz - cz.mean()) / (cz.std() + 1e-6) - FUSE_W * (dz - dz.mean()) / (dz.std() + 1e-6)
    else:
        fused = (ws[cand] - ws[cand].mean()) / (ws[cand].std() + 1e-6)
    order = np.argsort(-fused)
    b = int(order[0])
    word = actions[int(cand[b])]
    cos_b, d_b = float(ws_raw[cand[b]]), float(dist[b])
    info["candidate"], info["ratio"] = word, cos_b
    info["top"] = [(actions[int(cand[i])], float(ws[cand[i]])) for i in order[:3]]
    info["suggest"] = [actions[int(cand[i])] for i in order[:3]]
    info["dtw"] = d_b
    # DTW chỉ dùng để xếp hạng; chấp nhận/từ chối dựa vào cosine (thử nghiệm cho thấy tách "từ lạ" tốt hơn DTW).
    # Lúc đi thi người dùng chỉ làm từ trong danh sách nên chọn ngưỡng dễ chịu (giữ ~93% mẫu đúng).
    tau = max(MIN_SCORE * 0.9, min(bundle["tau"], ACCEPT_COS) - 0.12 * (tolerance - 1.0))
    o = np.sort(ws)[::-1]
    margin = float(o[0] - o[1]) if len(o) > 1 else 1.0
    info["margin"] = margin
    need = max(0.0, MIN_MARGIN - MARGIN_TOL * (tolerance - 1.0))
    if cos_b >= tau and margin >= need:
        return word, info
    info["reason"] = "không giống từ nào đã dạy" if cos_b < tau else "quá giống nhiều từ, máy chưa chắc"
    return None, info


# ------------------------------ Tách từng ký hiệu từ luồng camera ------------------------------ #
class SegmentDetector:
    """Gom khung hình thành 1 đoạn ký hiệu: bắt đầu khi tay cử động, kết thúc khi tay hạ xuống,
    biến mất hoặc đứng yên. push() trả về đoạn (T,126) khi xong, ngược lại None."""
    END_GAP = 8          # số khung liên tiếp không thấy tay -> kết thúc
    STILL_FRAMES = 20    # số khung tay đứng yên -> kết thúc
    MOVE_START = 0.017   # mức chuyển động tối thiểu để bắt đầu ghi (đơn vị: độ rộng vai / khung hình)
    STILL_EPS = 0.009    # dưới mức này coi là đứng yên
    MAX_LEN = 150        # quá dài thì ép kết thúc
    COOLDOWN = 4         # số khung nghỉ sau mỗi ký hiệu
    STATIC_MEAN = 0.010  # chuyển động trung bình thấp hơn mức này = tay giơ đứng yên, bỏ

    def __init__(self):
        self.reset()

    def reset(self):
        self.frames, self.recording = [], False
        self.gap, self.prev, self.cool = 0, None, 0
        self.motion = deque(maxlen=self.STILL_FRAMES)
        self.all_motion = []
        self.armed = True        # sẵn sàng bắt đầu ghi ngay khi thấy tay (tay vừa vào khung / vừa cử động trở lại)

    def _finish(self, trim):
        seg = np.array(self.frames[:len(self.frames) - trim] if trim else self.frames, dtype=np.float32)
        mean_motion = float(np.mean(self.all_motion)) if self.all_motion else 0.0
        self.frames, self.recording, self.gap, self.prev, self.cool = [], False, 0, None, self.COOLDOWN
        self.motion.clear()
        self.all_motion = []
        self.armed = False       # tay còn đang giơ sau khi xong: chờ tay rời khỏi khung / cử động lại mới ghi tiếp
        hand_frames = int((np.abs(seg).sum(axis=1) > 0).sum()) if len(seg) else 0
        if hand_frames < MIN_HAND_FRAMES or mean_motion < self.STATIC_MEAN:
            return None          # tay chỉ giơ lên đứng yên, không phải ký hiệu
        return seg

    def push(self, feat, present):
        motion = 0.0
        if present and self.prev is not None:
            both = (self.prev != 0) & (feat != 0)
            if both.any():
                motion = float(np.abs(feat[both] - self.prev[both]).mean())
        self.prev = feat if present else None

        if not present:
            self.armed = True
        elif motion > self.MOVE_START:
            self.armed = True

        if self.cool > 0:
            self.cool -= 1
            return None
        if not self.recording:
            if present and self.armed:      # thấy tay là ghi (hiện REC) ngay, không cần chờ tay cử động
                self.recording, self.frames, self.gap = True, [feat], 0
                self.motion.clear()
                self.all_motion = []
            return None

        self.frames.append(feat)
        self.gap = 0 if present else self.gap + 1
        self.motion.append(motion if present else 1.0)
        if present:
            self.all_motion.append(motion)
        if self.gap >= self.END_GAP:
            return self._finish(self.gap)
        if (len(self.motion) == self.STILL_FRAMES and np.mean(self.motion) < self.STILL_EPS
                and len(self.frames) > self.STILL_FRAMES + MIN_HAND_FRAMES):
            return self._finish(self.STILL_FRAMES - 2)
        if len(self.frames) >= self.MAX_LEN:
            return self._finish(0)
        return None




# ------------------------------ Hàm chính (giữ tên cũ để giao diện không phải đổi) ------------------------------ #
def load_model_and_labels():
    """Bản cũ trả về LSTM. Bản mới không dùng LSTM nữa: luôn (None, [])."""
    return None, []


def delete_word(label):
    """Xóa sạch thư mục dữ liệu .npy của một từ."""
    if label not in list_actions():
        raise ValueError(f"Không tìm thấy từ '{label}'.")
    shutil.rmtree(os.path.join(DATA_DIR, label))


def remove_model_files():
    """Xóa mô hình nhúng (và file của bản LSTM cũ nếu còn)."""
    for path in [EMBED_PATH] + [os.path.join(BASE_DIR, f) for f in OLD_FILES]:
        if os.path.exists(path):
            os.remove(path)
    _EMBED_CACHE["stamp"] = None


def _fit_and_save(actions, report):
    """Học lại toàn bộ phép nhúng (tham số actions chỉ để tương thích; luôn dùng toàn bộ dataset_vsl)."""
    fit_embedding(report)


def train_new_sign(video_path, label_name, progress_cb=None):
    """
    Học từ mới từ 1 video (gọi nhiều lần cùng 1 từ để thêm video). Thêm video không cần học lại phép nhúng.
    Trả về dict: {"trained": bool, "message": str, "actions": list}
    """
    def report(msg, frac):
        if progress_cb:
            progress_cb(msg, frac)

    label = safe_label(label_name)
    if not label:
        raise ValueError("Tên từ không hợp lệ.")
    if not _train_lock.acquire(blocking=False):
        raise RuntimeError("AI đang học một từ khác, vui lòng đợi.")
    try:
        report("Đang trích xuất khớp ngón tay từ video...", 0.05)
        feats, present, no_shoulder = extract_video_features(video_path)
        seg = extract_active_segment(feats, present)
        if seg is None:
            if no_shoulder >= MIN_HAND_FRAMES:
                raise ValueError("Thấy tay nhưng không thấy hai vai. Hãy quay nửa thân trên: có đủ hai vai và hai tay trong khung hình.")
            raise ValueError("Không phát hiện đủ bàn tay trong video. Hãy quay rõ tay, đủ sáng.")

        os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)
        np.save(os.path.join(DATA_DIR, label, f"{int(time.time() * 1000)}.npy"), seg)
        add_active_word(label)
        actions = active_actions()
        if len(list_actions()) < MIN_CLASSES:
            msg = f"Đã lưu '{label}'. Hãy dạy thêm ít nhất {MIN_CLASSES - len(list_actions())} từ khác để bắt đầu nhận diện."
            report(msg, 1.0)
            return {"trained": False, "message": msg, "actions": actions}
        model = _load_embed()
        if model is None:
            fit_embedding(report)
        else:
            report("Đang thêm nguyên mẫu mới...", 0.6)
            sync_embedding(model)
        msg = "Hoàn tất! AI đã học xong và sẵn sàng nhận diện."
        report(msg, 1.0)
        return {"trained": True, "message": msg, "actions": actions}
    finally:
        _train_lock.release()


def retrain_from_dataset(progress_cb=None):
    """Cập nhật sau khi xóa/thêm dữ liệu: đồng bộ nguyên mẫu (nhanh). Chưa có mô hình thì học mới."""
    def report(msg, frac):
        if progress_cb:
            progress_cb(msg, frac)

    if not _train_lock.acquire(blocking=False):
        raise RuntimeError("AI đang học một từ khác, vui lòng đợi.")
    try:
        actions = active_actions()
        if len(list_actions()) < MIN_CLASSES:
            remove_model_files()
            msg = f"Còn dưới {MIN_CLASSES} từ nên chưa nhận diện được; hãy dạy thêm từ."
            report(msg, 1.0)
            return {"trained": False, "message": msg, "actions": actions}
        model = _load_embed()
        if model is None:
            fit_embedding(report)
        else:
            report("Đang cập nhật nguyên mẫu...", 0.5)
            sync_embedding(model)
        msg = "Hoàn tất! AI đã học xong và sẵn sàng nhận diện."
        report(msg, 1.0)
        return {"trained": True, "message": msg, "actions": actions}
    finally:
        _train_lock.release()


load_region()

TUNING_PATH = os.path.join(BASE_DIR, "vsl_tuning.json")   # do danh_gia_nguoi_quay.py ghi khi tìm được hệ số tốt hơn


def load_tuning():
    """Đọc vsl_tuning.json (nếu có) để ghi đè các hệ số đã được đo là tốt hơn. File sai/không hợp lệ thì bỏ qua."""
    global CAM_BONUS
    try:
        with open(TUNING_PATH, "r", encoding="utf-8") as f:
            t = json.load(f)
        v = float(t.get("CAM_BONUS", CAM_BONUS))
        if 0.6 <= v <= 1.0:
            CAM_BONUS = v
    except Exception:
        pass


load_tuning()
