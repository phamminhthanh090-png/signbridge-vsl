import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import base64
import html
import json
import queue
import re
import time
import os
import glob as _glob
import tempfile
import threading
import sys
import subprocess
import importlib

HEADLESS = os.environ.get("VSL_HEADLESS") == "1"   # =1: chạy làm máy chủ web (cloud_server.py), không dựng giao diện Streamlit

# Tự kiểm tra thư viện dịch offline. Chỉ cài khi CHƯA có (Streamlit chạy lại file mỗi lần bấm nút,
# nếu gọi pip mỗi lần sẽ rất chậm). Tên gói trên PyPI là "argostranslate" (viết liền).
try:
    if HEADLESS:
        raise ImportError("bỏ qua gói dịch offline trên máy chủ")
    import argostranslate.package
    import argostranslate.translate
    ARGOS_OK = True
except ImportError:
    if HEADLESS:
        pass
    else:
        os.system(f'"{sys.executable}" -m pip install argostranslate')
    importlib.invalidate_caches()
    try:
        import argostranslate.package
        import argostranslate.translate
        ARGOS_OK = True
    except ImportError:
        ARGOS_OK = False

if ARGOS_OK:   # gói dịch lưu NGAY TRONG thư mục app (argos_packages) để chép sang máy khác vẫn chạy offline
    try:
        import argos_setup
        argos_setup.setup()
    except Exception:
        pass

# Cấu trúc phẳng: luôn chạy từ thư mục chứa file này để app.py, model.tflite, train.csv,
# video_paths.py, vsl_model.h5... đều tìm thấy nhau dù bạn gõ lệnh streamlit ở đâu.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# set_page_config phải là lệnh Streamlit ĐẦU TIÊN và chỉ được gọi 1 lần
try:
    from PIL import Image as _PILImage
    _page_icon = _PILImage.open(os.path.join(BASE_DIR, "logo_icon.png")).convert("RGBA").resize((64, 64))   # logo SignBridge trên thẻ trình duyệt
except Exception:
    _page_icon = "🤟"
st.set_page_config(page_title="SignBridge", page_icon=_page_icon, layout="wide")

import train_engine as te
try:
    import live_bridge
except Exception:      # thiếu live_bridge.py: chỉ mất khung trò chuyện trực tiếp
    live_bridge = None
try:
    import avatar_ky_hieu as avatar
except Exception:      # thiếu file avatar_ky_hieu.py thì chỉ mất tính năng xem ký hiệu, app vẫn chạy
    avatar = None


def extract_keypoints(results, aspect=1.0, ref=None):
    """Trích 126 tọa độ phẳng của hai bàn tay, TƯƠNG ĐỐI so với trung tâm hai vai:
        shoulder_center = (left_shoulder[11] + right_shoulder[12]) / 2
    rồi trừ shoulder_center khỏi mọi đốt ngón tay (và chia cho độ rộng vai) -> triệt tiêu khác biệt ngồi/đứng.
    Hàm này gọi thẳng train_engine.extract_keypoints, nên lúc huấn luyện từ video và lúc chạy camera
    dùng CHUNG MỘT thuật toán duy nhất, không thể lệch nhau. ref = te.ShoulderRef() của luồng camera."""
    return te.extract_keypoints(results, aspect, ref)


# ============================================================================================ #
#                                      GIAO DIỆN (CSS)                                         #
# ============================================================================================ #
# Lưu ý: st.container(key="...") tạo lớp CSS `.st-key-...` (cần Streamlit >= 1.39).
GLASS_CSS = """
<style>
:root{
  --bg1:#070b1a; --bg2:#0d1430; --ink:#e8eefc; --muted:#9fb0cf;
  --cyan:#22d3ee; --green:#34f5a1; --violet:#8b5cf6; --blue:#3b82f6; --red:#ff5d73;
  --glass:rgba(255,255,255,.06); --glass-line:rgba(255,255,255,.14);
}
.stApp{
  background:
    radial-gradient(900px 500px at 8% -5%, rgba(139,92,246,.28), transparent 60%),
    radial-gradient(800px 500px at 100% 10%, rgba(34,211,238,.20), transparent 60%),
    linear-gradient(160deg,var(--bg1),var(--bg2));
  color:var(--ink);
  font-family:'Segoe UI',system-ui,-apple-system,Roboto,sans-serif;
}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem; max-width:1200px;}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp label,.stApp li,.stApp span{color:var(--ink);}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted)!important;}
hr{border-color:var(--glass-line)!important;}

/* Sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(13,20,48,.92),rgba(7,11,26,.96));
  border-right:1px solid var(--glass-line);
}

/* Thẻ kính dùng chung */
.glass{
  background:var(--glass); border:1px solid var(--glass-line); border-radius:20px;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
}

/* ---------- Trang giới thiệu ---------- */
.hero{padding:42px 40px; margin-bottom:22px; position:relative; overflow:hidden;}
.hero::after{content:""; position:absolute; right:-80px; top:-80px; width:300px; height:300px; border-radius:50%;
  background:radial-gradient(circle,rgba(52,245,161,.30),transparent 65%);}
.hero h1{font-size:2.3rem; line-height:1.25; margin:0 0 12px; font-weight:800; letter-spacing:-.5px;
  background:linear-gradient(90deg,#fff,var(--cyan) 60%,var(--green)); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent;}
.hero p{max-width:760px; color:var(--muted)!important; font-size:1.05rem; line-height:1.7; margin:0;}
.badge-row{display:flex; gap:10px; flex-wrap:wrap; margin-top:20px;}
.badge{padding:6px 14px; border-radius:999px; font-size:.85rem; font-weight:600;
  border:1px solid rgba(52,245,161,.45); background:rgba(52,245,161,.10); color:var(--green);}
.grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:18px; margin-bottom:22px;}
.card{padding:26px 24px;}
.card .icon{width:56px; height:56px; border-radius:16px; display:flex; align-items:center; justify-content:center;
  font-size:28px; margin-bottom:16px;
  background:linear-gradient(135deg,rgba(34,211,238,.25),rgba(139,92,246,.25));
  border:1px solid rgba(34,211,238,.5); box-shadow:0 0 22px rgba(34,211,238,.45);}
.card.g .icon{border-color:rgba(52,245,161,.55); box-shadow:0 0 22px rgba(52,245,161,.45);
  background:linear-gradient(135deg,rgba(52,245,161,.25),rgba(34,211,238,.2));}
.card.v .icon{border-color:rgba(139,92,246,.6); box-shadow:0 0 22px rgba(139,92,246,.5);
  background:linear-gradient(135deg,rgba(139,92,246,.3),rgba(59,130,246,.2));}
.card h3{margin:0 0 8px; font-size:1.15rem; font-weight:700;}
.card p{margin:0; color:var(--muted)!important; line-height:1.6; font-size:.95rem;}
.ready{padding:16px 22px; display:flex; align-items:center; gap:14px;}
.dot{width:12px; height:12px; border-radius:50%; background:var(--green); box-shadow:0 0 12px var(--green);}

/* ---------- Tiêu đề từng khu vực ---------- */
.sec-head{display:flex; align-items:center; gap:14px; margin:6px 0 14px;}
.sec-head .ic{width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  font-size:22px; background:rgba(255,255,255,.08); border:1px solid var(--glass-line);}
.sec-head h3{margin:0; font-size:1.3rem; font-weight:800;}
.sec-head small{color:var(--muted); font-size:.9rem;}
.page-title{font-size:2rem; font-weight:800; margin:0 0 4px; letter-spacing:-.4px;}
.page-sub{color:var(--muted); margin:0 0 10px;}

/* ---------- Khu vực Micro ---------- */
.st-key-mic_box{
  background:linear-gradient(135deg,rgba(30,58,138,.75),rgba(109,40,217,.65));
  border:1px solid rgba(255,255,255,.22); border-radius:24px; padding:24px 26px;
  box-shadow:0 14px 40px rgba(76,29,149,.45), inset 0 1px 0 rgba(255,255,255,.18);
  backdrop-filter:blur(14px);
}
.st-key-mic_box [data-testid="stAudioInput"]{border-radius:14px;}
.stt-box{margin-top:14px; padding:18px 22px; border-radius:16px; background:rgba(5,8,25,.55);
  border:1px solid rgba(255,255,255,.2);}
.stt-box .lab{font-size:.85rem; color:#c7d2fe; margin-bottom:6px;}
.stt-box .txt{font-size:1.9rem; line-height:1.4; font-weight:800; color:#fff; word-break:break-word;}
.stt-box.warn .txt{font-size:1.1rem; font-weight:600; color:#fde68a;}

/* ---------- Expander + ô nhập ---------- */
[data-testid="stExpander"]{
  background:var(--glass); border:1px solid var(--glass-line)!important; border-radius:18px;
  backdrop-filter:blur(12px);
}
[data-testid="stExpander"] summary p{font-weight:700; font-size:1.02rem;}
.stTextInput input,[data-baseweb="select"]>div{
  background:rgba(255,255,255,.07)!important; color:var(--ink)!important;
  border:1px solid var(--glass-line)!important; border-radius:12px!important;
}
.stTextInput input:focus{border-color:var(--cyan)!important; box-shadow:0 0 0 3px rgba(34,211,238,.22)!important;}
[data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.05); border:1.5px dashed rgba(34,211,238,.55); border-radius:14px;
}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink)!important;}
.stButton>button{
  border-radius:12px; border:1px solid rgba(255,255,255,.2); color:#fff; font-weight:700;
  background:linear-gradient(135deg,var(--blue),var(--violet)); transition:transform .15s, box-shadow .15s;
}
.stButton>button:hover{transform:translateY(-1px); box-shadow:0 8px 22px rgba(139,92,246,.45); border-color:#fff; color:#fff;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#be123c,var(--red));}
.stButton>button[kind="primary"]:hover{box-shadow:0 8px 22px rgba(255,93,115,.45);}

/* Chip từ vựng + thanh tiến trình */
.chips-title{font-weight:700; margin:6px 0 10px;}
.chips{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:6px;}
.chip{padding:5px 13px; border-radius:999px; font-size:.9rem; font-weight:600; color:#d7fff0;
  background:rgba(52,245,161,.12); border:1px solid rgba(52,245,161,.45); box-shadow:0 0 10px rgba(52,245,161,.18);}
.chip.empty{color:var(--muted); background:rgba(255,255,255,.05); border-color:var(--glass-line); box-shadow:none;}
.prog-card{padding:16px 20px; margin:10px 0; border-radius:16px; background:rgba(5,8,25,.55);
  border:1px solid rgba(34,211,238,.4);}
.prog-card.ok{border-color:rgba(52,245,161,.5);}
.prog-card.err{border-color:rgba(255,93,115,.6);}
.prog-head{display:flex; justify-content:space-between; gap:12px; font-weight:600; margin-bottom:10px;}
.prog-head b{color:var(--cyan);}
.prog-card.ok .prog-head b{color:var(--green);}
.prog-track{height:12px; border-radius:999px; background:rgba(255,255,255,.1); overflow:hidden;}
.prog-fill{height:100%; border-radius:999px; background:linear-gradient(90deg,var(--cyan),var(--green));
  box-shadow:0 0 16px rgba(34,211,238,.9), 0 0 30px rgba(52,245,161,.5); transition:width .4s ease;}
.prog-card.err .prog-fill{background:var(--red); box-shadow:0 0 16px rgba(255,93,115,.9);}

/* ---------- Khu vực Camera ---------- */
.st-key-ctl_box{padding:14px 18px 6px; border-radius:18px; background:var(--glass);
  border:1px solid var(--glass-line); backdrop-filter:blur(12px); margin-bottom:14px;}
.cam-wrap{display:flex; justify-content:center; margin:6px 0 16px;}
.neon-frame{padding:8px; border-radius:22px; background:rgba(5,8,25,.6); width:100%; max-width:600px;
  border:2px solid var(--green);
  box-shadow:0 0 14px rgba(52,245,161,.65), 0 0 42px rgba(52,245,161,.30), inset 0 0 18px rgba(52,245,161,.18);}
.neon-frame img{display:block; width:100%; border-radius:15px;}
.result-big{margin:6px auto 12px; max-width:900px; text-align:center; padding:26px 28px; border-radius:22px;
  background:linear-gradient(135deg,rgba(16,185,129,.28),rgba(34,211,238,.22));
  border:1.5px solid rgba(52,245,161,.7); box-shadow:0 0 30px rgba(52,245,161,.30);}
.result-big .lab{font-size:.9rem; color:#b8fbe0; margin-bottom:8px;}
.result-big .txt{font-size:3rem; font-weight:900; line-height:1.25; color:#fff; word-break:break-word;
  text-shadow:0 0 22px rgba(52,245,161,.8);}
.result-idle{margin:6px auto 12px; max-width:900px; text-align:center; padding:20px 24px; border-radius:18px;
  background:var(--glass); border:1px dashed var(--glass-line); color:var(--muted); font-size:1.05rem;}

/* ---------- Sửa lỗi nền trắng/chữ vô hình khi trình duyệt dùng giao diện sáng ---------- */
:root{color-scheme:dark;}
[data-testid="stWidgetLabel"] *{color:var(--ink)!important; font-weight:600;}
[data-testid="stToolbar"] *{color:var(--ink)!important;}

/* Ô nhập chữ: nền sáng, chữ ĐEN đậm, viền xanh phát sáng khi bấm vào (thay viền đỏ mặc định) */
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"]{
  background:#f5f8ff!important; border-radius:12px!important;
}
[data-baseweb="input"]{
  border:1.5px solid rgba(34,211,238,.45)!important; box-shadow:0 4px 14px rgba(0,0,0,.18)!important;
  transition:border-color .2s, box-shadow .2s;
}
[data-baseweb="input"]:hover{border-color:var(--cyan)!important;}
[data-baseweb="input"]:focus-within{
  border-color:var(--cyan)!important; box-shadow:0 0 0 3px rgba(34,211,238,.30), 0 0 20px rgba(34,211,238,.35)!important;
}
.stTextInput input,.stTextArea textarea{
  background:transparent!important; border:none!important; box-shadow:none!important; border-radius:12px!important;
  color:#0f172a!important; -webkit-text-fill-color:#0f172a!important; caret-color:var(--violet)!important;
  font-size:1.02rem; font-weight:600; padding:12px 14px!important;
}
.stTextInput input:focus{border:none!important; box-shadow:none!important;}
.stTextInput input::placeholder{color:#64748b!important; -webkit-text-fill-color:#64748b!important; font-weight:400;}

/* Khung chọn (selectbox) + danh sách thả xuống */
[data-baseweb="select"]>div{
  background:#f5f8ff!important; border:1.5px solid rgba(34,211,238,.45)!important; border-radius:12px!important;
  min-height:46px; box-shadow:0 4px 14px rgba(0,0,0,.18)!important; transition:border-color .2s, box-shadow .2s;
}
[data-baseweb="select"]>div:hover{border-color:var(--cyan)!important;}
[data-baseweb="select"]:focus-within>div{
  border-color:var(--cyan)!important; box-shadow:0 0 0 3px rgba(34,211,238,.30), 0 0 20px rgba(34,211,238,.35)!important;
}
[data-baseweb="select"] *{color:#0f172a!important; font-weight:600;}
[data-baseweb="select"] svg{fill:var(--violet)!important;}
[data-baseweb="popover"] ul,[data-baseweb="popover"] [data-baseweb="menu"]{
  background:#f5f8ff!important; border:1.5px solid rgba(34,211,238,.5)!important; border-radius:14px!important;
  box-shadow:0 14px 34px rgba(0,0,0,.45)!important; overflow:hidden; padding:6px!important;
}
[data-baseweb="popover"] li{background:transparent!important; border-radius:10px!important; margin:2px 0;}
[data-baseweb="popover"] li *{color:#0f172a!important; font-weight:600;}
[data-baseweb="popover"] li:hover{background:linear-gradient(90deg,rgba(34,211,238,.25),rgba(139,92,246,.20))!important;}
[data-baseweb="popover"] li[aria-selected="true"]{background:linear-gradient(90deg,rgba(34,211,238,.42),rgba(139,92,246,.35))!important;}

/* Expander: thanh tiêu đề gradient thay cho nền trắng */
[data-testid="stExpander"] details{background:transparent!important; border:none!important;}
[data-testid="stExpander"] summary{
  background:linear-gradient(135deg,rgba(34,211,238,.16),rgba(139,92,246,.22))!important;
  border-radius:18px; padding:14px 18px!important; transition:filter .2s;
}
[data-testid="stExpander"] summary:hover{filter:brightness(1.2);}
[data-testid="stExpander"] summary *{color:#fff!important;}
[data-testid="stExpander"] summary svg{fill:var(--cyan)!important;}
[data-testid="stExpander"] details[open] summary{border-bottom-left-radius:0; border-bottom-right-radius:0;}

/* Ô tải video */
[data-testid="stFileUploaderDropzone"]{
  background:rgba(34,211,238,.06)!important; border:1.5px dashed rgba(34,211,238,.6)!important;
  border-radius:16px!important; padding:18px!important; transition:background .2s, border-color .2s;
}
[data-testid="stFileUploaderDropzone"]:hover{background:rgba(34,211,238,.12)!important; border-color:var(--green)!important;}
[data-testid="stFileUploaderDropzone"] button{
  background:linear-gradient(135deg,var(--blue),var(--violet))!important; border:1px solid rgba(255,255,255,.25)!important;
  border-radius:12px!important; font-weight:700;
}
[data-testid="stFileUploaderDropzone"] button *{color:#fff!important;}
[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span{color:var(--muted)!important;}
[data-testid="stFileUploaderFile"] *{color:var(--ink)!important;}

/* Thanh ghi âm trong hộp Micro */
[data-testid="stAudioInput"]{
  background:rgba(5,8,25,.55)!important; border:1px solid rgba(255,255,255,.25)!important; border-radius:16px!important;
}
[data-testid="stAudioInput"] div{background:transparent!important;}
[data-testid="stAudioInput"] *{color:var(--ink)!important;}
[data-testid="stAudioInput"] button{background:rgba(255,255,255,.12)!important;}
[data-testid="stAudioInput"] svg{fill:var(--cyan)!important;}

/* Thanh trượt, thông báo, spinner */
[data-testid="stSlider"] div[role="slider"]{background:var(--cyan)!important; box-shadow:0 0 12px var(--cyan)!important;}
[data-testid="stSlider"] [data-testid="stTickBarMin"],[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"]{color:var(--ink)!important;}
[data-testid="stAlert"]{background:rgba(255,255,255,.07)!important; border:1px solid var(--glass-line)!important; border-radius:14px!important;}
[data-testid="stAlert"] *{color:var(--ink)!important;}
[data-testid="stSpinner"] *{color:#fff!important;}

/* ====================== HIỆU ỨNG ĐỘNG (rê chuột / nhấp chuột) ====================== */
/* Chỉ dùng transform/opacity để mượt, không làm giật luồng camera. Dùng "backwards" để hiệu ứng vào không đè lên hover. */
@keyframes fadeUp{from{opacity:0; transform:translateY(20px);} to{opacity:1; transform:none;}}
@keyframes popIn{0%{opacity:0; transform:scale(.94);} 60%{transform:scale(1.02);} 100%{opacity:1; transform:scale(1);}}
@keyframes floatY{0%,100%{translate:0 0;} 50%{translate:0 -6px;}}
@keyframes orb{0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(-34px,26px) scale(1.18);}}
@keyframes shimmer{to{background-position:200% center;}}
@keyframes dotPulse{0%{box-shadow:0 0 0 0 rgba(52,245,161,.75);} 100%{box-shadow:0 0 0 14px rgba(52,245,161,0);}}
@keyframes stripes{to{background-position:40px 0;}}
@keyframes neonBreath{0%,100%{opacity:.25;} 50%{opacity:1;}}
@keyframes idlePulse{0%,100%{opacity:.6;} 50%{opacity:1;}}

html{scroll-behavior:smooth;}
::selection{background:rgba(34,211,238,.45); color:#fff;}
::-webkit-scrollbar{width:10px; height:10px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:linear-gradient(var(--cyan),var(--violet)); border-radius:10px;}

/* Tiêu đề trang + trang giới thiệu */
.page-title,.page-sub{animation:fadeUp .5s ease backwards;}
.hero{animation:fadeUp .6s ease backwards; transition:box-shadow .3s ease, border-color .3s ease;}
.hero:hover{border-color:rgba(34,211,238,.5); box-shadow:0 14px 44px rgba(0,0,0,.4), 0 0 34px rgba(34,211,238,.18);}
.hero::after{animation:orb 9s ease-in-out infinite;}
.hero h1{background-image:linear-gradient(90deg,#fff,var(--cyan),var(--green),var(--cyan),#fff);
  background-size:200% auto; animation:shimmer 7s linear infinite;}
.badge{transition:transform .2s ease, box-shadow .2s ease, background .2s ease; cursor:default;}
.badge:hover{transform:translateY(-3px) scale(1.07); background:rgba(52,245,161,.24); box-shadow:0 0 18px rgba(52,245,161,.55);}

.grid .card{animation:fadeUp .6s ease backwards;
  transition:transform .28s ease, box-shadow .28s ease, border-color .28s ease;}
.grid .card:nth-child(1){animation-delay:.05s;} .grid .card:nth-child(2){animation-delay:.12s;}
.grid .card:nth-child(3){animation-delay:.19s;} .grid .card:nth-child(4){animation-delay:.26s;}
.grid .card:nth-child(5){animation-delay:.33s;} .grid .card:nth-child(6){animation-delay:.40s;}
.card:hover{transform:translateY(-8px); border-color:rgba(34,211,238,.65);
  box-shadow:0 18px 42px rgba(0,0,0,.45), 0 0 30px rgba(34,211,238,.30);}
.card.g:hover{border-color:rgba(52,245,161,.65); box-shadow:0 18px 42px rgba(0,0,0,.45), 0 0 30px rgba(52,245,161,.30);}
.card.v:hover{border-color:rgba(139,92,246,.7); box-shadow:0 18px 42px rgba(0,0,0,.45), 0 0 30px rgba(139,92,246,.35);}
.card .icon{animation:floatY 4s ease-in-out infinite; transition:transform .3s ease;}
.card:hover .icon{transform:scale(1.18) rotate(-7deg);}
.ready{transition:border-color .3s ease, box-shadow .3s ease;}
.ready:hover{border-color:rgba(52,245,161,.6); box-shadow:0 0 26px rgba(52,245,161,.22);}
.dot{animation:dotPulse 1.6s ease-out infinite;}

/* Tiêu đề từng khu vực */
.sec-head .ic{transition:transform .3s ease, box-shadow .3s ease, border-color .3s ease;}
.sec-head:hover .ic{transform:rotate(-8deg) scale(1.14); border-color:var(--cyan); box-shadow:0 0 20px rgba(34,211,238,.55);}

/* Nút bấm: nổi lên khi rê, vệt sáng quét ngang, lún xuống khi nhấp */
.stButton>button{position:relative; overflow:hidden;
  transition:transform .18s ease, box-shadow .25s ease, filter .2s ease, border-color .2s ease;}
.stButton>button::after{content:""; position:absolute; top:0; left:-120%; width:55%; height:100%;
  background:linear-gradient(120deg,transparent,rgba(255,255,255,.38),transparent); transform:skewX(-20deg);
  transition:left .65s ease; pointer-events:none;}
.stButton>button:hover{transform:translateY(-3px);}
.stButton>button:hover::after{left:145%;}
.stButton>button:active{transform:translateY(0) scale(.95); filter:brightness(.88);}
[data-testid="stFileUploaderDropzone"] button{transition:transform .18s ease, box-shadow .25s ease, filter .2s ease;}
[data-testid="stFileUploaderDropzone"] button:hover{transform:translateY(-2px); box-shadow:0 8px 22px rgba(139,92,246,.5);}
[data-testid="stFileUploaderDropzone"] button:active{transform:scale(.94); filter:brightness(.88);}
[data-testid="stFileUploaderDropzone"]:hover{transform:translateY(-2px); box-shadow:0 10px 28px rgba(34,211,238,.22);}

/* Hộp Micro / thanh điều khiển / expander: chỉ đổi sáng viền + bóng (không dịch chuyển để dropdown không bị lệch) */
.st-key-mic_box,.st-key-ctl_box,[data-testid="stExpander"]{transition:box-shadow .3s ease, border-color .3s ease;}
.st-key-mic_box:hover{box-shadow:0 18px 52px rgba(124,58,237,.65), inset 0 1px 0 rgba(255,255,255,.22); border-color:rgba(255,255,255,.4);}
.st-key-ctl_box:hover{border-color:rgba(34,211,238,.5); box-shadow:0 10px 30px rgba(34,211,238,.16);}
[data-testid="stExpander"]:hover{border-color:rgba(34,211,238,.5)!important; box-shadow:0 10px 30px rgba(34,211,238,.16);}
[data-testid="stAudioInput"]{transition:border-color .25s ease, box-shadow .25s ease;}
[data-testid="stAudioInput"]:hover{border-color:var(--cyan)!important; box-shadow:0 0 18px rgba(34,211,238,.35);}
[data-baseweb="popover"] li{transition:background .15s ease, transform .15s ease;}
[data-baseweb="popover"] li:hover{transform:translateX(5px);}

/* Kết quả nói / dịch */
.stt-box{animation:popIn .4s ease backwards; transition:box-shadow .3s ease, border-color .3s ease;}
.stt-box:hover{border-color:rgba(255,255,255,.5); box-shadow:0 0 24px rgba(255,255,255,.14);}
.result-big{animation:popIn .45s ease backwards; transition:transform .25s ease, box-shadow .25s ease;}
.result-big:hover{transform:scale(1.02); box-shadow:0 0 46px rgba(52,245,161,.50);}
.result-idle{animation:idlePulse 2.4s ease-in-out infinite;}

/* Camera: viền neon "thở" nhẹ bằng lớp phủ (chỉ đổi opacity nên không nặng máy) */
.neon-frame{position:relative;}
.neon-frame::after{content:""; position:absolute; inset:-2px; border-radius:24px; pointer-events:none;
  box-shadow:0 0 32px rgba(52,245,161,.85), 0 0 78px rgba(52,245,161,.45); animation:neonBreath 3s ease-in-out infinite;}
.neon-frame:hover::after{animation:none; opacity:1;}

/* Chip từ vựng + thanh tiến độ */
.chip{animation:popIn .4s ease backwards; transition:transform .2s ease, box-shadow .2s ease, background .2s ease; cursor:default;}
.chip:hover{transform:translateY(-3px) scale(1.09); background:rgba(52,245,161,.24); box-shadow:0 0 18px rgba(52,245,161,.6);}
.chip.empty:hover{transform:none; box-shadow:none;}
.prog-card{animation:popIn .35s ease backwards;}
.prog-fill{position:relative; overflow:hidden;}
.prog-fill::after{content:""; position:absolute; inset:0; background-size:40px 40px;
  background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.30) 0 10px,transparent 10px 20px);
  animation:stripes 1s linear infinite;}
.prog-card.ok .prog-fill::after,.prog-card.err .prog-fill::after{animation:none; opacity:0;}

/* Người dùng bật "giảm chuyển động" trong hệ điều hành thì tắt hết hiệu ứng */
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important; transition:none!important;}
}

@media (max-width:700px){
  .hero{padding:28px 22px;} .hero h1{font-size:1.7rem;} .result-big .txt{font-size:2.1rem;}
  .stt-box .txt{font-size:1.4rem;}
}
</style>
"""


def inject_css():
    st.markdown(GLASS_CSS, unsafe_allow_html=True)


def sec_head(icon, title, sub=""):
    st.markdown(
        f'<div class="sec-head"><div class="ic">{icon}</div><div><h3>{html.escape(title)}</h3>'
        f'<small>{html.escape(sub)}</small></div></div>', unsafe_allow_html=True)


def chips_html(words):
    if not words:
        return '<div class="chips"><span class="chip empty">Chưa có từ nào được dạy</span></div>'
    return '<div class="chips">' + "".join(f'<span class="chip">{html.escape(w)}</span>' for w in words) + '</div>'


def progress_html(kind, msg, prog):
    pct = int(max(0.0, min(1.0, prog)) * 100)
    cls = {"ok": "ok", "err": "err"}.get(kind, "")
    icon = {"ok": "✅", "err": "❌"}.get(kind, "🧠")
    right = "" if kind == "err" else f"<b>{pct}%</b>"
    bar = f'<div class="prog-track"><div class="prog-fill" style="width:{100 if kind == "ok" else pct}%"></div></div>'
    return (f'<div class="prog-card {cls}"><div class="prog-head"><span>{icon} {html.escape(str(msg))}</span>{right}</div>'
            f'{bar}</div>')


# ------------------------------ Loa phát thanh (giữ nguyên) ------------------------------ #
def speak_text(text_to_speak, lang='vi'):
    """Phát âm KHÔNG cần Internet (giọng Windows SAPI5 qua pyttsx3). Chạy ngầm, không làm khựng camera."""
    get_tts().speak(text_to_speak, lang)


# ------------------------------ Micro (giữ nguyên) ------------------------------ #
# Ngôn ngữ người nói: nhãn hiển thị, mã nhận diện giọng nói (SpeechRecognition), mã dịch (Argos)
INPUT_LANGS = [
    {"label": "🇻🇳 Tiếng Việt",          "name": "Tiếng Việt",  "sr": "vi-VN", "argos": "vi"},
    {"label": "🇺🇸 Tiếng Anh (English)", "name": "Tiếng Anh",   "sr": "en-US", "argos": "en"},
    {"label": "🇨🇳 Tiếng Trung (中文)",   "name": "Tiếng Trung", "sr": "zh-CN", "argos": "zh"},
    {"label": "🇰🇷 Tiếng Hàn (한국어)",    "name": "Tiếng Hàn",   "sr": "ko-KR", "argos": "ko"},
    {"label": "🇯🇵 Tiếng Nhật (日本語)",   "name": "Tiếng Nhật",  "sr": "ja-JP", "argos": "ja"},
]
_argos_lock = threading.Lock()


def _argos_pair_ready(src, dst):
    """True nếu máy đã có đường dịch src -> dst (trực tiếp hoặc bắc cầu qua tiếng Anh)."""
    langs = argostranslate.translate.get_installed_languages()
    s = next((l for l in langs if l.code == src), None)
    d = next((l for l in langs if l.code == dst), None)
    if s is None or d is None:
        return False
    try:
        return s.get_translation(d) is not None
    except Exception:
        return False


def ensure_argos_model(src, dst="vi"):
    """Có sẵn gói ngôn ngữ thì dùng luôn (offline). Chưa có thì tải 1 lần duy nhất (cần Internet lần đầu)."""
    if _argos_pair_ready(src, dst):
        return
    with _argos_lock:
        if _argos_pair_ready(src, dst):
            return
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()

        def find(a, b):
            return next((p for p in available if p.from_code == a and p.to_code == b), None)

        direct = find(src, dst)
        if direct:
            needed = [direct]
        else:   # không có gói trực tiếp -> bắc cầu src -> en -> dst
            needed = [] if src == "en" else [find(src, "en")]
            needed.append(find("en", dst))
        if any(p is None for p in needed):
            raise RuntimeError(f"Không tìm thấy gói dịch {src} → {dst} trong danh sách Argos.")
        for pkg in needed:
            argostranslate.package.install_from_path(pkg.download())
        if not _argos_pair_ready(src, dst):
            raise RuntimeError(f"Đã tải gói nhưng chưa dùng được đường dịch {src} → {dst}.")


_net_cache = [0.0, False]


def is_online():
    """Kiểm tra nhanh có Internet không (nhớ kết quả 5 giây để khỏi kiểm tra liên tục)."""
    import socket
    if time.time() - _net_cache[0] < 5:
        return _net_cache[1]
    try:
        socket.create_connection(("translate.googleapis.com", 443), timeout=1.5).close()
        ok = True
    except Exception:
        ok = False
    _net_cache[0], _net_cache[1] = time.time(), ok
    return ok


def online_translate(text, src, dst):
    """Dịch trực tuyến (cần Internet), chính xác hơn gói dịch trên máy."""
    import urllib.parse, urllib.request
    code = {"zh": "zh-CN"}
    url = ("https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&sl=%s&tl=%s&q=%s"
           % (code.get(src, src), code.get(dst, dst), urllib.parse.quote(text)))
    with urllib.request.urlopen(url, timeout=6) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = "".join(seg[0] for seg in data[0] if seg and seg[0])
    if not out.strip():
        raise RuntimeError("bản dịch trực tuyến rỗng")
    return out


GEMINI_KEY_PATH = os.path.join(BASE_DIR, "gemini_key.txt")   # dòng 1: khóa API; dòng 2 (tuỳ chọn): tên mô hình


LOG_DIR = os.path.join(BASE_DIR, "nhat_ky")      # nhật ký nhận diện: chỉ lưu trên máy, dùng để tìm lỗi thật (không gửi đi đâu)
LOG_KEEP = 300                                    # chỉ giữ 300 đoạn gần nhất
_log_lock = threading.Lock()


def log_recognition(segment, word, info):
    """Ghi 1 dòng vào nhat_ky/nhan_dien.csv (từ máy chọn, 3 ứng viên + điểm, lý do bỏ qua...) và lưu đoạn tay thô (.npy)
    để sau này chạy lại offline đúng dữ liệu thật. Lỗi ghi nhật ký không bao giờ làm hỏng app."""
    try:
        import csv
        with _log_lock:
            os.makedirs(os.path.join(LOG_DIR, "doan"), exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            seg = np.asarray(segment, dtype=np.float32)
            np.save(os.path.join(LOG_DIR, "doan", ts + ".npy"), seg)
            top = info.get("top") or []
            path = os.path.join(LOG_DIR, "nhan_dien.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow(["thoi_gian", "tu_nhan", "ly_do_bo_qua", "ung_vien_1", "diem_1", "ung_vien_2", "diem_2",
                                 "ung_vien_3", "diem_3", "do_khop", "cach_biet", "so_khung"])
                row = [ts, word or "", info.get("reason", "") if not word else ""]
                for i in range(3):
                    row += [top[i][0], f"{top[i][1]:.3f}"] if i < len(top) else ["", ""]
                row += [f"{info['ratio']:.3f}" if info.get("ratio") is not None else "",
                        f"{info['margin']:.3f}" if info.get("margin") is not None else "", len(seg)]
                wr.writerow(row)
            files = sorted(_glob.glob(os.path.join(LOG_DIR, "doan", "*.npy")))
            for old in files[:-LOG_KEEP]:
                os.remove(old)
    except Exception:
        pass


_CONV = []          # vài câu AI vừa ghép gần đây (làm ngữ cảnh cho câu sau)
_CONV_MAX = 3
_GEM = {"fails": 0, "skip_until": 0.0, "last_ok": None, "last_err": "", "cache": {}, "lock": threading.Lock()}
GEMINI_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]   # thử lần lượt nếu mô hình đầu lỗi/hết lượt
GEMINI_BUDGET = 14.0     # tổng số giây tối đa cho 1 lần gọi AI (quá thì bỏ, dùng luật ghép để câu không bị chậm)


def _gemini_key():
    """(khóa, mô hình ưu tiên) từ file gemini_key.txt hoặc biến môi trường GEMINI_API_KEY; không có -> (None, None)."""
    try:
        with open(GEMINI_KEY_PATH, "r", encoding="utf-8-sig") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if lines and len(lines[0]) > 15 and " " not in lines[0]:
            return lines[0], (lines[1] if len(lines) > 1 else None)
    except Exception:
        pass
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    return (k, None) if k else (None, None)


_DISCOVERED = {"models": None}


def _discover_models(key):
    """Hỏi Google xem khóa này đang dùng được mô hình nào (tránh lỗi 404 do mô hình cũ bị gỡ). Ưu tiên loại flash."""
    if _DISCOVERED["models"] is not None:
        return _DISCOVERED["models"]
    import urllib.request
    out = []
    try:
        req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                     headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=5.0) as r:
            data = json.loads(r.read().decode("utf-8"))
        names = [m_["name"].split("/", 1)[-1] for m_ in data.get("models", [])
                 if "generateContent" in m_.get("supportedGenerationMethods", [])]
        bad = ("image", "tts", "audio", "embed", "vision", "live", "robotics", "computer", "thinking", "preview-0", "exp")
        names = [n for n in names if n.startswith("gemini") and "flash" in n and not any(b in n for b in bad)]
        names.sort(key=lambda n: ("lite" in n, "latest" not in n, n))   # "latest" + bản thường trước, lite sau
        out = names[:5]
    except Exception:
        out = []
    _DISCOVERED["models"] = out
    return out


def _gemini_call(key, models, prompt, deadline, _retry=True):
    """Gọi Gemini, tự thử lại/đổi mô hình khi lỗi tạm thời (429/500/503/hết giờ). Trả về văn bản JSON hoặc ném lỗi."""
    import urllib.request
    import urllib.error
    last = "chưa gọi"
    saw404 = False
    if _DISCOVERED["models"]:
        models = list(_DISCOVERED["models"]) + [m for m in models if m not in _DISCOVERED["models"]]
    for model in models:
        for attempt in range(2):
            left = deadline - time.time()
            if left < 1.5:
                raise RuntimeError("quá thời gian chờ AI (" + last + ")")
            gen = {"temperature": 0, "maxOutputTokens": 600, "responseMimeType": "application/json"}
            if "2.5" in model:
                gen["thinkingConfig"] = {"thinkingBudget": 0}   # không "nghĩ" dài: tránh hết token trước khi ra câu trả lời
            body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}).encode("utf-8")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=min(14.0, left)) as r:
                    data = json.loads(r.read().decode("utf-8"))
                cand = data["candidates"][0]
                txt = "".join(p_.get("text", "") for p_ in cand.get("content", {}).get("parts", [])).strip()
                if not txt:
                    last = f"{model}: AI trả về rỗng ({cand.get('finishReason', '?')})"
                    break                                   # đổi sang mô hình khác
                return txt
            except urllib.error.HTTPError as e:
                code = e.code
                if code in (400, 401, 403):                 # sai khóa / mô hình không hỗ trợ -> thử mô hình khác, khóa sai thì dừng hẳn
                    last = f"{model}: HTTP {code} (khóa sai hoặc chưa bật?)"
                    if code in (401, 403):
                        raise RuntimeError(last)
                    break
                last = f"{model}: HTTP {code}" + (" (hết lượt miễn phí/phút)" if code == 429 else "")
                if code == 404:
                    saw404 = True
                    break
                time.sleep(min(0.8, max(0.0, deadline - time.time() - 1.5)))
            except Exception as e:
                last = f"{model}: {type(e).__name__} (AI trả lời chậm hoặc mạng yếu)"
                time.sleep(min(0.5, max(0.0, deadline - time.time() - 1.5)))
    if saw404 and _retry and _DISCOVERED["models"] is None:
        found = _discover_models(key)
        if found:
            return _gemini_call(key, found, prompt, deadline, _retry=False)
        last += " - không lấy được danh sách mô hình, kiểm tra mạng/khóa"
    raise RuntimeError(last)


def _extract_json(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        return json.loads(m.group(0)) if m else None


def ai_status():
    """Chữ trạng thái AI để hiện cho người dùng."""
    key, _ = _gemini_key()
    if not key:
        return "⚪ Chưa bật AI (thiếu file gemini_key.txt) - đang dùng luật ghép câu offline."
    if _GEM["last_err"] and (_GEM["last_ok"] is None or _GEM["last_ok"] < _GEM.get("last_fail_t", 0)):
        return "🟠 AI lần gần nhất bị lỗi: " + _GEM["last_err"] + " - tạm dùng luật ghép câu offline."
    if _GEM["last_ok"]:
        return "🟢 AI Gemini hoạt động (lần gần nhất thành công)."
    return "🟡 Đã có khóa AI, chưa dùng lần nào. Bấm “Kiểm tra AI” để thử."


def ai_selftest():
    """Gọi thử Gemini bằng một câu rất ngắn. Trả về (ok, thông báo)."""
    key, pref = _gemini_key()
    if not key:
        return False, "Chưa có khóa. Tạo file gemini_key.txt, dòng đầu là khóa lấy từ aistudio.google.com/apikey."
    models = ([pref] if pref else []) + [m for m in GEMINI_MODELS if m != pref]
    t0 = time.time()
    try:
        txt = _gemini_call(key, models, 'Trả về đúng JSON: {"ok": true}', t0 + 40.0)
        obj = _extract_json(txt)
        if isinstance(obj, dict):
            with _GEM["lock"]:
                _GEM.update(fails=0, skip_until=0.0, last_ok=time.time(), last_err="")
            return True, f"AI hoạt động bình thường (trả lời sau {time.time() - t0:.1f} giây)."
        return False, "AI trả lời nhưng sai định dạng: " + txt[:80]
    except Exception as e:
        with _GEM["lock"]:
            _GEM.update(last_err=str(e), last_fail_t=time.time())
        return False, str(e)


def ai_compose(words, alts=None):
    """CÓ Internet + có khóa Gemini (file gemini_key.txt): AI sắp xếp các từ rời thành 1 câu tiếng Việt tự nhiên,
    lịch sự, chỉ xưng "tôi" và "bạn". alts[i] = các từ máy cũng cho là có thể ở vị trí i: AI chỉ đổi sang phương án đó
    khi từ chính làm câu vô nghĩa. Trả về (câu, danh sách từ đã chọn) hoặc None (không có khóa/lỗi/quá chậm -> dùng luật ghép).
    Có bộ ngắt: lỗi liên tiếp thì tạm nghỉ 30 giây để câu không bị chậm; kết quả giống nhau được nhớ lại."""
    key, pref = _gemini_key()
    if not key or time.time() < _GEM["skip_until"]:
        return None
    alts = [list(a or []) for a in (alts or [])]
    alts += [[] for _ in range(len(words) - len(alts))]
    ck = (tuple(words), tuple(tuple(a[:2]) for a in alts), tuple(_CONV[-_CONV_MAX:]))
    if ck in _GEM["cache"]:
        return _GEM["cache"][ck]
    slots = []
    for i, w in enumerate(words):
        extra = [x for x in alts[i] if x and x != w][:2]
        slots.append(f"{i + 1}. {w}" + (f"  (phương án khác nếu từ này sai: {' / '.join(extra)})" if extra else ""))
    ctx = ""
    if _CONV:
        ctx = ("Các câu người ký đã nói ngay trước đó trong cùng cuộc trò chuyện (chỉ để hiểu chủ đề, không lặp lại): "
               + " | ".join(_CONV[-_CONV_MAX:]) + "\n")
    prompt = ("Bạn là bộ ghép câu cho Ngôn ngữ Ký hiệu Việt Nam. " + ctx + "Máy nhận diện đưa ra các từ rời theo thứ tự người ký:\n"
              + "\n".join(slots) + "\n"
              "Nhận diện đôi khi nhầm sang từ gần giống. Mặc định GIỮ NGUYÊN từ chính. Chỉ đổi sang phương án khác của đúng vị trí đó "
              "khi từ chính làm câu vô nghĩa hoặc rất vô lý, còn phương án khác làm câu hợp lý rõ ràng. Không tự thêm từ mới.\n"
              "Viết đúng MỘT câu tiếng Việt ngắn, đúng ngữ pháp, tự nhiên, lịch sự. Chỉ xưng \"tôi\" (người ký) và \"bạn\" "
              "(người đối diện), không dùng anh/chị/em/con/cháu/mình; không thêm ý hay thông tin ngoài các từ đã cho.\n"
              "Trả về JSON đúng dạng: {\"chosen\": [từ ở vị trí 1, từ ở vị trí 2, ...], \"sentence\": \"câu\"}")
    models = ([pref] if pref else []) + [m for m in GEMINI_MODELS if m != pref]
    try:
        obj = _extract_json(_gemini_call(key, models, prompt, time.time() + GEMINI_BUDGET))
        if not isinstance(obj, dict):
            raise RuntimeError("AI trả lời sai định dạng")
        sentence = str(obj.get("sentence", "")).strip().strip('"').strip()
        chosen = obj.get("chosen")
        ok = bool(sentence) and isinstance(chosen, list) and len(chosen) == len(words)
        if ok:
            for i, c in enumerate(chosen):
                if _norm(c) not in {_norm(words[i])} | {_norm(x) for x in alts[i]}:
                    ok = False
        if not ok:
            # AI trả từ lạ/sai định dạng: không để AI bịa từ -> dùng luật ghép (đây KHÔNG phải lỗi mạng, không tính vào bộ ngắt)
            with _GEM["lock"]:
                _GEM.update(last_ok=time.time(), last_err="")
            return None
        sentence = sentence.splitlines()[0].strip()
        res = (sentence, list(chosen))
        with _GEM["lock"]:
            _GEM.update(fails=0, last_ok=time.time(), last_err="")
            if len(_GEM["cache"]) > 200:
                _GEM["cache"].clear()
            _GEM["cache"][ck] = res
            _CONV.append(sentence)
            del _CONV[:-_CONV_MAX]
        return res
    except Exception as e:
        with _GEM["lock"]:
            _GEM["fails"] += 1
            _GEM["last_err"], _GEM["last_fail_t"] = str(e), time.time()
            if _GEM["fails"] >= 2:
                _GEM["skip_until"] = time.time() + 30
        return None


def _argos_only(text, src, dst):
    """Dịch bằng gói Argos trên máy (offline). Tự tải gói 1 lần nếu có mạng và còn thiếu."""
    if not ARGOS_OK:
        raise RuntimeError("Chưa cài được thư viện argostranslate.")
    try:
        ensure_argos_model(src, dst)
    except Exception:
        pass   # vẫn thử dịch bên dưới bằng những gói đã có
    try:
        argostranslate.translate.get_installed_languages.cache_clear()
    except Exception:
        pass
    try:
        out = argostranslate.translate.translate(text, src, dst)
        if out and out.strip():
            return out
    except Exception:
        pass
    # bắc cầu thủ công qua tiếng Anh
    if src != "en" and dst != "en":
        mid = argostranslate.translate.translate(text, src, "en")
        out = argostranslate.translate.translate(mid, "en", dst)
        if out and out.strip():
            return out
    installed = ", ".join(sorted(l.code for l in argostranslate.translate.get_installed_languages()))
    raise RuntimeError(f"Máy chưa có gói dịch {src} → {dst} (đã có: {installed or 'chưa có gói nào'}). Cần Internet 1 lần để tải.")


def argos_translate(text, src, dst):
    """Tự chọn: CÓ Internet thì dịch trực tuyến (chính xác hơn); MẤT mạng hoặc lỗi thì dùng gói dịch trên máy."""
    if src == dst:
        return text
    err = None
    if is_online():
        try:
            return online_translate(text, src, dst)
        except Exception as e:
            err = e
    try:
        return _argos_only(text, src, dst)
    except Exception as e2:
        raise RuntimeError(str(e2) if err is None else f"{e2} (dịch trực tuyến cũng lỗi: {err})")


def translate_to_vietnamese(text, source_code):
    return argos_translate(text, source_code, "vi")


class TranslatorWarmup:
    """Chuẩn bị sẵn gói dịch Việt -> ngôn ngữ đầu ra ở luồng nền ngay khi người dùng chọn ngôn ngữ,
    để câu đầu tiên không bị chờ (tải gói + nạp mô hình có thể mất vài chục giây)."""
    RETRY_AFTER = 30   # giây: lỗi thì lần sau mới thử lại, tránh tải lặp liên tục

    def __init__(self):
        self.lock = threading.Lock()
        self.status = {}   # mã ngôn ngữ -> (trạng thái, thông báo, thời điểm)

    def ensure(self, dst):
        with self.lock:
            cur = self.status.get(dst)
            if cur and (cur[0] in ("loading", "ready") or time.time() - cur[2] < self.RETRY_AFTER):
                return
            self.status[dst] = ("loading", "", time.time())
        threading.Thread(target=self._run, args=(dst,), daemon=True).start()

    def _run(self, dst):
        try:
            argos_translate("xin chào", "vi", dst)   # có mạng: dịch trực tuyến; mất mạng: gói trên máy
            result = ("ready", "", time.time())
            try:
                if is_online():
                    _argos_only("xin chào", "vi", dst)   # tranh thủ tải sẵn gói để lần sau mất mạng vẫn dịch được
            except Exception:
                pass
        except Exception as e:
            result = ("error", str(e), time.time())
        with self.lock:
            self.status[dst] = result

    def get(self, dst):
        with self.lock:
            cur = self.status.get(dst)
            return (cur[0], cur[1]) if cur else ("idle", "")


@st.cache_resource
def get_translator():
    return TranslatorWarmup()


def process_audio_file(audio_data, input_cfg=None):
    """Nhận diện giọng nói theo ngôn ngữ đã chọn rồi dịch sang Tiếng Việt bằng Argos (offline).
    Trả về (chữ gốc, bản dịch tiếng Việt, thông báo lỗi). Lỗi nhận diện: (None, None, msg).
    Lỗi dịch: (chữ gốc, None, msg)."""
    input_cfg = input_cfg or INPUT_LANGS[0]
    recognized = None
    wav_bytes = bytes(audio_data.getbuffer())

    # 1) Có Internet -> dùng Google (chính xác hơn)
    if is_online():
        try:
            import speech_recognition as sr
            with open("recorded_audio.wav", "wb") as f:
                f.write(wav_bytes)
            r = sr.Recognizer()
            with sr.AudioFile("recorded_audio.wav") as source:
                audio_file_data = r.record(source)
                recognized = r.recognize_google(audio_file_data, language=input_cfg["sr"])
        except Exception:
            recognized = None
        finally:
            if os.path.exists("recorded_audio.wav"):
                os.remove("recorded_audio.wav")

    # 2) Không có Internet (hoặc Google lỗi) -> nhận diện OFFLINE bằng Whisper/Vosk
    if not recognized:
        try:
            import offline_stt
            if offline_stt.available(input_cfg["argos"]):
                recognized = offline_stt.transcribe(wav_bytes, input_cfg["argos"]) or None
        except Exception:
            recognized = None

    if not recognized:
        return None, None, "⚠️ Không nghe rõ hoặc chưa nói gì, vui lòng bấm ghi âm lại..."

    try:
        translated = translate_to_vietnamese(recognized, input_cfg["argos"])
        return recognized, translated, None
    except Exception as e:
        return recognized, None, f"⚠️ Chưa dịch được sang Tiếng Việt: {e}"


# ------------------------------ Ngôn ngữ đầu ra + nhãn (không cần Internet) ------------------------------ #
# Muốn thêm ngôn ngữ (ví dụ Thái): thêm 1 dòng vào LANGS, ô nhập nhãn sẽ tự xuất hiện.
LANGS = [
    {"key": "vi", "label": "🇻🇳 Tiếng Việt",          "name": "Tiếng Việt", "tts": "vi",    "join": " "},
    {"key": "zh", "label": "🇨🇳 中文 (Tiếng Trung)",   "name": "Tiếng Trung", "tts": "zh-CN", "join": ""},
    {"key": "ko", "label": "🇰🇷 한국어 (Tiếng Hàn)",    "name": "Tiếng Hàn",  "tts": "ko",    "join": " "},
    {"key": "ja", "label": "🇯🇵 日本語 (Tiếng Nhật)",   "name": "Tiếng Nhật", "tts": "ja",    "join": ""},
    {"key": "en", "label": "🇬🇧 English (Tiếng Anh)", "name": "Tiếng Anh",  "tts": "en",    "join": " "},
]
EXTRA_LANGS = [lg for lg in LANGS if lg["key"] != "vi"]

LABELS_I18N_PATH = os.path.join(BASE_DIR, "vsl_labels_i18n.json")   # {"Cơm": {"en": "Rice", "zh": "米饭", ...}}
_labels_lock = threading.Lock()


def _read_labels_unlocked():
    try:
        with open(LABELS_I18N_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_labels():
    with _labels_lock:
        return _read_labels_unlocked()


def save_labels_for(word, labels):
    """Lưu nhãn các ngôn ngữ cho 1 từ (ô để trống = không có nhãn)."""
    clean = {k: v.strip() for k, v in labels.items() if v and v.strip()}
    with _labels_lock:
        data = _read_labels_unlocked()
        if clean:
            data[word] = clean
        else:
            data.pop(word, None)
        with open(LABELS_I18N_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)


def delete_labels_for(word):
    save_labels_for(word, {})


def get_label(word, key):
    """(chữ hiển thị, có_nhãn). Tiếng Việt chính là tên từ; ngôn ngữ khác tra trong file nhãn."""
    if key == "vi":
        return word, True
    text = load_labels().get(word, {}).get(key)
    return (text, True) if text else (word, False)


# ------------------------------ Ghép câu offline (bộ lọc ngữ pháp) ------------------------------ #
# Ký hiệu thường là từ rời: "tôi" + "đau đầu" + "bác sĩ". Bộ lọc này (không cần Internet, không cần AI) ghép
# thành câu trôi chảy: "Tôi bị đau đầu, tôi cần gặp bác sĩ". Hai lớp:
#   1) Bảng cụm từ cố định (phrases): tra đúng chuỗi từ -> câu hoàn chỉnh (có thể kèm bản dịch từng ngôn ngữ).
#   2) Luật ngữ pháp theo nhóm từ: chủ ngữ + (triệu chứng / người cần gặp / đồ ăn...) -> khung câu có sẵn.
# Mọi thứ nằm trong file vsl_sentence_rules.json (tự tạo lần đầu): thêm từ hoặc cụm từ mới chỉ cần sửa file này.
GRAMMAR_PATH = os.path.join(BASE_DIR, "vsl_sentence_rules.json")
DEFAULT_GRAMMAR = {
    "default_subject": "tôi",    # không ai nói chủ ngữ thì mặc định người đang ra hiệu là "tôi"
    "subject": ["tôi", "bạn", "em", "con", "cháu", "mình", "anh", "chị", "chúng tôi"],
    "symptom": ["đau đầu", "đau bụng", "đau răng", "đau họng", "đau tai", "đau mắt", "đau chân", "đau tay",
                "đau ngực", "đau lưng", "sốt", "ho", "chóng mặt", "buồn nôn", "mệt", "khó thở"],
    "meet": ["bác sĩ", "y tá", "cô giáo", "thầy giáo", "giáo viên", "công an", "bố", "mẹ", "ba", "má", "cha",
             "bố mẹ", "cha mẹ", "thầy cô", "nhân viên", "giám đốc", "bộ đội", "người quen", "bạn gái", "bạn trai",
             "ông bà", "anh chị em", "anh em"],
    "food": ["cơm", "phở", "bún", "bánh", "cháo", "trái cây", "rau"],
    "drink": ["nước", "sữa", "trà", "cà phê"],
    "need": ["thuốc", "giúp đỡ", "tiền", "sách", "bút", "điện thoại", "thuốc bổ", "cấp cứu", "xe cứu thương",
             "bút bi", "giấy", "cặp sách", "máy tính (laptop)", "máy vi tính", "internet", "xe máy", "xe đạp", "ô tô",
             "xe tải", "xe khách", "phiên dịch", "thông tin", "địa chỉ", "đồng hồ", "mũ", "kính", "áo"],
    "place": ["nhà vệ sinh", "bệnh viện", "trường học", "về nhà", "cửa hàng", "siêu thị", "chợ nổi", "khách sạn",
              "bến xe", "bến tàu", "công viên", "nhà trường", "trạm xe buýt", "trường tiểu học"],
    "goto": ["khám bệnh"],
    "greet": ["chào", "tạm biệt", "xin lỗi", "chúc mừng"],
    "action": ["đi", "chạy", "ngồi", "lấy", "đợi", "gọi", "hỏi", "trả lời", "mở", "nói chuyện", "gửi", "nhận",
               "chia sẻ", "gặp gỡ", "đi học", "đi dạo", "đi vệ sinh", "lên xe", "vào lớp", "ra ngoài", "làm việc",
               "làm bài tập", "học nhóm", "ăn uống", "rửa tay", "rửa mặt", "đánh răng", "nghỉ ngơi", "chơi",
               "vui chơi", "hiểu", "quên", "nghe", "nhìn", "xem", "không hiểu", "không muốn", "không cần",
               "cố gắng", "mời", "hẹn", "dạy", "về nhà", "chia tay", "múa", "nói dối", "học giỏi", "làm việc",
               "chăm học", "thương yêu", "tự tin"],
    "state": ["vui mừng", "buồn thảm", "giận dữ", "lo sợ", "mệt mỏi", "khát nước", "ốm đau", "khoẻ mạnh",
              "bình an", "đói", "mệt", "vui", "buồn", "sợ", "khóc"],
    "adj": ["xinh xắn", "dễ thương", "giỏi", "thông minh", "thật thà", "to lớn", "dài", "ngắn", "cao (người)",
            "mới", "gần gũi", "xa xôi", "quan trọng", "đặc biệt", "nguy hiểm", "may mắn", "bình thường", "khác",
            "thành công"],
    "family": ["em trai", "em gái", "con trai", "con gái", "ông nội", "ông ngoại", "cháu trai", "cháu gái",
               "người yêu"],
    "templates": {
        "adj": "{s} {w}",
        "family": "{s} có {w}",
        "action": "{s} {w}",
        "state": "{s} {w}",
        "goto": "{s} đi {w}",
        "symptom": "{s} bị {w}",
        "meet": "{s} cần gặp {w}",
        "food": "{s} muốn ăn {w}",
        "drink": "{s} muốn uống {w}",
        "need": "{s} cần {w}",
        "place": "{s} muốn đi {w}",
    },
    "phrases": {
        "tôi|đau đầu|bác sĩ": {
            "vi": "Tôi bị đau đầu, tôi cần gặp bác sĩ",
            "en": "I have a headache and I need to see a doctor",
            "zh": "我头疼，我需要看医生",
            "ko": "머리가 아파요. 의사를 만나야 해요",
            "ja": "頭が痛いです。医者に診てもらいたいです",
        },
    },
}
_KIND_ORDER = ["subject", "greet", "action", "goto", "state", "symptom", "meet", "food", "drink", "need", "place",
               "adj", "family"]
_SUBJECT_ONLY = {"adj", "family", "question"}   # chỉ ghép khi có chủ ngữ đứng ngay trước, không tự thêm "tôi"
_MODAL = {"muốn"}                     # "muốn" + món ăn/đồ uống/nơi chốn: câu mẫu đã có sẵn "muốn"


def _clean(w):
    """Bỏ phần chú thích trong ngoặc và phần sau dấu / của nhãn: "cao (người)" -> "cao", "trẻ con/con nít" -> "trẻ con"."""
    c = re.sub(r"\s*\([^)]*\)", "", str(w)).split("/")[0].strip()
    return c or str(w).strip()


def _norm(word):
    return re.sub(r"\s+", " ", str(word).strip().lower())


def ensure_grammar_file():
    if not os.path.exists(GRAMMAR_PATH):
        try:
            with open(GRAMMAR_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_GRAMMAR, f, ensure_ascii=False, indent=1)
        except OSError:
            pass


def load_grammar():
    """Mặc định + nội dung file (file bổ sung/ghi đè). Lỗi đọc file thì dùng mặc định, không bao giờ làm sập app."""
    g = json.loads(json.dumps(DEFAULT_GRAMMAR))
    try:
        with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
            user = json.load(f)
    except Exception:
        return g
    for kind in _KIND_ORDER:
        extra = user.get(kind)
        if isinstance(extra, list):
            g[kind] = list(dict.fromkeys(g[kind] + [str(x) for x in extra]))
    for key in ("templates", "phrases"):
        if isinstance(user.get(key), dict):
            g[key].update(user[key])
    if isinstance(user.get("default_subject"), str) and user["default_subject"].strip():
        g["default_subject"] = user["default_subject"].strip()
    return g


SUBJECT_MAP = {"tôi": "tôi", "em": "tôi", "con": "tôi", "cháu": "tôi", "mình": "tôi", "chúng tôi": "tôi",
               "anh": "bạn", "chị": "bạn", "bạn": "bạn"}   # chỉ xưng "tôi" (người ký) và "bạn" (người đối diện)


def compose_vietnamese(words, g):
    """Ghép các từ rời thành câu tiếng Việt lịch sự, chỉ xưng "tôi" và "bạn". Từ không thuộc nhóm nào vẫn được giữ nguyên."""
    toks = []
    for w in words:
        n = _norm(w)
        if n and (not toks or toks[-1] != n or n.isdigit()):   # bỏ từ lặp liền kề do ký hiệu bị nhận hai lần (chữ số thì giữ)
            toks.append(n)
    toks = [_norm(_clean(t)) for t in toks]
    merged = []
    for t in toks:                                  # các chữ số ký liền nhau ("1", "0", "0") gộp thành một cụm số
        if merged and t.isdigit() and merged[-1].replace(" ", "").isdigit():
            merged[-1] += " " + t
        else:
            merged.append(t)
    toks = merged
    kind_of = {}
    for kind in reversed(_KIND_ORDER):               # nhóm đứng trước được ưu tiên
        for w in g.get(kind, []):
            kind_of[_norm(w)] = kind
            kind_of.setdefault(_norm(_clean(w)), kind)
    for t in toks:
        if t not in kind_of and t.endswith("?"):
            kind_of[t] = "question"
    # "bạn" đứng liền sau lời chào/xin lỗi... là người được chào, không phải chủ ngữ mới
    clauses, pending, last_subject = [], None, None
    i = 0
    while i < len(toks):
        w = toks[i]
        i += 1
        kind = kind_of.get(w)
        if kind == "greet":
            if pending:
                clauses.append(pending)
                pending = None
            if i < len(toks) and SUBJECT_MAP.get(toks[i]) == "bạn" and kind_of.get(toks[i]) == "subject":
                i += 1                                # "chào" + "bạn" -> "Chào bạn"
            clauses.append(f"{w} bạn")
            continue
        if kind == "subject":
            if pending:                              # hai chủ ngữ liền nhau: chủ ngữ trước đứng một mình
                clauses.append(pending)
            pending = last_subject = SUBJECT_MAP.get(w, "tôi")
            continue
        if w in _MODAL:
            nxt = toks[i] if i < len(toks) else None
            nk = kind_of.get(nxt) if nxt else None
            if nk in ("food", "drink", "place"):          # câu mẫu của các nhóm này đã có sẵn "muốn"
                continue
            subject = pending or last_subject or SUBJECT_MAP.get(g["default_subject"], "tôi")
            if nk in ("action", "goto"):                  # "muốn" + hành động: "Tôi muốn đi học"
                clauses.append(f"{subject} muốn {nxt}")
                pending, i = None, i + 1
            else:
                clauses.append(f"{subject} muốn")
                pending = None
            last_subject = subject
            continue
        if kind in _SUBJECT_ONLY:
            if pending:
                tpl = g["templates"].get(kind, "{s} {w}")
                clauses.append(tpl.format(s=pending, w=w))
                last_subject, pending = pending, None
            else:
                clauses.append(w)
            continue
        template = g["templates"].get(kind) if kind else None
        if template:
            subject = pending or last_subject or SUBJECT_MAP.get(g["default_subject"], "tôi")
            clauses.append(template.format(s=subject, w=w))
            pending = None
        else:
            if pending:
                clauses.append(pending)
                pending = None
            clauses.append(w)
    if pending:
        clauses.append(pending)
    text = ", ".join(clauses)
    if text and text[-1] not in ".?!":
        text += "."
    return text[:1].upper() + text[1:]


def build_vietnamese_sentence(words):
    """Bước 1 (luôn làm trước): hoàn thiện CÂU TIẾNG VIỆT từ các từ rời. Câu này chỉ là trung gian,
    chỉ hiển thị khi người dùng chọn đầu ra Tiếng Việt."""
    g = load_grammar()
    phrase = g["phrases"].get("|".join(_norm(w) for w in words))
    if isinstance(phrase, str):
        return phrase
    if isinstance(phrase, dict) and phrase.get("vi"):
        return phrase["vi"]
    return compose_vietnamese(words, g)


def _polish_ok(orig, new):
    """Chỉ nhận câu Google làm mềm khi vẫn giữ đúng cách xưng tôi/bạn, không thêm đại từ lạ, không dài bất thường."""
    if not new:
        return False
    tk = lambda t: set(re.findall(r"\w+", t.lower()))
    o, n = tk(orig), tk(new)
    for pron in ("tôi", "bạn"):
        if pron in o and pron not in n:
            return False
    for bad in ("anh", "chị", "em", "mình", "cháu", "ngài", "quý", "con"):
        if bad in n and bad not in o:
            return False
    return len(new) <= len(orig) * 2 + 12


TRANS_CACHE_PATH = os.path.join(BASE_DIR, "dich_cache.json")
_cache_lock = threading.Lock()


def _cache_get(vi_text, key):
    try:
        with _cache_lock, open(TRANS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get(key + "|" + _norm(vi_text))
    except Exception:
        return None


def _cache_put(vi_text, key, out):
    try:
        with _cache_lock:
            try:
                with open(TRANS_CACHE_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                d = {}
            d[key + "|" + _norm(vi_text)] = out
            with open(TRANS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def translate_sentence(words, vi_text, lang_cfg, alts=None):
    """Bước 2: dịch nguyên câu tiếng Việt sang ngôn ngữ đầu ra (chạy ở luồng ngầm, KHÔNG gọi st.*).
    Thứ tự: (1) cụm câu đã kiểm duyệt trong vsl_sentence_rules.json -> (2) dịch cả câu bằng Argos
    -> (3) ghép nhãn đã nhập cho từng từ -> (4) cuối cùng mới hiện tiếng Việt kèm cảnh báo.
    Trả về (câu hiển thị, mã giọng đọc, nguồn, thông báo lỗi)."""
    key, tts_code = lang_cfg["key"], lang_cfg["tts"]
    g = load_grammar()
    phrase = g["phrases"].get("|".join(_norm(w) for w in words))
    if isinstance(phrase, dict) and phrase.get(key):
        return phrase[key], tts_code, "phrase", ""
    err = ""
    ai_used = False
    en_pivot = None
    if is_online():
        ai = ai_compose(words, alts)                # (tuỳ chọn) có khóa Gemini: AI tự sắp xếp câu + sửa từ nhận nhầm theo ngữ cảnh
        if ai:
            vi_text, ai_used = ai[0], True
            words = ai[1]
        try:                                        # KHÔNG cần khóa: Google dịch câu sang tiếng Anh để làm câu tự nhiên, lịch sự hơn
            en_pivot = online_translate(vi_text, "vi", "en").strip() or None
        except Exception as e:
            err = f"Google lỗi: {e}"
            _net_cache[0], _net_cache[1] = time.time(), False   # vừa mất mạng: các bước sau đi thẳng đường offline
    if key == "vi":
        if en_pivot and not ai_used:                # Việt -> Anh -> Việt: Google tự làm mềm câu ("Xin chào", "đang"...)
            try:
                back = online_translate(en_pivot, "en", "vi").strip()
                if _polish_ok(vi_text, back):
                    return back[:1].upper() + back[1:], tts_code, "polish", ""
            except Exception as e:
                err = f"Google lỗi: {e}"
        return vi_text, tts_code, ("ai" if ai_used else "rule"), err
    if en_pivot:                                    # dịch từ câu tiếng Anh tự nhiên sang ngôn ngữ đích
        try:
            out = en_pivot if key == "en" else online_translate(en_pivot, "en", key).strip()
            if out:
                out = out[:1].upper() + out[1:] if key == "en" else out
                _cache_put(vi_text, key, out)           # nhớ lại để lần sau mất mạng vẫn có bản dịch chuẩn của Google
                return out, tts_code, "online", ""
        except Exception as e:
            err = f"Google lỗi: {e}"
    cached = _cache_get(vi_text, key)
    if cached:
        return cached, tts_code, "online", ""
    try:
        out = argos_translate(vi_text, "vi", key)
        if out and out.strip():
            return out.strip(), tts_code, "argos", ""
        err = "bản dịch rỗng"
    except Exception as e:
        err = str(e)
    parts = [get_label(w, key) for w in words]
    if all(found for _text, found in parts):
        return lang_cfg["join"].join(text for text, _found in parts), tts_code, "labels", err
    return vi_text, "vi", "fallback", err


def _translate_worker(words, vi_text, lang_cfg, out_queue, tts=None, alts=None):
    """Dịch xong là ĐỌC NGAY trong luồng này (không chờ vòng lặp camera), rồi mới báo kết quả để hiển thị."""
    try:
        result = translate_sentence(words, vi_text, lang_cfg, alts)
    except Exception as e:   # không để luồng ngầm chết mà không báo
        result = (vi_text, "vi", "fallback", str(e))
    try:
        if tts is not None:
            tts.speak(result[0], result[1])
    except Exception:
        pass
    out_queue.put(result)


class OfflineTTS:
    """Một luồng riêng đọc lần lượt từng câu, KHÔNG cần Internet. Thứ tự thử cho mỗi ngôn ngữ:
    (1) giọng Windows SAPI5 qua pyttsx3 -> (2) giọng Windows OneCore (WinRT, các giọng cài ở Cài đặt > Giọng nói)
    -> (3) gTTS (chỉ khi có Internet)."""
    KEYWORDS = {
        "vi": ["vi-vn", "vi_vn", "vivn", "vietnam"],
        "zh-CN": ["zh-cn", "zh_cn", "zhcn", "huihui", "yaoyao", "kangkang", "chinese", "mandarin"],
        "ko": ["ko-kr", "ko_kr", "kokr", "heami", "korean"],
        "ja": ["ja-jp", "ja_jp", "jajp", "haruka", "ayumi", "sayaka", "japanese"],
        "en": ["en-us", "en_us", "enus", "en-gb", "engb", "david", "zira", "english"],
    }
    WINRT_PREFIX = {"vi": "vi", "zh-CN": "zh", "ko": "ko", "ja": "ja", "en": "en"}

    def __init__(self):
        self.q = queue.Queue()
        self.voices = {}                 # mã ngôn ngữ -> id giọng SAPI5 (None nếu máy chưa cài)
        self.winrt_langs = set()         # ngôn ngữ có giọng OneCore
        self.error = None
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _find(voices, keywords):
        for kw in keywords:
            for v in voices:
                hay = f"{getattr(v, 'id', '')} {getattr(v, 'name', '')} {getattr(v, 'languages', '')}".lower()
                if kw in hay:
                    return v.id
        return None

    @staticmethod
    def _make_engine():
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        return engine

    def _detect_winrt(self):
        try:
            from winrt.windows.media.speechsynthesis import SpeechSynthesizer
            langs = [str(v.language).lower() for v in SpeechSynthesizer.all_voices]
            for code, pre in self.WINRT_PREFIX.items():
                if any(l.startswith(pre) for l in langs):
                    self.winrt_langs.add(code)
        except Exception:
            pass

    def _run(self):
        engine = None
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()      # Windows: luồng riêng cần khởi tạo COM
            except Exception:
                pass
            try:
                engine = self._make_engine()
                installed = engine.getProperty("voices")
                for lang, kws in self.KEYWORDS.items():
                    self.voices[lang] = self._find(installed, kws)
            except Exception as e:
                self.error = str(e)
            self._detect_winrt()
            if self.winrt_langs:
                self.error = None if self.error and self.winrt_langs else self.error
        finally:
            self.ready.set()

        while True:
            text, lang = self.q.get()
            while not self.q.empty():          # có câu mới hơn đang chờ thì bỏ câu cũ, để đọc ngay câu mới nhất
                try:
                    text, lang = self.q.get_nowait()
                except queue.Empty:
                    break
            spoken = False
            vid = self.voices.get(lang)

            def via_sapi():
                if not vid:
                    raise RuntimeError("máy chưa có giọng SAPI cho " + lang)
                eng = self._make_engine()                   # engine mới mỗi câu: tránh lỗi pyttsx3 chỉ nói được 1 giọng/1 lần
                eng.setProperty("voice", vid)
                eng.say(text)
                eng.runAndWait()
                try: eng.stop()
                except Exception: pass

            def via_winrt():
                if lang not in self.winrt_langs:
                    raise RuntimeError("máy chưa có giọng OneCore cho " + lang)
                self._speak_winrt(text, lang)

            def via_online():
                self._speak_online(text, lang)

            # Có Internet: Google đọc trước (giọng chuẩn, đủ mọi ngôn ngữ); mất mạng hoặc lỗi: giọng Windows trong máy.
            order = [("google", via_online), ("sapi", via_sapi), ("winrt", via_winrt)] if is_online() \
                else [("sapi", via_sapi), ("winrt", via_winrt), ("google", via_online)]
            for name, fn in order:
                try:
                    fn()
                    spoken = True
                    self._log(f"OK  {lang} bang {name}")
                    break
                except Exception as e:
                    self._log(f"LOI {lang} bang {name}: {e}")
            if not spoken:
                self._log(f"KHONG DOC DUOC {lang}: {text}")

    @staticmethod
    def _log(msg):
        try:
            with open(os.path.join(BASE_DIR, "tts_log.txt"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + msg + "\n")
        except Exception:
            pass

    def _speak_winrt(self, text, lang):
        """Giọng Windows OneCore, chạy offline: tổng hợp ra WAV trong bộ nhớ rồi phát."""
        import asyncio, winsound
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer
        from winrt.windows.storage.streams import DataReader
        pre = self.WINRT_PREFIX[lang]

        async def synth():
            sy = SpeechSynthesizer()
            cand = [v for v in SpeechSynthesizer.all_voices if str(v.language).lower().startswith(pre)]
            if cand:
                sy.voice = cand[0]
            stream = await sy.synthesize_text_to_stream_async(text)
            reader = DataReader(stream.get_input_stream_at(0))
            await reader.load_async(stream.size)
            buf = bytearray(stream.size)
            reader.read_bytes(buf)
            return bytes(buf)

        data = asyncio.run(synth())
        winsound.PlaySound(data, winsound.SND_MEMORY)

    @staticmethod
    def _speak_online(text, lang):
        """Google đọc (cần Internet). Phát bằng playsound; nếu lỗi thì phát bằng Windows Media Player qua PowerShell."""
        from gtts import gTTS
        filename = os.path.abspath(f"speech_{int(time.time() * 1000)}.mp3")
        try:
            gTTS(text=text, lang=lang).save(filename)
            try:
                from playsound import playsound
                playsound(filename)
            except Exception:
                import subprocess
                ps = ("Add-Type -AssemblyName presentationCore;$p=New-Object System.Windows.Media.MediaPlayer;"
                      f"$p.Open([uri]'{filename}');$p.Play();Start-Sleep -Milliseconds 800;"
                      "while(-not $p.NaturalDuration.HasTimeSpan){Start-Sleep -Milliseconds 100};"
                      "Start-Sleep -Seconds ([math]::Ceiling($p.NaturalDuration.TimeSpan.TotalSeconds))")
                subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=60)
        finally:
            if os.path.exists(filename):
                try: os.remove(filename)
                except Exception: pass

    def speak(self, text, lang):
        self.q.put((text, lang))

    def available(self):
        langs = {lang for lang, vid in self.voices.items() if vid}
        return langs | self.winrt_langs


@st.cache_resource
def get_tts():
    return OfflineTTS()


# ------------------------------ Trạng thái học ngầm ------------------------------ #
class TrainingState:
    """Luồng học ngầm không gọi st.*; nó chỉ ghi trạng thái vào đây, vòng lặp camera đọc ra."""
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.message = ""
        self.progress = 0.0
        self.error = None
        self.model_ready = False
        self.hidden_words = set()   # từ vừa xóa nhưng mô hình cũ chưa được thay

    def start(self):
        with self.lock:
            if self.running:
                return False
            self.running, self.message, self.progress = True, "Bắt đầu...", 0.0
            self.error, self.model_ready = None, False
            return True

    def update(self, message, progress):
        with self.lock:
            self.message, self.progress = message, progress

    def finish(self, message, trained):
        with self.lock:
            self.running, self.message, self.progress = False, message, 1.0
            self.model_ready = trained

    def fail(self, message):
        with self.lock:
            self.running, self.error = False, message

    def consume_model_ready(self):
        with self.lock:
            ready, self.model_ready = self.model_ready, False
            if ready:
                self.hidden_words.clear()   # mô hình mới đã nạp, từ đã xóa không còn trong "bộ não"
            return ready

    def mark_deleted(self, word):
        with self.lock:
            self.hidden_words.add(word)

    def is_hidden(self, word):
        with self.lock:
            return word in self.hidden_words

    def snapshot(self):
        with self.lock:
            return self.running, self.message, self.progress, self.error


@st.cache_resource
def get_train_state():
    return TrainingState()


def _train_worker(video_path, word, state):
    try:
        result = te.train_new_sign(video_path, word, progress_cb=state.update)
        state.finish(result["message"], True)   # luôn nạp lại: mẫu DTW mới có hiệu lực ngay cả khi chưa train LSTM
    except Exception as e:
        state.fail(str(e))
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


def _retrain_worker(state):
    try:
        result = te.retrain_from_dataset(progress_cb=state.update)
        state.finish(result["message"], True)   # True: luôn nạp lại (có thể mô hình đã bị xóa)
    except Exception as e:
        state.fail(str(e))


def delete_word_and_update(word, state):
    """Xóa dữ liệu của 1 từ; còn >=2 từ thì train lại ngầm, <2 từ thì xóa mô hình. True nếu đã xử lý."""
    if not state.start():   # chặn xóa khi AI đang học dở (tránh đọc dữ liệu đang bị xóa)
        st.warning("⏳ AI đang học, vui lòng đợi xong rồi hãy xóa từ.")
        return False
    try:
        te.delete_word(word)
    except Exception as e:
        state.fail(f"Không xóa được '{word}': {e}")
        return False

    delete_labels_for(word)
    state.mark_deleted(word)   # camera ngừng đọc từ này ngay, kể cả khi mô hình cũ chưa được thay
    if len(te.list_actions()) < te.MIN_CLASSES:
        try:
            te.remove_model_files()
        except OSError as e:
            state.fail(f"Đã xóa '{word}' nhưng chưa xóa được file mô hình: {e}")
            return True
        state.finish(f"Đã xóa '{word}'. Còn dưới {te.MIN_CLASSES} từ nên mô hình LSTM đã được xóa; "
                     f"AI vẫn nhận diện các từ còn lại bằng khớp mẫu.", True)
    else:
        state.update(f"Đã xóa '{word}'. AI đang học lại để quên từ này...", 0.05)
        threading.Thread(target=_retrain_worker, args=(state,), daemon=True).start()
    return True


def _accept_suggestion(word, seg):
    """Người dùng bấm xác nhận 1 gợi ý: thêm từ vào câu VÀ lưu đoạn ký hiệu này làm mẫu mới (AI học từ chính camera)."""
    st.session_state.vsl_word_buffer.append(word)
    st.session_state.vsl_alt_buffer.append([])
    st.session_state.last_vsl_speak_time = time.time()
    try:
        te.save_sample(word, seg)
    except Exception:
        pass


def load_ai():
    """Trả về (None, [], bộ nhận diện dựng từ dataset_vsl)."""
    model, actions = None, []
    try:
        model, actions = te.load_model_and_labels()
    except Exception as e:
        st.warning(f"⚠️ Không tải được mô hình LSTM: {e}")
    try:
        with st.spinner("Đang nạp bộ nhận diện (lần đầu có thể mất khoảng 1 phút)..."):
            bundle = te.load_templates_cached()   # dùng bản lưu sẵn, chỉ đồng bộ phần thay đổi
    except Exception as e:
        st.warning(f"⚠️ Không dựng được bộ mẫu nhận diện: {e}")
        bundle = {"actions": [], "templates": {}, "thresholds": {}}
    return model, actions, bundle


# ------------------------------ Giao diện VSL ------------------------------ #
CONF_THRESHOLD = 0.6      # độ tự tin tối thiểu của LSTM để chấp nhận một từ (cùng với khớp mẫu DTW)
SHOW_SUGGESTIONS = False  # ĐI THI: giữ False (giao diện cũ). Ở nhà muốn thu mẫu từ camera thì đặt True. True: hiện nút gợi ý khi AI chưa chắc. Đặt False để TẮT hẳn tính năng này (đúng như báo cáo)
SENTENCE_TIMEOUT = 2.5    # giây: tay hạ xuống / dừng múa quá thời gian này -> coi là xong câu, ghép và đọc 1 lần


def run_vsl_mode():
    inject_css()
    ensure_grammar_file()

    # Bộ đệm ghép câu của phiên làm việc (giữ nguyên khi Streamlit chạy lại trang)
    if "vsl_word_buffer" not in st.session_state:
        st.session_state.vsl_word_buffer = []
    if "vsl_alt_buffer" not in st.session_state:
        st.session_state.vsl_alt_buffer = []
    if "last_vsl_speak_time" not in st.session_state:
        st.session_state.last_vsl_speak_time = time.time()
    if "vsl_last_sentence" not in st.session_state:
        st.session_state.vsl_last_sentence = ""

    st.markdown('<p class="page-title">🇻🇳 Hệ thống giao tiếp hai chiều Tiếng Việt</p>'
                '<p class="page-sub">Nhận diện ký hiệu tay bằng AI kết hợp nhận diện giọng nói, đọc kết quả thành tiếng.</p>',
                unsafe_allow_html=True)
    if te.LEGACY_MOVED:
        st.warning(f"⚠️ Cách chuẩn hóa tọa độ đã đổi (theo vai) nên dữ liệu cũ không dùng chung được. "
                   f"Dữ liệu cũ được cất an toàn trong thư mục “{te.LEGACY_MOVED}”. Hãy dạy lại các từ bằng video.")

    train_state = get_train_state()

    with st.expander("🤖 Trạng thái AI ghép câu (Gemini)"):
        st.caption(ai_status())
        if st.button("Kiểm tra AI ngay", key="ai_selftest_btn"):
            with st.spinner("Đang thử gọi AI..."):
                _ok, _msg = ai_selftest()
            (st.success if _ok else st.warning)(_msg)
        st.caption("Muốn bật AI: lấy khóa miễn phí ở aistudio.google.com/apikey, ghi vào file gemini_key.txt (dòng đầu là khóa). "
                   "Không có mạng hoặc AI lỗi thì app tự dùng luật ghép câu offline, không bị đứng.")

    # --- Khu vực Micro: hộp Gradient ---
    with st.container(key="mic_box"):
        _live_on = bool(LIVE_CHAT and live_bridge is not None)   # có chat trực tiếp (nghe liên tục) thì bỏ phần ghi âm thủ công
        sign_text = ""
        if _live_on:
            sec_head("👐", "Xem ký hiệu", "Gõ chữ để xem ký hiệu. Người bình thường cứ nói ở khung trò chuyện bên phải, máy nghe liên tục.")
        else:
            sec_head("🎙️", "Người bình thường nói", "Bấm ghi âm, nói xong bấm dừng. Chữ sẽ hiện thật lớn bên dưới.")
            input_lang = st.selectbox("🌍 Chọn ngôn ngữ người nói:", [lg["label"] for lg in INPUT_LANGS])
            input_cfg = next(lg for lg in INPUT_LANGS if lg["label"] == input_lang)
            audio_file = st.audio_input("Ghi âm giọng nói", label_visibility="collapsed")
            if audio_file is not None:
                with st.spinner("🔄 Đang nhận diện giọng nói và dịch sang Tiếng Việt..."):
                    recognized_text, translated_text, audio_err = process_audio_file(audio_file, input_cfg)
                if recognized_text is None:   # không nghe rõ
                    box = f'<div class="stt-box warn"><div class="txt">{html.escape(audio_err)}</div></div>'
                elif translated_text is None:   # nghe được nhưng dịch lỗi: vẫn hiện chữ gốc
                    box = (f'<div class="stt-box warn"><div class="lab">🌍 Chữ gốc ({html.escape(input_cfg["name"])}): '
                           f'{html.escape(recognized_text)}</div><div class="txt">{html.escape(audio_err)}</div></div>')
                elif input_cfg["argos"] == "vi":   # nói Tiếng Việt: không cần dịch
                    box = (f'<div class="stt-box"><div class="lab">💬 Văn bản từ giọng nói</div>'
                           f'<div class="txt">{html.escape(translated_text)}</div></div>')
                else:   # song ngữ: chữ gốc nhỏ ở trên, Tiếng Việt to ở dưới
                    box = (f'<div class="stt-box"><div class="lab">🌍 Chữ gốc ({html.escape(input_cfg["name"])}): '
                           f'{html.escape(recognized_text)}</div>'
                           f'<div class="lab">🇻🇳 Dịch sang Tiếng Việt</div>'
                           f'<div class="txt">{html.escape(translated_text)}</div></div>')
                st.markdown(box, unsafe_allow_html=True)
                if recognized_text is not None and translated_text:
                    sign_text = translated_text
        if avatar is not None:
            # CHIỀU NGƯỢC LẠI: lời nói/chữ gõ -> avatar ra ký hiệu cho người khiếm thính xem (dùng dữ liệu tay có sẵn, không cần Internet)
            typed = st.text_input("⌨️ Hoặc gõ chữ để xem ký hiệu:", key="sign_typed",
                                  placeholder="Ví dụ: chào bạn, tôi đau đầu")
            _src = (typed or "").strip() or sign_text
            if _src:
                try:
                    _html, _found, _missing = avatar.build_player(_src)
                except Exception as _e:
                    _html, _found, _missing = None, [], []
                    st.caption(f"Chưa tạo được ký hiệu: {_e}")
                if _html:
                    import streamlit.components.v1 as _components
                    st.markdown("**👐 Ký hiệu cho người khiếm thính xem**")
                    _components.html(_html, height=600)
                    if _missing:
                        st.caption("Chưa có ký hiệu cho: " + ", ".join(_missing))
                elif _src:
                    st.caption("Chưa có ký hiệu nào khớp với câu này trong bộ từ.")
    st.write("")

    # --- Bảng công nghệ ẩn: AI tự học từ video ---
    model, actions, bundle = load_ai()

    with st.expander("🛠️ Cập nhật bộ dữ liệu AI tự học"):
        new_word = st.text_input("Từ mới (ví dụ: Cơm, Nước):")
        video_up = st.file_uploader("Video ký hiệu (.mp4)", type=["mp4"])
        st.caption("Khi quay video: ngồi hoặc đứng đều được, nhưng phải thấy đủ HAI VAI và hai tay trong khung hình.")

        word_key = te.safe_label(new_word)   # đúng tên thư mục mà engine sẽ dùng
        existing_labels = load_labels().get(word_key, {}) if word_key else {}
        st.caption("Nhãn từng ngôn ngữ của từ này, dùng DỰ PHÒNG khi chưa dịch tự động được cả câu (để trống cũng được):")
        label_inputs = {}
        label_cols = st.columns(len(EXTRA_LANGS))
        for col, lg in zip(label_cols, EXTRA_LANGS):
            label_inputs[lg["key"]] = col.text_input(
                lg["label"], value=existing_labels.get(lg["key"], ""), key=f"lbl_{lg['key']}_{word_key}")

        if st.button("💾 CHỈ LƯU NHÃN (không cần video)"):
            if not word_key:
                st.warning("⚠️ Vui lòng nhập tên từ (tiếng Việt).")
            elif word_key not in te.list_actions():
                st.warning("⚠️ Từ này chưa được dạy bằng video. Hãy tải video và bấm “TỰ ĐỘNG CẬP NHẬT AI”.")
            else:
                save_labels_for(word_key, label_inputs)
                st.success(f"✅ Đã lưu nhãn cho '{word_key}'.")

        if st.button("TỰ ĐỘNG CẬP NHẬT AI"):
            if not word_key:
                st.warning("⚠️ Vui lòng nhập tên từ mới.")
            elif video_up is None:
                st.warning("⚠️ Vui lòng tải lên video .mp4.")
            elif not train_state.start():
                st.warning("⏳ AI đang học một từ khác, vui lòng đợi xong.")
            else:
                save_labels_for(word_key, label_inputs)   # lưu nhãn các ngôn ngữ cùng lúc với video
                # Lưu video tạm ở luồng chính rồi giao đường dẫn cho luồng học ngầm
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(video_up.getbuffer())
                tmp.close()
                threading.Thread(target=_train_worker,
                                 args=(tmp.name, new_word.strip(), train_state),
                                 daemon=True).start()
                missing = [lg["name"] for lg in EXTRA_LANGS if not label_inputs[lg["key"]].strip()]
                st.info("🚀 AI đang học ngầm, camera vẫn hoạt động bình thường."
                        + (f" Chưa có nhãn: {', '.join(missing)} (bổ sung sau bằng nút “CHỈ LƯU NHÃN”)." if missing else ""))

        # Bảng tiến độ học ngầm (Progress Dashboard) - vòng lặp camera cập nhật vào đây
        status_placeholder = st.empty()

        # --- Quản lý bộ từ vựng đã dạy ---
        st.divider()
        taught_words = te.list_actions()   # quét thư mục dataset_vsl bằng os.listdir
        st.markdown('<div class="chips-title">🧩 Các từ AI đã có dữ liệu</div>' + chips_html(taught_words[:60]),
                    unsafe_allow_html=True)
        if len(taught_words) > 60:
            st.caption("Chỉ hiện một phần các từ ở trên. Chọn từ ở ô bên dưới để xem chi tiết.")
        if taught_words:
            pick = st.selectbox("Chọn từ để xem / xóa:", taught_words, key="pick_word")
            st.markdown(f"**{pick}** · {te.count_samples(pick)} video")
            word_labels = load_labels().get(pick, {})
            st.caption("  ·  ".join(f"{lg['label'].split()[0]} {word_labels[lg['key']]}"
                                    for lg in EXTRA_LANGS if lg["key"] in word_labels) or "chưa có nhãn ngoại ngữ")
            if st.button(f"❌ Xóa từ '{pick}'", key="del_pick", type="primary"):
                if delete_word_and_update(pick, train_state):
                    st.rerun()   # vẽ lại danh sách; luồng học ngầm vẫn chạy tiếp

    # --- Camera + kết quả ---
    st.write("")
    sec_head("🖐️", "Dành cho người khiếm thính",
             "Ngồi hoặc đứng đều được, miễn thấy hai vai. Làm ký hiệu rồi hạ tay xuống; dừng khoảng 2,5 giây để máy ghép và đọc cả câu.")
    with st.container(key="ctl_box"):
        col_lang, col_tol, col_clear = st.columns([2, 3, 1])
        lang_label = col_lang.selectbox("🌐 Ngôn ngữ đầu ra:", [lg["label"] for lg in LANGS])
        lang_cfg = next(lg for lg in LANGS if lg["label"] == lang_label)
        translator = get_translator()
        if lang_cfg["key"] != "vi":
            translator.ensure(lang_cfg["key"])   # tải/nạp gói dịch Việt -> ngôn ngữ này ở luồng nền
        translate_status_placeholder = col_lang.empty()
        tts = get_tts()
        tts.ready.wait(timeout=3)
        if tts.error:
            col_lang.caption("")
        elif lang_cfg["tts"] not in tts.available():
            col_lang.caption(f"🔇 Máy chưa có giọng đọc {lang_cfg['name']} (offline). Cài ở: Cài đặt Windows → "
                             f"Thời gian & ngôn ngữ → Giọng nói → Thêm giọng nói.")
        tolerance = col_tol.slider(
            "Độ dung sai nhận diện", 0.5, 2.0, 1.0, 0.1,
            help="Nhỏ = nghiêm (ít đọc nhầm, dễ bỏ sót). Lớn = dễ dãi (dễ nhận hơn, có thể đọc nhầm). "
                 "Nếu AI hay bỏ sót ký hiệu đúng thì tăng; nếu hay đọc bừa thì giảm.")
        _reg_opts = {"Không chọn": None, "Bắc": "B", "Trung": "T", "Nam": "N"}
        _reg_labels = list(_reg_opts)
        _reg_idx = next((i for i, v in enumerate(_reg_opts.values()) if v == te.REGION_PREF), 0)
        _reg = col_tol.selectbox(
            "🗺️ Vùng miền ký hiệu:", _reg_labels, index=_reg_idx,
            help="Mỗi vùng (Bắc/Trung/Nam) có cách ký riêng cho nhiều từ. Chọn vùng bạn ký thì máy ưu tiên mẫu của vùng đó.")
        if _reg_opts[_reg] != te.REGION_PREF:
            te.save_region(_reg_opts[_reg])
        # Bộ từ theo tình huống: file words_<tên>.txt (tạo bằng tao_bo_tu.bat). Ít từ hơn = ít bị nhầm hơn.
        _set_files = sorted(os.path.basename(p) for p in _glob.glob(os.path.join(te.BASE_DIR, "words_*.txt")))
        if _set_files:
            _sets = {"Mặc định (words.txt)": "words.txt"}
            for _fn in _set_files:
                _sets[_fn[6:-4].replace("_", " ")] = _fn
            _sets["Toàn bộ kho"] = None
            _set_path = os.path.join(te.BASE_DIR, "vsl_wordset.txt")
            try:
                _cur = open(_set_path, encoding="utf-8").read().strip()
            except OSError:
                _cur = ""
            _names = list(_sets)
            _set_idx = _names.index(_cur) if _cur in _sets else (0 if os.path.exists(te.ACTIVE_PATH) else len(_names) - 1)
            _pick = col_tol.selectbox("📚 Ưu tiên bộ từ:", _names, index=_set_idx,
                                      help="Máy luôn nhận diện được toàn bộ kho từ. Bộ từ chọn ở đây (ví dụ bệnh viện, trường học) chỉ được ưu tiên nhẹ, không cần chọn cũng được.")
            if _pick != _cur and (_cur or _pick != _names[_set_idx]):
                try:
                    if _sets[_pick] is None:
                        if os.path.exists(te.ACTIVE_PATH):
                            os.remove(te.ACTIVE_PATH)
                    else:
                        with open(os.path.join(te.BASE_DIR, _sets[_pick]), "r", encoding="utf-8-sig") as _f:
                            _ws = [te.safe_label(_l) for _l in _f if _l.strip() and not _l.strip().startswith("#")]
                        te.set_active_words([w_ for w_ in _ws if w_])
                    with open(_set_path, "w", encoding="utf-8") as _f:
                        _f.write(_pick)
                    model, actions, bundle = load_ai()
                except Exception as _e:
                    st.warning(f"Không đổi được bộ từ: {_e}")
        col_clear.write("")
        if col_clear.button("🧹 Xóa câu"):   # xóa bộ đệm + câu đang hiện (bấm cũng chạy lại trang)
            st.session_state.vsl_word_buffer = []
            st.session_state.vsl_alt_buffer = []
            te.clear_recent()
            st.session_state.vsl_last_sentence = ""
            if live_bridge is not None:
                live_bridge.clear_current()
    _live_port = None
    if LIVE_CHAT and live_bridge is not None:
        try:
            _live_port = live_bridge.start()
        except Exception:
            _live_port = None
    if _live_port:
        col_cam, col_chat = st.columns([3, 2])
        with col_chat:
            import streamlit.components.v1 as _components_live
            _components_live.html(live_bridge.panel_html(_live_port), height=700)
    else:
        col_cam = st.container()
    with col_cam:
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        buffer_placeholder = st.empty()
        info_placeholder = st.empty()
        suggest_placeholder = st.empty()   # gợi ý "có phải bạn muốn nói..." khi AI chưa chắc
    suggest_id = 0

    drawing = mp.solutions.drawing_utils
    holistic_sol = mp.solutions.holistic

    cap = None
    _cam_src = None      # file camera_url.txt: dòng đầu là địa chỉ camera điện thoại (vd http://192.168.1.5:8080/video); không có file = webcam máy tính
    try:
        with open(os.path.join(BASE_DIR, "camera_url.txt"), "r", encoding="utf-8-sig") as _f:
            _ln = [x.strip() for x in _f.read().splitlines() if x.strip() and not x.strip().startswith("#")]
        if _ln:
            _cam_src = _ln[0]
    except Exception:
        _cam_src = None
    if _cam_src:
        cap = cv2.VideoCapture(_cam_src)
        if not cap.isOpened():
            cap.release()
            cap = None
            st.warning("Không mở được camera điện thoại. Kiểm tra điện thoại và máy tính cùng Wi-Fi, app IP Webcam đã bấm Start server, "
                       "và địa chỉ trong camera_url.txt đúng. Tạm dùng webcam máy tính.")
    if cap is None and os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)     # DirectShow mở camera nhanh và ít trễ hơn trên Windows
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(0)
    if not _cam_src:
      cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # nhẹ băng thông USB, đỡ giật trên máy yếu
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    grabber = LatestFrameGrabber(cap)                # luồng riêng luôn giữ khung hình MỚI NHẤT -> không bị trễ dồn
    last_draw = 0.0                                  # chỉ vẽ lại hình xem trước ~15 khung/giây (nhận diện vẫn chạy mọi khung)

    detector = te.SegmentDetector()   # tách từng ký hiệu: tay cử động -> tay hạ/đứng yên
    gesture = GestureButtons() if GESTURE_UI else None
    paused = False
    shoulder_ref = te.ShoulderRef()   # nhớ vị trí hai vai của luồng camera này
    last_status = None
    last_result_html = None           # chỉ vẽ lại khung kết quả khi nội dung đổi -> đỡ giật
    last_buffer_view = None
    last_translate_status = None
    no_shoulder_frames = 0
    translate_q = queue.Queue()   # luồng dịch ngầm gửi câu đã dịch về đây
    pending = 0                   # số câu đang chờ dịch

    def preview_words(buf):
        """Các từ đang chờ ghép. Khác Tiếng Việt thì KHÔNG lộ chữ Việt: chỉ hiện nhãn nếu đủ, không thì hiện số ký hiệu."""
        if lang_cfg["key"] == "vi":
            return " · ".join(buf)
        parts = [get_label(w, lang_cfg["key"]) for w in buf]
        if all(found for _text, found in parts):
            return " · ".join(text for text, _found in parts)
        return f"{len(buf)} ký hiệu"

    try:
        with te.create_holistic() as holistic:
            while cap.isOpened():
                ret, frame = grabber.read()
                if not ret:
                    break

                # Học xong -> tự nạp lại mô hình + bộ mẫu, không cần tải lại trang
                if train_state.consume_model_ready():
                    model, actions, bundle = load_ai()
                    detector.reset()

                h, w = frame.shape[:2]
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.left_hand_landmarks:
                    drawing.draw_landmarks(image, results.left_hand_landmarks, holistic_sol.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    drawing.draw_landmarks(image, results.right_hand_landmarks, holistic_sol.HAND_CONNECTIONS)

                # 126 tọa độ tay, tương đối so với trung tâm hai vai (cùng thuật toán với lúc học từ video)
                feat = extract_keypoints(results, w / h, shoulder_ref)
                present = te.keypoints_present(feat)

                # Thấy tay mà không thấy vai -> nhắc người dùng ngồi/đứng lùi ra
                if te.has_hand(results) and not present:
                    no_shoulder_frames += 1
                    cv2.putText(image, "NO SHOULDERS", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                    if no_shoulder_frames == 20:
                        info_placeholder.warning("⚠️ Thấy tay nhưng chưa thấy hai vai. Hãy ngồi/đứng lùi ra để camera thấy nửa thân trên.")
                else:
                    no_shoulder_frames = 0

                if gesture is not None:
                    try:
                        act = gesture.update(results, detector.recording, paused)
                    except Exception:
                        act = None
                    if act == "clear":
                        st.session_state.vsl_word_buffer = []
                        st.session_state.vsl_alt_buffer = []
                        te.clear_recent()
                        st.session_state.vsl_last_sentence = ""
                        if live_bridge is not None:
                            live_bridge.clear_current()
                        info_placeholder.caption("🖐️ Đã xóa câu.")
                    elif act == "replay":
                        _txt = st.session_state.vsl_last_sentence
                        if _txt:
                            threading.Thread(target=lambda t_=_txt, c_=st.session_state.get("vsl_last_tts", "vi"):
                                             tts.speak(t_, c_), daemon=True).start()
                            info_placeholder.caption("🖐️ Đang đọc lại câu.")
                    elif act == "pause":
                        paused = not paused
                        detector.reset()
                        info_placeholder.caption("🖐️ Đã tạm dừng nhận diện." if paused else "🖐️ Tiếp tục nhận diện.")
                if paused:
                    segment = None
                    detector.reset()
                else:
                    segment = detector.push(feat, present)
                if detector.recording:
                    cv2.putText(image, "REC", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    st.session_state.last_vsl_speak_time = time.time()   # đang múa = còn hoạt động, chưa hết câu

                if segment is not None:   # vừa kết thúc 1 ký hiệu -> nhận diện đúng 1 lần
                    try:
                        word, info = te.classify_segment(model, actions, bundle, segment,
                                                         tolerance=tolerance, min_conf=CONF_THRESHOLD)
                    except Exception as e:
                        word, info = None, {"reason": f"lỗi nhận diện: {e}", "candidate": None, "ratio": None}
                    log_recognition(segment, word, info)
                    if word and not train_state.is_hidden(word):
                        # KHÔNG phát loa từng từ: đẩy vào bộ đệm, chờ ghép cả câu
                        st.session_state.vsl_word_buffer.append(word)
                        st.session_state.vsl_alt_buffer.append([x for x in info.get("suggest", []) if x != word][:2])
                        te.note_recent(word)      # máy tự nghiêng về chủ đề đang nói
                        if live_bridge is not None:
                            live_bridge.push_word(word)
                        st.session_state.last_vsl_speak_time = time.time()
                        ok_msg = (f"✅ Đã nhận 1 ký hiệu · độ khớp mẫu {info['ratio']:.2f}"
                                  + (f" · LSTM {info['conf'] * 100:.0f}%" if info.get("conf") else ""))
                        info_placeholder.caption(ok_msg)
                        suggest_placeholder.empty()
                    else:
                        # Chỉ lộ tên từ tiếng Việt khi đang ở chế độ Tiếng Việt
                        cand = (f" (gần giống '{info['candidate']}', tỉ lệ {info['ratio']:.2f})"
                                if lang_cfg["key"] == "vi" and info.get("candidate") else "")
                        info_placeholder.caption(f"⚠️ Bỏ qua 1 cử chỉ: {info.get('reason') or 'từ đã bị xóa'}{cand}")
                        # Chưa chắc chắn -> đưa tối đa 3 gợi ý để người dùng bấm chọn (chọn xong AI còn học thêm)
                        sug = []
                        for w_ in (info.get("suggest", []) if SHOW_SUGGESTIONS else []):
                            if train_state.is_hidden(w_):
                                continue
                            txt, found = get_label(w_, lang_cfg["key"])
                            if lang_cfg["key"] == "vi" or found:
                                sug.append((w_, txt))
                        suggest_id += 1
                        if sug:
                            with suggest_placeholder.container():
                                st.caption("🤔 Chưa chắc chắn. Có phải bạn muốn nói:")
                                scols = st.columns(len(sug) + 1)
                                for j, (w_, txt) in enumerate(sug):
                                    scols[j].button(f"✅ {txt}", key=f"sg_{suggest_id}_{j}",
                                                    on_click=_accept_suggestion, args=(w_, segment))
                                scols[-1].button("🚫 Không phải", key=f"sg_{suggest_id}_x")
                        else:
                            suggest_placeholder.empty()

                # --- Đếm ngược: tay hạ xuống / dừng múa quá SENTENCE_TIMEOUT giây = xong câu ---
                buf = st.session_state.vsl_word_buffer
                idle = time.time() - st.session_state.last_vsl_speak_time
                if buf and not detector.recording and idle > SENTENCE_TIMEOUT:
                    words = list(buf)
                    alts = list(st.session_state.vsl_alt_buffer)
                    st.session_state.vsl_word_buffer = []
                    st.session_state.vsl_alt_buffer = []
                    te.clear_recent()
                    vi_sentence = build_vietnamese_sentence(words)   # bước 1: câu tiếng Việt hoàn chỉnh (nội bộ)
                    # bước 2 (luồng ngầm): có mạng -> Google chỉnh thành câu tự nhiên; mất mạng -> luật + gói dịch trên máy
                    threading.Thread(target=_translate_worker, args=(words, vi_sentence, lang_cfg, translate_q, tts, alts),
                                     daemon=True).start()
                    pending += 1
                    buf = st.session_state.vsl_word_buffer

                # Nhận câu đã dịch từ luồng ngầm -> hiển thị + đọc đúng 1 lần
                while True:
                    try:
                        out_text, out_tts, out_source, out_err = translate_q.get_nowait()
                    except queue.Empty:
                        break
                    pending = max(0, pending - 1)
                    st.session_state.vsl_last_sentence = out_text   # (đã được đọc ngay ở luồng dịch, không đọc lại)
                    st.session_state.vsl_last_tts = out_tts
                    if live_bridge is not None:
                        live_bridge.push_sentence(out_text)
                    if out_source == "labels":
                        info_placeholder.warning(f"⚠️ Chưa dịch tự động được nên ghép từ các nhãn đã nhập ({out_err or 'không rõ lỗi'}).")
                    elif out_source == "fallback":
                        info_placeholder.warning(f"⚠️ Chưa dịch được sang {lang_cfg['name']} nên tạm hiện tiếng Việt "
                                                 f"({out_err or 'không rõ lỗi'}). Kiểm tra gói dịch hoặc bổ sung nhãn ở bảng “Cập nhật bộ dữ liệu AI tự học”.")
                    elif out_source == "polish":
                        info_placeholder.caption("🌐 Google đã làm câu tự nhiên, lịch sự hơn.")
                    elif out_source == "ai":
                        info_placeholder.caption("🤖 AI của Google đã tự sắp xếp thành câu tự nhiên.")
                    elif out_source == "online":
                        info_placeholder.caption("🌐 Đã dịch cả câu bằng Google.")
                    elif out_source == "argos":
                        info_placeholder.caption("🌐 Đã dịch cả câu bằng gói dịch trên máy.")


                now_draw = time.time()
                if now_draw - last_draw >= 0.066:        # ~15 khung/giây là đủ mượt; mã hóa + gửi hình mỗi khung rất tốn máy
                    last_draw = now_draw
                    if gesture is not None:
                        try:
                            image = gesture.draw(image, paused)
                        except Exception:
                            pass
                    _, buffer_jpg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    img_base64 = base64.b64encode(buffer_jpg).decode('utf-8')
                    frame_placeholder.markdown(
                        f'<div class="cam-wrap"><div class="neon-frame">'
                        f'<img src="data:image/jpeg;base64,{img_base64}" /></div></div>',
                        unsafe_allow_html=True)

                if buf:
                    new_html = ('<div class="result-idle">✍️ Đang ghép câu: '
                                f'<b>{html.escape(preview_words(buf))}</b></div>')
                elif st.session_state.vsl_last_sentence:
                    new_html = ('<div class="result-big"><div class="lab">🗣️ Câu hoàn chỉnh (VSL)</div>'
                                f'<div class="txt">{html.escape(st.session_state.vsl_last_sentence)}</div></div>')
                elif not bundle["actions"]:
                    new_html = ('<div class="result-idle">🧠 AI chưa biết từ nào. Mở bảng “Cập nhật bộ dữ liệu AI tự học” '
                                'và dạy ít nhất 1 từ.</div>')
                else:
                    new_html = ('<div class="result-idle">⌛ Hãy đưa bàn tay ra trước camera, làm ký hiệu '
                                'rồi hạ tay xuống...</div>')
                if new_html != last_result_html:
                    last_result_html = new_html
                    result_placeholder.markdown(new_html, unsafe_allow_html=True)

                # Dòng đếm ngược (chỉ vẽ lại khi đổi, mỗi 0,2 giây)
                if buf:
                    remaining = None if detector.recording else max(0.0, SENTENCE_TIMEOUT - idle)
                    buffer_view = (len(buf), None if remaining is None else int(remaining * 5))
                else:
                    remaining, buffer_view = None, None
                if buffer_view != last_buffer_view:
                    last_buffer_view = buffer_view
                    if buffer_view is None:
                        buffer_placeholder.empty()
                    elif remaining is None:
                        buffer_placeholder.caption(f"🖐️ Đang nhận ký hiệu... ({len(buf)} từ trong câu)")
                    else:
                        buffer_placeholder.caption(f"⏳ Dừng múa… sẽ ghép và đọc cả câu sau {remaining:.1f} giây "
                                                   f"({len(buf)} từ trong câu)")

                # Tiến độ học ngầm (chỉ vẽ lại khi có thay đổi)
                running, msg, prog, err = train_state.snapshot()
                status = (running, msg, round(prog, 2), err)
                if status != last_status:
                    last_status = status
                    if err:
                        status_placeholder.markdown(progress_html("err", err, 0), unsafe_allow_html=True)
                    elif running:
                        status_placeholder.markdown(progress_html("run", msg, prog), unsafe_allow_html=True)
                    elif msg:
                        status_placeholder.markdown(progress_html("ok", msg, 1.0), unsafe_allow_html=True)

                time.sleep(0.01)
    finally:
        # Streamlit ngắt script mỗi khi bấm nút/đổi menu -> luôn nhả camera
        grabber.stop()
        cap.release()


LIVE_CHAT = True     # True: khung "Trò chuyện trực tiếp" (nói + ký hiệu hai chiều) nằm cạnh camera. False: bố cục cũ.
GESTURE_UI = True    # True: có nút ảo trên khung camera, chạm bằng ngón trỏ giơ lên (không cần chuột). False: tắt hẳn.
GESTURE_DWELL = 1.0  # giữ ngón trỏ trong nút bao lâu (giây) thì nút được bấm
GESTURE_FONTS = [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


class GestureButtons:
    """Nút ảo vẽ trên khung camera. Người dùng giơ NGÓN TRỎ (các ngón khác co lại) vào nút và giữ yên khoảng 1 giây để bấm.
    Nút nằm sát hai mép khung (nơi hiếm khi ra ký hiệu) và chỉ đếm khi KHÔNG đang ghi một ký hiệu -> không bấm nhầm lúc ký."""

    def __init__(self):
        self.buttons = [("clear", "Xóa câu", (0.01, 0.30, 0.19, 0.46)),
                        ("replay", "Đọc lại", (0.01, 0.52, 0.19, 0.68)),
                        ("pause", "Tạm dừng", (0.81, 0.30, 0.99, 0.46))]
        self.hover, self.since, self.cool_until = None, 0.0, 0.0
        self.flash = None
        self._font = {}

    @staticmethod
    def pointer(results):
        """Vị trí đầu ngón trỏ (x, y chuẩn hóa) nếu một tay đang chỉ ngón trỏ lên; không thì None."""
        for hand in (results.right_hand_landmarks, results.left_hand_landmarks):
            if not hand:
                continue
            lm = hand.landmark
            idx_up = lm[8].y < lm[6].y - 0.02
            curled = sum(lm[t].y > lm[p].y for t, p in ((12, 10), (16, 14), (20, 18)))
            if idx_up and curled >= 3:
                return lm[8].x, lm[8].y
        return None

    def update(self, results, busy, paused):
        """Trả về tên hành động vừa được bấm ("clear"/"replay"/"pause") hoặc None."""
        now = time.time()
        pt = None if busy else self.pointer(results)
        hit = None
        if pt is not None:
            for name, _label, (x0, y0, x1, y1) in self.buttons:
                if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1:
                    hit = name
        if hit is None or now < self.cool_until:
            self.hover = None if hit is None else self.hover
            if hit is None:
                self.since = 0.0
            return None
        if hit != self.hover:
            self.hover, self.since = hit, now
            return None
        if now - self.since >= GESTURE_DWELL:
            self.hover, self.since, self.cool_until = None, 0.0, now + 1.5
            self.flash = (hit, now + 0.8)
            return hit
        return None

    def _f(self, size):
        if size not in self._font:
            from PIL import ImageFont
            f = None
            for pth in GESTURE_FONTS:
                if os.path.exists(pth):
                    try:
                        f = ImageFont.truetype(pth, size)
                        break
                    except Exception:
                        pass
            self._font[size] = f or ImageFont.load_default()
        return self._font[size]

    def draw(self, image, paused):
        """Vẽ các nút (chỉ gọi khi cập nhật hình xem trước)."""
        from PIL import Image, ImageDraw
        h, w = image.shape[:2]
        now = time.time()
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        f = self._f(max(14, int(h * 0.04)))
        for name, label, (x0, y0, x1, y1) in self.buttons:
            if name == "pause" and paused:
                label = "Tiếp tục"
            box = [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)]
            active = self.hover == name
            flash = self.flash and self.flash[0] == name and now < self.flash[1]
            fill = (40, 200, 120, 200) if flash else ((255, 190, 40, 170) if active else (20, 20, 30, 140))
            d.rounded_rectangle(box, radius=10, fill=fill, outline=(255, 255, 255, 200), width=2)
            if active and self.since:
                frac = min(1.0, (now - self.since) / GESTURE_DWELL)
                d.rectangle([box[0] + 4, box[3] - 8, box[0] + 4 + int((box[2] - box[0] - 8) * frac), box[3] - 4],
                            fill=(255, 255, 255, 230))
            tw = d.textlength(label, font=f)
            d.text((box[0] + (box[2] - box[0] - tw) / 2, box[1] + (box[3] - box[1]) / 2 - f.size * 0.6),
                   label, font=f, fill=(255, 255, 255, 255))
        if paused:
            msg = "ĐANG TẠM DỪNG"
            f2 = self._f(max(18, int(h * 0.06)))
            tw = d.textlength(msg, font=f2)
            d.rectangle([int(w / 2 - tw / 2) - 10, 8, int(w / 2 + tw / 2) + 10, 12 + f2.size + 8], fill=(200, 40, 40, 210))
            d.text((int(w / 2 - tw / 2), 12), msg, font=f2, fill=(255, 255, 255, 255))
        return cv2.cvtColor(np.array(Image.alpha_composite(pil, over).convert("RGB")), cv2.COLOR_RGB2BGR)


class LatestFrameGrabber:
    """Đọc camera ở luồng riêng và chỉ giữ khung hình mới nhất. Nhờ vậy nếu máy xử lý chậm thì bỏ khung cũ,
    hình luôn là hiện tại (không bị trễ dồn)."""
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = True
        self.seq = 0
        self.taken = 0
        self.alive = True
        self.cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            ret, fr = self.cap.read()
            with self.cv:
                self.ok = bool(ret)
                if ret:
                    self.frame = fr
                    self.seq += 1
                self.cv.notify_all()
            if not ret:
                time.sleep(0.05)

    def read(self, timeout=2.0):
        with self.cv:
            end = time.time() + timeout
            while self.seq == self.taken and self.alive:
                left = end - time.time()
                if left <= 0:
                    return False, None
                self.cv.wait(left)
            self.taken = self.seq
            return self.ok and self.frame is not None, self.frame

    def stop(self):
        self.alive = False


# ------------------------------ Trang giới thiệu ------------------------------ #
def run_intro():
    inject_css()
    st.markdown("""
    <div class="glass hero">
      <h1>Phần mềm dịch ngôn ngữ ký hiệu thời gian thực</h1>
      <p>Một cầu nối để người khiếm thính và người nghe nói chuyện trực tiếp với nhau, không cần phiên dịch viên.
         Ký hiệu tay được AI nhận diện qua camera, ghép thành câu hoàn chỉnh và đọc thành tiếng; lời nói được chuyển thành chữ lớn, dễ đọc.</p>
      <div class="badge-row">
        <span class="badge">Hai chiều</span>
        <span class="badge">Thời gian thực</span>
        <span class="badge">Đọc to không cần Internet</span>
        <span class="badge">5 ngôn ngữ đầu ra</span>
      </div>
    </div>

    <div class="grid">
      <div class="glass card"><div class="icon">🤝</div>
        <h3>Mục tiêu nhân văn</h3>
        <p>Giúp người khiếm thính tự tin giao tiếp ở trường học, bệnh viện, nơi công cộng, những nơi hiếm khi có người biết ngôn ngữ ký hiệu.</p></div>
      <div class="glass card g"><div class="icon">🖐️</div>
        <h3>Nhận diện ký hiệu bằng AI</h3>
        <p>Camera theo dõi từng khớp ngón tay so với hai vai, nên ngồi hay đứng đều nhận diện được; nhiều ký hiệu liên tiếp được ghép thành một câu.</p></div>
      <div class="glass card v"><div class="icon">🎙️</div>
        <h3>Giọng nói thành chữ lớn</h3>
        <p>Người nghe nói vào micro, lời nói hiện thành văn bản cỡ lớn để người khiếm thính đọc ngay.</p></div>
      <div class="glass card"><div class="icon">🧠</div>
        <h3>AI tự học từ video</h3>
        <p>Muốn thêm một ký hiệu mới, chỉ cần tải lên video ngắn. AI học ngầm trong khi camera vẫn chạy bình thường.</p></div>
      <div class="glass card g"><div class="icon">🌐</div>
        <h3>Đa ngôn ngữ đầu ra</h3>
        <p>Kết quả hiển thị và đọc to bằng Tiếng Việt, Trung, Hàn, Nhật hoặc Anh theo nhãn do bạn nhập.</p></div>
      <div class="glass card v"><div class="icon">🔒</div>
        <h3>Chạy ngay trên máy</h3>
        <p>Nhận diện, ghép câu và đọc to hoạt động ngoại tuyến, dữ liệu camera không phải gửi đi đâu.</p></div>
    </div>

    <div class="glass ready"><div class="dot"></div>
      <span>Hệ thống đã khởi động và sẵn sàng. Chọn chế độ dịch ở thanh menu bên trái để bắt đầu.</span></div>
    """, unsafe_allow_html=True)


if not HEADLESS:
    # ------------------------------ Điều hướng ------------------------------ #
    st.sidebar.title("")
    st.sidebar.subheader("Phần mềm dịch đa ngôn ngữ")

    app_mode = st.sidebar.selectbox(
        "Chọn ngôn ngữ dịch:",
        ["Giới thiệu dự án", "Ngôn ngữ ký hiệu Mỹ (ASL)", "Ngôn ngữ ký hiệu Việt Nam (VSL)"]
    )

    if app_mode == "Giới thiệu dự án":
        run_intro()

    elif app_mode == "Ngôn ngữ ký hiệu Mỹ (ASL)":
        # app.py gốc tự gọi st.set_page_config (lệnh này chỉ được gọi 1 lần/lượt chạy và launcher đã gọi rồi),
        # nên tạm vô hiệu nó trong lúc chạy app.py; code của tác giả vẫn giữ nguyên 100%.
        _real_set_page_config = st.set_page_config
        st.set_page_config = lambda *args, **kwargs: None
        try:
            with open("app.py", "r", encoding="utf-8") as f:
                exec(f.read(), globals())
        finally:
            st.set_page_config = _real_set_page_config

    elif app_mode == "Ngôn ngữ ký hiệu Việt Nam (VSL)":
        run_vsl_mode()
