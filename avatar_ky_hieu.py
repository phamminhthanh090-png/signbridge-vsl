# -*- coding: utf-8 -*-
"""AVATAR KÝ HIỆU: người bình thường nói/gõ chữ -> máy vẽ lại các ký hiệu để người khiếm thính xem (chiều ngược lại).
Dùng ngay các mẫu đã có trong dataset_vsl (tọa độ tay của chính video bộ giáo dục), KHÔNG cần video, KHÔNG cần Internet.
Gọi: html, found, missing = build_player(text)  ->  hiển thị html bằng streamlit.components.v1.html."""
import json
import os
import re
import unicodedata

import numpy as np

import train_engine as te

SKIP = {"xin", "là", "và", "của", "thì", "đã", "đang", "sẽ", "rất", "các", "những", "một", "cái", "ạ", "nhé", "nha",
        "à", "ơi", "dạ", "vâng", "ừ", "hãy", "được", "thế", "này", "đó", "ấy", "với", "cho", "để", "mà", "nữa"}
MAX_WORDS = 12          # tối đa số ký hiệu phát một lượt
MAX_FRAMES = 90         # mỗi ký hiệu tối đa số khung (cắt bớt nếu dài bất thường)
_cache = {"key": None, "map": {}}


def _clean(w):
    c = re.sub(r"\s*\([^)]*\)", "", str(w)).split("/")[0].strip().rstrip("?").strip()
    return unicodedata.normalize("NFC", c).lower()


def _label_map():
    acts = te.list_actions()
    key = (len(acts), te.DATA_DIR)
    if _cache["key"] != key:
        m = {}
        for a in acts:
            m.setdefault(_clean(a), a)
        _cache["key"], _cache["map"] = key, m
    return _cache["map"]


def tokenize(text):
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return t.split()


def match_words(text):
    """Chữ -> (danh sách nhãn có ký hiệu theo thứ tự nói, danh sách từ không có ký hiệu). Ưu tiên cụm dài nhất."""
    m = _label_map()
    toks = tokenize(text)
    found, missing, i = [], [], 0
    while i < len(toks):
        hit = None
        for n in range(min(4, len(toks) - i), 0, -1):
            cand = " ".join(toks[i:i + n])
            if cand in m:
                hit = (m[cand], n)
                break
        if hit:
            found.append(hit[0])
            i += hit[1]
        else:
            if toks[i] not in SKIP:
                missing.append(toks[i])
            i += 1
    return found[:MAX_WORDS], missing


def _pick_file(word):
    d = os.path.join(te.DATA_DIR, word)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    plain = [f for f in fs if not f.startswith("cam_") and not f.startswith("xau_")]
    return (plain or fs)[0] if fs else None


def _frames(word):
    f = _pick_file(word)
    if f is None:
        return None
    raw = np.load(os.path.join(te.DATA_DIR, word, f))
    if raw.ndim != 2 or raw.shape[1] != te.NUM_FEATURES or len(raw) < 3:
        return None
    seg = te.trim_rest(te.fill_gaps(raw))
    if len(seg) > MAX_FRAMES:
        seg = seg[np.linspace(0, len(seg) - 1, MAX_FRAMES).astype(int)]
    x = np.asarray(seg, np.float32).reshape(len(seg), 2, 21, 3)[..., :2]
    pres = np.abs(np.asarray(seg).reshape(len(seg), 2, 63)).sum(axis=2) > 0
    k = np.ones(3, np.float32) / 3                       # làm mượt nhẹ theo thời gian
    for h in range(2):
        for j in range(21):
            for c in range(2):
                v = x[:, h, j, c]
                x[:, h, j, c] = np.convolve(np.pad(v, 1, mode="edge"), k, mode="valid")
    out = []
    for t in range(len(seg)):
        fr = []
        for h in range(2):
            fr.append(np.round(x[t, h].reshape(-1), 3).tolist() if pres[t, h] else None)
        out.append(fr)
    return out


def build_clips(text):
    """Trả về (clips, các từ có ký hiệu, các từ không có). clips = [{"word":..., "frames":[...]}] (dùng cho trình phát JS)."""
    found, missing = match_words(text)
    clips = []
    for w in found:
        try:
            fr = _frames(w)
        except Exception:
            fr = None
        if fr:
            clips.append({"word": w, "frames": fr})
    return clips, [c["word"] for c in clips], missing


def build_player(text):
    """Trả về (html, các từ có ký hiệu, các từ không có). html = None nếu không có ký hiệu nào."""
    clips, found, missing = build_clips(text)
    if not clips:
        return None, [], missing
    html = _HTML.replace("__DATA__", json.dumps(clips, ensure_ascii=False, separators=(",", ":")))
    return html, found, missing


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#10131c;--fg:#f2f4f8;--mut:#9aa3b5;--acc:#ffbe28}
@media (prefers-color-scheme: light){:root{--bg:#eef1f7;--fg:#161a24;--mut:#5b6478;--acc:#d98a00}}
html,body{margin:0;background:transparent;font-family:Segoe UI,Arial,sans-serif;color:var(--fg)}
.wrap{max-width:560px;margin:0 auto;padding:8px 12px 12px}
canvas{width:100%;border-radius:14px;background:var(--bg);display:block}
.cap{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 6px;justify-content:center}
.chip{padding:4px 10px;border-radius:999px;background:rgba(128,128,128,.18);font-size:15px}
.chip.on{background:var(--acc);color:#111;font-weight:600}
.bar{display:flex;gap:8px;justify-content:center;align-items:center;font-size:14px;color:var(--mut)}
button,select{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid rgba(128,128,128,.4);background:transparent;color:var(--fg);cursor:pointer}
</style></head><body><div class="wrap">
<canvas id="c" width="560" height="460"></canvas>
<div class="cap" id="cap"></div>
<div class="bar"><button id="re">&#8635; Xem lại</button><label>Tốc độ <select id="sp"><option value="0.6">Chậm</option><option value="1" selected>Vừa</option><option value="1.5">Nhanh</option></select></label></div>
</div><script>
const CL=__DATA__;
const cv=document.getElementById('c'),cx=cv.getContext('2d'),cap=document.getElementById('cap');
const E=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const COL=['#4cc9f0','#ff7a90'];
CL.forEach((c,i)=>{const s=document.createElement('span');s.className='chip';s.id='ch'+i;s.textContent=c.word.replace(/\s*\([^)]*\)/,'');cap.appendChild(s)});
const S=140,OX=280,OY=225,FPS=24;
function P(x,y){return [OX+x*S,OY+y*S]}
function body(){
  const css=getComputedStyle(document.documentElement);
  cx.fillStyle='rgba(128,140,170,.28)';
  cx.beginPath();cx.moveTo(...P(-.62,0));cx.quadraticCurveTo(...P(-.62,-.22),...P(-.3,-.25));cx.lineTo(...P(.3,-.25));cx.quadraticCurveTo(...P(.62,-.22),...P(.62,0));cx.lineTo(...P(.7,2.2));cx.lineTo(...P(-.7,2.2));cx.closePath();cx.fill();
  cx.beginPath();cx.arc(...P(0,-.88),.38*S,0,7);cx.fill();
  cx.fillRect(...P(-.1,-.55),.2*S,.32*S);
}
function arm(sh,w,col){
  const mx=(sh[0]+w[0])/2,my=(sh[1]+w[1])/2,d=Math.hypot(w[0]-sh[0],w[1]-sh[1]);
  cx.strokeStyle='rgba(150,160,185,.7)';cx.lineWidth=14;cx.lineCap='round';
  cx.beginPath();cx.moveTo(...sh);cx.quadraticCurveTo(mx,my+d*.35,...w);cx.stroke();
}
function hand(pts,col){
  cx.strokeStyle=col;cx.lineWidth=4;cx.lineCap='round';
  E.forEach(([a,b])=>{cx.beginPath();cx.moveTo(...P(pts[2*a],pts[2*a+1]));cx.lineTo(...P(pts[2*b],pts[2*b+1]));cx.stroke()});
  cx.fillStyle='#fff';
  for(let j=0;j<21;j++){const p=P(pts[2*j],pts[2*j+1]);cx.beginPath();cx.arc(p[0],p[1],j%4==0?3.6:2.4,0,7);cx.fill()}
}
function draw(fr){
  cx.clearRect(0,0,cv.width,cv.height);body();
  if(!fr)return;
  const sh=[P(-.5,0),P(.5,0)];
  fr.forEach((h,i)=>{
    if(!h)return;const w=P(h[0],h[1]);
    const s=Math.abs(w[0]-sh[0][0])<Math.abs(w[0]-sh[1][0])?sh[0]:sh[1];
    arm(s,w,COL[i]);
  });
  fr.forEach((h,i)=>{if(h)hand(h,COL[i])});
}
let timer=null,tok=0;
function play(){
  clearTimeout(timer);const my=++tok;let ci=0,fi=0,pause=0;
  const sp=()=>parseFloat(document.getElementById('sp').value);
  function mark(){CL.forEach((_,i)=>document.getElementById('ch'+i).classList.toggle('on',i===ci))}
  function step(){
    if(my!==tok)return;
    if(pause>0){pause--;draw(null);timer=setTimeout(step,1000/FPS);return}
    if(ci>=CL.length){CL.forEach((_,i)=>document.getElementById('ch'+i).classList.remove('on'));draw(null);return}
    mark();const fr=CL[ci].frames;
    draw(fr[fi]);fi++;
    if(fi>=fr.length){ci++;fi=0;pause=Math.round(10/sp())}
    timer=setTimeout(step,1000/(FPS*sp()));
  }
  step();
}
document.getElementById('re').onclick=play;document.getElementById('sp').onchange=play;
play();
</script></body></html>"""
