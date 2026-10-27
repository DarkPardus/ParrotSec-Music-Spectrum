#!/usr/bin/env python3
"""
Parrot Music Spectrum v1.0  (plasmadock)
Masaüstünde taşınabilir "araç çubuğu" içinde canlı görselleştiriciler
KDE Plasma / Parrot / Debian  •  PyQt5 + numpy

Modlar (araç çubuğundaki düğmeler ya da çift tık ile geçilir):
  Çubuk  - cava tarzı spektrum (normal / ayna)
  Daire  - dairesel spektrum: pencere otomatik KARE kutuya geçer, ortada Parrot logosu bas vuruşuyla atar
  Şelale - spektrogram: zaman soldan sağa akar, renk şiddeti gösterir
  Ağ     - canlı ağ çubukları: indirme/yükleme, paket, TCP/UDP, HTTPS/HTTP/SSH/DNS bağlantıları
Renkler (başlıktaki renk düğmesi ya da ⚙ → Renk): Linux dağıtımı adıyla seçilir
  Parrot Sec. = Turkuaz   Kali Linux = Mavi   openSUSE = Yeşil
  Debian = Kırmızı   Arch Linux = Gök Mavisi   Ubuntu = Turuncu
  (+ Albüm kapağı: çalan şarkının kapağından baskın renk, MPRIS / playerctl; Plasma vurgusu; Gökkuşağı)

Kurulum : sudo apt install python3-pyqt5 python3-numpy pulseaudio-utils playerctl
Çalıştır: python3 plasmadock.py
          python3 plasmadock.py --mode circle --color kali
Otomatik başlat: python3 plasmadock.py --install-autostart

Kullanım:
  Sol tık + sürükle (herhangi bir yerinden) → istediğin yere taşı (sağa, sola, her yere)
  Sağ alt köşedeki tutamaç → boyutlandır      Çift tık → sonraki mod
  Sağ tık ya da ⚙ → ayarlar (renk, stil, konum önayarları, boyut, katman…)
Konum, boyut ve ayarlar ~/.config/plasmadock/config.json içinde saklanır.
"""
import argparse
import colorsys
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen

# Wayland'de şeffaf/üstte pencere için XWayland (xcb) daha güvenilir
if os.environ.get("XDG_SESSION_TYPE") == "wayland" and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
from PyQt5.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QLinearGradient, QPainter,
                         QPen, QPixmap, QPolygonF, QRadialGradient)
from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QButtonGroup, QHBoxLayout,
                             QLabel, QMenu, QSizeGrip, QSizePolicy, QSystemTrayIcon,
                             QToolButton, QVBoxLayout, QWidget)

RATE = 44100
FFT_N = 4096
DB_RANGE = 60.0
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "plasmadock"
CONFIG_FILE = CONFIG_DIR / "config.json"

MODES = [("bars", "Çubuk"), ("circle", "Daire"), ("waterfall", "Şelale"), ("net", "Ağ")]
AUDIO_MODES = ("bars", "circle", "waterfall")
APP_NAME = "Parrot Music Spectrum"
APP_VERSION = "v1.0"
# Linux dağıtımı renkleri (anahtar, dağıtım, renk adı, hex). Hex kodları resmi kaynaklardan doğrulandı:
#  Parrot  : parrotsec-logo.svg dairesi            Kali   : Kali'nin kendi tema paleti mavisi
#  openSUSE: resmi "openSUSE green" #73BA25         Debian : "DebianRed" (X11 rgb.txt) #D70751
#  Arch    : logo mavisi #1793D1                     Ubuntu : Ubuntu Orange (ekran) #E95420
DISTROS = [
    ("parrot", "Parrot Sec.", "Turkuaz", "#15E0ED"),
    ("kali", "Kali Linux", "Mavi", "#367BF0"),
    ("opensuse", "openSUSE", "Yeşil", "#73BA25"),
    ("debian", "Debian", "Kırmızı", "#D70751"),
    ("arch", "Arch Linux", "Gök Mavisi", "#1793D1"),
    ("ubuntu", "Ubuntu", "Turuncu", "#E95420"),
]
DISTRO_RGB = {k: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for k, _n, _c, h in DISTROS}
COLOR_CHOICES = ([(k, f"{n}  —  {cn}") for k, n, cn, _h in DISTROS] +
                 [("album", "Albüm kapağı (otomatik)"), ("accent", "Plasma vurgu rengi"),
                  ("rainbow", "Gökkuşağı")])
SHORT_NAMES = {"parrot": "Parrot", "kali": "Kali", "opensuse": "openSUSE", "debian": "Debian",
               "arch": "Arch", "ubuntu": "Ubuntu", "album": "Albüm", "accent": "Plasma",
               "rainbow": "Gökkuşağı"}
# Pencere boyutları: daire modu KARE kutu ister, diğer modlar geniş şerit
DEFAULT_SIZE = {"circle": (380, 440), "wide": (720, 230)}
DEFAULTS = {
    "mode": "bars", "style": "bars", "color": "parrot", "caps": True, "layer": "above",
    "toolbar_hover": False, "dim": True, "panel_alpha": 190, "bars": 48,
    "geom": None, "logo": "",
}


# --- PURE START -------------------------------------------------------------
def mix(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def make_bands(n, rate, fft_n, lo, hi):
    hi = min(hi, rate / 2 - 200)
    edges = np.geomspace(lo, hi, n + 1)
    binw = rate / fft_n
    lo_i = np.maximum(np.floor(edges[:-1] / binw).astype(int), 1)
    hi_i = np.maximum(np.ceil(edges[1:] / binw).astype(int), lo_i + 1)
    hi_i = np.minimum(hi_i, fft_n // 2 + 1)
    lo_i = np.minimum(lo_i, hi_i - 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return lo_i, hi_i, centers


class Gravity:
    """Hızlı yükselme + yerçekimiyle düşme + tepe işaretçileri."""

    def __init__(self, n, fall=4.0):
        self.n, self.fall = n, fall
        self.cur = np.zeros(n)
        self.vel = np.zeros(n)
        self.peak = np.zeros(n)
        self.hold = np.zeros(n)
        self.pvel = np.zeros(n)

    def step(self, dt, v):
        rise = v > self.cur
        k = 1.0 - math.exp(-dt / 0.015)
        self.cur[rise] += (v[rise] - self.cur[rise]) * k
        self.vel[rise] = 0.0
        fall = ~rise
        self.vel[fall] += self.fall * dt
        self.cur[fall] = np.maximum(v[fall], self.cur[fall] - self.vel[fall] * dt)
        self.vel[fall & (self.cur <= v + 1e-6)] = 0.0

        up = self.cur >= self.peak
        self.peak[up] = self.cur[up]
        self.hold[up] = 0.45
        self.hold[~up] -= dt
        drop = (~up) & (self.hold <= 0)
        self.pvel[drop] += 2.5 * dt
        self.pvel[~drop] = 0.0
        self.peak[drop] = np.maximum(self.cur[drop], self.peak[drop] - self.pvel[drop] * dt)

    def reset(self):
        for a in (self.cur, self.vel, self.peak, self.hold, self.pvel):
            a[:] = 0.0


class Analyzer:
    """Örneklerden çubuk yüksekliği (0..1): FFT → log bantlar → eğim → AGC → yerçekimi."""

    def __init__(self, n, lo=40, hi=16000, tilt=3.0, sens=0.0, fall=4.0, agc=True):
        self.n = n
        self.lo_i, self.hi_i, centers = make_bands(n, RATE, FFT_N, lo, hi)
        self.tilt_db = tilt * np.log2(centers / centers[0])
        self.sens, self.agc = sens, agc
        self.win = np.hanning(FFT_N).astype(np.float32)
        self.g = Gravity(n, fall)
        self.agc_peak = 0.5

    @property
    def cur(self):
        return self.g.cur

    @property
    def peak(self):
        return self.g.peak

    def update(self, dt, samples):
        spec = np.abs(np.fft.rfft(samples * self.win)) / (FFT_N / 4.0)
        vals = np.array([spec[l:h].max() for l, h in zip(self.lo_i, self.hi_i)])
        db = 20.0 * np.log10(vals + 1e-9) + self.tilt_db + self.sens
        v = np.clip((db + DB_RANGE) / DB_RANGE, 0.0, 1.0)
        v = np.convolve(np.pad(v, 1, mode="edge"), [0.2, 0.6, 0.2], mode="valid")
        if self.agc:
            self.agc_peak = max(self.agc_peak * (1.0 - 0.3 * dt), float(v.max()))
            v = np.clip(v * (0.95 / max(self.agc_peak, 0.3)), 0.0, 1.0)
        self.g.step(dt, v)

    def decay(self, dt):
        self.g.step(dt, np.zeros(self.n))


class NetStats:
    """Yetki gerektirmeyen ağ ölçümleri: /proc/net/{dev,tcp,tcp6,udp,udp6}."""
    LABELS = ["İndirme", "Yükleme", "Paket↓", "Paket↑", "TCP", "Dinleyen",
              "UDP", "HTTPS", "HTTP", "SSH", "DNS", "Diğer"]
    FLOORS = np.array([50e3, 50e3, 200, 200, 5, 5, 5, 5, 5, 5, 5, 5], dtype=float)

    def __init__(self, fall=4.0):
        n = len(self.LABELS)
        self.g = Gravity(n, fall)
        self.raw = np.zeros(n)
        self.target = np.zeros(n)
        self.ref = self.FLOORS.copy()
        self.prev, self.prev_t = None, None

    @staticmethod
    def _read(path):
        try:
            with open(path) as f:
                return f.read().splitlines()
        except OSError:
            return []

    @staticmethod
    def _port(field):
        try:
            return int(field.rsplit(":", 1)[1], 16)
        except (IndexError, ValueError):
            return -1

    def sample(self):
        now = time.monotonic()
        rxb = txb = rxp = txp = 0
        for line in self._read("/proc/net/dev")[2:]:
            if ":" not in line:
                continue
            name, data = line.split(":", 1)
            if name.strip() == "lo":
                continue
            f = data.split()
            if len(f) >= 10:
                rxb += int(f[0]); rxp += int(f[1]); txb += int(f[8]); txp += int(f[9])
        cur = (rxb, txb, rxp, txp)
        rates = np.zeros(4)
        if self.prev is not None and now > self.prev_t:
            rates = np.maximum(np.array(cur, float) - np.array(self.prev, float), 0) / (now - self.prev_t)
        self.prev, self.prev_t = cur, now

        est = listen = 0
        cls = {443: 0, 80: 0, 22: 0, 53: 0, "o": 0}
        for path in ("/proc/net/tcp", "/proc/net/tcp6"):
            for line in self._read(path)[1:]:
                f = line.split()
                if len(f) < 4:
                    continue
                if f[3] == "0A":
                    listen += 1
                elif f[3] == "01":
                    est += 1
                    port = self._port(f[2])
                    cls[port if port in cls else "o"] += 1
        udp = 0
        for path in ("/proc/net/udp", "/proc/net/udp6"):
            for line in self._read(path)[1:]:
                f = line.split()
                if len(f) < 4:
                    continue
                udp += 1
                if self._port(f[2]) == 53:
                    cls[53] += 1
        self.raw = np.array([*rates, est, listen, udp, cls[443], cls[80], cls[22], cls[53], cls["o"]], float)
        self.ref = np.maximum(np.maximum(self.ref * 0.97, self.raw), self.FLOORS)
        self.target = np.sqrt(np.clip(self.raw / self.ref, 0.0, 1.0))

    def step(self, dt):
        self.g.step(dt, self.target)

    def text(self, i):
        v = self.raw[i]
        if i < 2:
            for unit, div in (("GB/s", 1e9), ("MB/s", 1e6), ("KB/s", 1e3)):
                if v >= div:
                    return f"{v / div:.1f} {unit}"
            return f"{v:.0f} B/s"
        if i < 4:
            return f"{v:.0f}/s"
        return f"{int(v)}"


def dominant_from_rgb(a):
    """a: (N,3) float 0..1 → baskın canlı renk (r,g,b 0..255); renksiz kapakta açık gri."""
    mx, mn = a.max(1), a.min(1)
    d = mx - mn
    w = d * mx + 1e-3
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    h = np.zeros(len(a), dtype=np.float32)
    m = d > 1e-6
    rc = (mx == r) & m
    gc = (mx == g) & m & ~rc
    bc = m & ~rc & ~gc
    h[rc] = ((g - b)[rc] / d[rc]) % 6
    h[gc] = (b - r)[gc] / d[gc] + 2
    h[bc] = (r - g)[bc] / d[bc] + 4
    bins = (h / 6.0 * 12).astype(int) % 12
    score = np.bincount(bins, weights=w, minlength=12)
    if score.max() < 5:
        return (205.0, 205.0, 205.0)
    sel = bins == score.argmax()
    col = (a[sel] * w[sel, None]).sum(0) / w[sel].sum()
    hh, ss, vv = colorsys.rgb_to_hsv(*[float(x) for x in col])
    vv = max(vv, 0.85)
    if ss > 0.12:
        ss = max(ss, 0.5)
    return tuple(c * 255.0 for c in colorsys.hsv_to_rgb(hh, ss, vv))


def build_lut(low, high, rainbow=False):
    """Şelale için 256 girişli ARGB32 renk tablosu."""
    t = np.linspace(0.0, 1.0, 256)
    if rainbow:
        rgb = np.array([colorsys.hsv_to_rgb(0.66 * (1 - x), 0.9, 1.0) for x in t]) * 255.0
    else:
        lo, hi = np.array(low, float), np.array(high, float)
        rgb = lo[None, :] * (1 - t[:, None]) + hi[None, :] * t[:, None]
        wm = (np.clip((t - 0.8) / 0.2, 0, 1) * 0.6)[:, None]
        rgb = rgb * (1 - wm) + 255.0 * wm
    alpha = np.clip(30 + 225 * t ** 0.7, 0, 255)
    a = alpha.astype(np.uint32)
    r, g, b = (np.clip(rgb[:, i], 0, 255).astype(np.uint32) for i in range(3))
    return (a << 24) | (r << 16) | (g << 8) | b


def palette_from_base(rgb):
    """Marka rengi → (koyu alt, parlak üst) çubuk gradyanı."""
    return tuple(x * 0.42 for x in rgb), mix(rgb, (255, 255, 255), 0.22)


FIXED = {k: palette_from_base(v) for k, v in DISTRO_RGB.items()}
LEGACY_COLORS = {"green": "opensuse", "fire": "ubuntu", "ice": "kali"}  # eski config'ler için


def mode_category(mode):
    """Daire = kare kutu düzeni; diğerleri geniş şerit düzeni."""
    return "circle" if mode == "circle" else "wide"


def clamp_rect(x, y, w, h, ax, ay, aw, ah):
    """(x,y,w,h) dikdörtgenini (ax,ay,aw,ah) alanı içine sığdır."""
    w, h = min(w, aw), min(h, ah)
    return (min(max(x, ax), ax + aw - w), min(max(y, ay), ay + ah - h), w, h)
# --- PURE END ---------------------------------------------------------------


def dominant_color(img):
    img = img.scaled(40, 40, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    img = img.convertToFormat(QImage.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    a = np.frombuffer(ptr, np.uint8).reshape(-1, 4)[:, 2::-1].astype(np.float32) / 255.0
    return dominant_from_rgb(a)


# ------------------------------------------------------------------ Parrot logosu
# parrotsec-logo.svg'den birebir (QtSvg gerekmez; 175x175 tasarım alanı)
_LOGO_CIRCLE = (87.5, 87.5, 87.5, "#15E0ED")
_LOGO_GRAD = (61.3272, 97.5719, 167.614, 91.5736)
_LOGO_STOPS = [
    (0, "#000000", 0.9),
    (0.0856376, "#000909", 0.8229),
    (0.2278, "#002023", 0.695),
    (0.4089, "#00464C", 0.532),
    (0.6218, "#007A85", 0.3404),
    (0.8587, "#00BCCD", 0.1271),
    (1, "#00E7FC", 0),
]
_LOGO_SHADOW = [
    (60.6, 31.7), (141.9, 19), (146.8, 23.2), (149.4, 25.6), (153.4, 30),
    (156.3, 33.5), (159.3, 37.5), (162, 41.6), (164.4, 45.8), (166.2, 49.3),
    (168.2, 53.7), (169.9, 58.1), (171.5, 62.8), (172.9, 68.6), (174, 73.9),
    (174.5, 78.4), (175, 83.7), (175, 89.9), (174.7, 94.3), (174.1, 100),
    (173.4, 104.4), (171.9, 110.6), (170, 116.7), (168.5, 120.8), (166.7, 124.7),
    (164.9, 128.3), (162.7, 132.2), (160.3, 136.1), (156.8, 141), (154.4, 143.9),
    (150.4, 148.3), (146.2, 152.4), (141.1, 156.7), (135.8, 160.4), (132.8, 162.4),
    (128.3, 164.9), (121.7, 168.2), (116.6, 162.4), (80.6, 122.1), (54.6, 61.3),
]
_LOGO_BIRD = [
    (80.6, 122.1), (116.6, 162.4), (104.6, 137.2), (115.8, 148.9), (79.3, 83),
    (95, 63.4), (153.7, 39.1), (95, 40.4), (133.2, 20.3), (60.6, 31.7),
    (54.6, 61.3), (25.3, 63.2), (20.2, 88.3), (34.9, 76.3),
]


def paint_logo(p, x, y, size):
    """Parrot logosunu (x, y) konumuna size x size piksel olarak çizer."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.translate(x, y)
    p.scale(size / 175.0, size / 175.0)
    p.setPen(Qt.NoPen)
    cx, cy, r, col = _LOGO_CIRCLE
    p.setBrush(QColor(col))
    p.drawEllipse(QPointF(cx, cy), r, r)
    gx1, gy1, gx2, gy2 = _LOGO_GRAD
    g = QLinearGradient(gx1, gy1, gx2, gy2)
    for off, hexcol, alpha in _LOGO_STOPS:
        cc = QColor(hexcol)
        cc.setAlphaF(alpha)
        g.setColorAt(off, cc)
    p.setBrush(QBrush(g))
    p.drawPolygon(QPolygonF([QPointF(a, b) for a, b in _LOGO_SHADOW]))
    p.setBrush(QColor("#000000"))
    p.drawPolygon(QPolygonF([QPointF(a, b) for a, b in _LOGO_BIRD]))
    p.restore()


def logo_pixmap(size, dpr=2.0):
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    pm.setDevicePixelRatio(dpr)
    p = QPainter(pm)
    paint_logo(p, 0, 0, size)
    p.end()
    return pm


def logo_icon():
    ic = QIcon()
    for s in (16, 24, 32, 48, 64, 128):
        ic.addPixmap(logo_pixmap(s, 1.0))
    return ic


def swatch_icon(rgb, size=14):
    """Renk düğmesi/menü için yuvarlak renk örneği; rgb=None → gökkuşağı pastası."""
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    pm.setDevicePixelRatio(dpr)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    r = QRectF(1, 1, size - 2, size - 2)
    p.setPen(QPen(QColor(255, 255, 255, 110), 1))
    if rgb:
        p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        p.drawEllipse(r)
    else:
        p.setPen(Qt.NoPen)
        for k in range(6):
            p.setBrush(QColor.fromHsvF(k / 6.0 * 0.82, 0.85, 1.0))
            p.drawPie(r, int(k * 60 * 16), 60 * 16)
    p.end()
    return QIcon(pm)


# ------------------------------------------------------------------ ses / medya
class Capture(threading.Thread):
    """parec ile varsayılan çıkışın monitörünü dinler (PulseAudio ve PipeWire)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.buf = np.zeros(FFT_N, dtype=np.float32)
        self.lock = threading.Lock()
        self.proc = None
        self.stop_flag = False
        self.error = ""

    @staticmethod
    def default_monitor():
        try:
            out = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                                 text=True, timeout=3).stdout.strip()
            if out:
                return out + ".monitor"
        except Exception:  # noqa: BLE001
            pass
        return "@DEFAULT_MONITOR@"

    def push(self, arr):
        with self.lock:
            if len(arr) >= FFT_N:
                self.buf = arr[-FFT_N:].copy()
            else:
                self.buf = np.concatenate((self.buf[len(arr):], arr))

    def snapshot(self):
        with self.lock:
            b = self.buf.copy()
        return b, float(np.abs(b[-2048:]).max())

    def run(self):
        while not self.stop_flag:
            src = self.default_monitor()
            cmd = ["parec", "-d", src, "--format=float32le", f"--rate={RATE}",
                   "--channels=1", "--latency-msec=20"]
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                self.error = "parec bulunamadı. Kurun: sudo apt install pulseaudio-utils"
                return
            fd = self.proc.stdout.fileno()
            left, t_check = b"", time.monotonic()
            while not self.stop_flag:
                data = os.read(fd, 4096)
                if not data:
                    break
                data = left + data
                n = len(data) // 4 * 4
                left = data[n:]
                if n:
                    self.push(np.frombuffer(data[:n], dtype="<f4"))
                if time.monotonic() - t_check > 3:
                    t_check = time.monotonic()
                    if self.default_monitor() != src:  # çıkış cihazı değişti
                        break
            self.kill()
            if not self.stop_flag:
                time.sleep(1)

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def stop(self):
        self.stop_flag = True
        self.kill()


class MediaWatcher(QObject):
    """playerctl ile (MPRIS) çalan şarkıyı izler, kapaktan baskın rengi çıkarır."""
    changed = pyqtSignal(str, str, str, int, int, int)  # durum, başlık, sanatçı, r, g, b (-1 = yok)
    FMT = "{{status}}\t{{artist}}\t{{title}}\t{{mpris:artUrl}}"

    def __init__(self):
        super().__init__()
        self.available = shutil.which("playerctl") is not None
        self.stop_flag = False
        self.last_art = None
        self.color = (-1, -1, -1)

    def start(self):
        if self.available:
            threading.Thread(target=self.run, daemon=True).start()

    def color_for(self, art):
        try:
            if art.startswith("file://"):
                data = Path(unquote(art[7:])).read_bytes()
            elif art.startswith(("http://", "https://")):
                req = Request(art, headers={"User-Agent": "plasmadock/1.0"})
                data = urlopen(req, timeout=5).read(3_000_000)
            else:
                return (-1, -1, -1)
            img = QImage()
            if not img.loadFromData(data):
                return (-1, -1, -1)
            r, g, b = dominant_color(img)
            return (int(r), int(g), int(b))
        except Exception:  # noqa: BLE001
            return (-1, -1, -1)

    def run(self):
        while not self.stop_flag:
            status = title = artist = art = ""
            try:
                out = subprocess.run(["playerctl", "metadata", "--format", self.FMT],
                                     capture_output=True, text=True, timeout=3)
                if out.returncode == 0 and out.stdout.strip():
                    parts = (out.stdout.rstrip("\n").split("\t") + ["", "", "", ""])[:4]
                    status, artist, title, art = parts
            except Exception:  # noqa: BLE001
                pass
            if art != self.last_art:
                self.last_art = art
                self.color = self.color_for(art) if art else (-1, -1, -1)
            self.changed.emit(status, title, artist, *self.color)
            time.sleep(2)


# ------------------------------------------------------------------ arayüz
class ElidedLabel(QLabel):
    def __init__(self):
        super().__init__("")
        self.full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def set_full(self, text):
        self.full = text
        self.refresh()

    def refresh(self):
        self.setText(self.fontMetrics().elidedText(self.full, Qt.ElideRight, max(self.width(), 10)))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.refresh()


class Canvas(QWidget):
    """Seçili modu çizer. Fareyi görmez; sürükleme ve menü Dock'a gider."""

    def __init__(self, dock):
        super().__init__(dock)
        self.dock = dock
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, _):
        d = self.dock
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = float(self.width()), float(self.height())
        {"bars": self.draw_bars, "circle": self.draw_circle,
         "waterfall": self.draw_waterfall, "net": self.draw_net}[d.cfg["mode"]](p, W, H)

    @staticmethod
    def rect(p, x, y, w, h, rad):
        if rad > 0:
            p.drawRoundedRect(QRectF(x, y, w, h), rad, rad)
        else:
            p.drawRect(QRectF(x, y, w, h))

    def vgradient(self, H, mirror=False):
        low, high = self.dock.qlow(), self.dock.qhigh()
        if mirror:
            g = QLinearGradient(0, 0, 0, H)
            g.setColorAt(0.0, high)
            g.setColorAt(0.5, low)
            g.setColorAt(1.0, high)
        else:
            g = QLinearGradient(0, H, 0, 0)
            g.setColorAt(0.0, low)
            g.setColorAt(1.0, high)
        return QBrush(g)

    # --- çubuklar
    def draw_bars(self, p, W, H):
        d = self.dock
        an = d.an
        n, gap = an.n, 3.0
        bw = max((W - gap * (n - 1)) / n, 1.0)
        rad = min(bw / 2.0, 4.0)
        mirror = d.cfg["style"] == "mirror"
        rainbow = d.cfg["color"] == "rainbow"
        brush = self.vgradient(H, mirror)
        mid = H / 2.0
        cap_col = QColor(255, 255, 255, 215)
        p.setPen(Qt.NoPen)
        for i in range(n):
            x = i * (bw + gap)
            p.setBrush(QColor.fromHsvF((i / n) * 0.82, 0.85, 1.0) if rainbow else brush)
            c, pk = float(an.cur[i]), float(an.peak[i])
            if mirror:
                half = max(c * (mid - 2), 1.0)
                self.rect(p, x, mid - half, bw, 2 * half, rad)
                if d.cfg["caps"] and pk > 0.02:
                    p.setBrush(cap_col)
                    off = pk * (mid - 2)
                    self.rect(p, x, mid - off - 4, bw, 3, rad)
                    self.rect(p, x, mid + off + 1, bw, 3, rad)
            else:
                hgt = max(c * (H - 4), 2.0)
                self.rect(p, x, H - hgt, bw, hgt, rad)
                if d.cfg["caps"] and pk > 0.02:
                    p.setBrush(cap_col)
                    self.rect(p, x, H - pk * (H - 4) - 4, bw, 3, rad)

    # --- daire: her zaman ortalanmış KARE kutu içinde, ortada Parrot logosu
    def draw_circle(self, p, W, H):
        d = self.dock
        an = d.an
        n = an.n
        side = min(W, H)
        bx, by = (W - side) / 2.0, (H - side) / 2.0
        cx, cy = W / 2.0, H / 2.0

        # kare kutu
        edge = d.qhigh()
        edge.setAlpha(130)
        p.setPen(QPen(edge, 1.5))
        p.setBrush(QColor(9, 12, 18, 150))
        p.drawRoundedRect(QRectF(bx + 0.75, by + 0.75, side - 1.5, side - 1.5), 18, 18)

        R = side / 2.0 - 14
        r0 = R * 0.32
        bass = float(an.cur[: max(2, n // 12)].mean())
        r_in = r0 * (1.0 + 0.12 * bass)
        low, high = d.low, d.high
        rainbow = d.cfg["color"] == "rainbow"

        # ince kılavuz halka
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), R, R)

        # merkez parıltısı (bas ile güçlenir)
        glow = QRadialGradient(QPointF(cx, cy), r_in * 1.9)
        gc = d.qhigh()
        gc.setAlpha(int(30 + 130 * bass))
        g1 = QColor(gc)
        g1.setAlpha(0)
        glow.setColorAt(0.0, QColor(gc))
        glow.setColorAt(1.0, g1)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(cx, cy), r_in * 1.9, r_in * 1.9)

        # halka + ortadaki logo (Parrot; --logo ile özel PNG verilirse o)
        ring = d.qlow()
        ring.setAlpha(235)
        p.setPen(QPen(ring, 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r_in, r_in)
        size = max(r_in * 2.0 - 7.0, 8.0)
        if d.logo is not None and not d.logo.isNull():
            s_px = int(size)
            pm = d.logo.scaled(s_px, s_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap(int(cx - pm.width() / 2), int(cy - pm.height() / 2), pm)
        else:
            paint_logo(p, cx - size / 2.0, cy - size / 2.0, size)

        # çubuklar: sağ ve sol yarı simetrik, bas üstte, tiz altta
        span = max(R - r_in - 8, 4.0)
        width = max(2.0, min(math.pi * r_in / n * 0.75, 7.0))
        for i in range(n):
            c = float(an.cur[i])
            length = max(c * span, 2.0)
            col = (QColor.fromHsvF((i / n) * 0.82, 0.85, 1.0) if rainbow
                   else d.qcolor(mix(low, high, c)))
            pen = QPen(col, width)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            for sign in (1, -1):
                t = -math.pi / 2 + sign * math.pi * (i + 0.5) / n
                dx, dy = math.cos(t), math.sin(t)
                p.drawLine(QPointF(cx + dx * (r_in + 6), cy + dy * (r_in + 6)),
                           QPointF(cx + dx * (r_in + 6 + length), cy + dy * (r_in + 6 + length)))

    # --- şelale (spektrogram)
    def draw_waterfall(self, p, W, H):
        d = self.dock
        rows = d.wf[::-1]  # üstte tiz, altta bas
        argb = np.ascontiguousarray(d.lut[rows])
        h, w = argb.shape
        d._wf_bytes = argb.tobytes()
        img = QImage(d._wf_bytes, w, h, w * 4, QImage.Format_ARGB32)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(QRectF(0, 0, W, H), img)

    # --- ağ
    def draw_net(self, p, W, H):
        d = self.dock
        net = d.net
        n = len(net.LABELS)
        top, bot, gap = 14.0, 24.0, 6.0
        bw = max((W - gap * (n - 1)) / n, 4.0)
        area = max(H - top - bot, 10.0)
        rainbow = d.cfg["color"] == "rainbow"
        brush = self.vgradient(top + area)
        f = QFont()
        f.setPixelSize(10 if bw > 40 else 8)
        p.setFont(f)
        rad = min(bw / 2.0, 4.0)
        for i in range(n):
            x = i * (bw + gap)
            c = float(net.g.cur[i])
            hgt = max(c * area, 2.0)
            y = top + area - hgt
            p.setPen(Qt.NoPen)
            p.setBrush(QColor.fromHsvF((i / n) * 0.82, 0.85, 1.0) if rainbow else brush)
            self.rect(p, x, y, bw, hgt, rad)
            if d.cfg["caps"] and net.g.peak[i] > 0.02:
                p.setBrush(QColor(255, 255, 255, 215))
                self.rect(p, x, top + area - net.g.peak[i] * area - 4, bw, 3, rad)
            p.setPen(QColor(201, 209, 217, 220))
            p.drawText(QRectF(x - gap / 2, max(y - 14, 0), bw + gap, 13), Qt.AlignCenter, net.text(i))
            p.setPen(QColor(139, 148, 158))
            p.drawText(QRectF(x - gap / 2, H - bot + 4, bw + gap, bot - 4),
                       Qt.AlignHCenter | Qt.AlignTop, net.LABELS[i])


STYLE = """
QToolButton { color:#c9d1d9; background:transparent; border:none; padding:3px 9px;
              border-radius:6px; font-size:11px; }
QToolButton:hover { background:rgba(255,255,255,0.13); }
QToolButton:checked { background:rgba(88,166,255,0.40); color:white; }
QToolButton::menu-indicator { image:none; }
QLabel { color:#8b949e; font-size:11px; }
QMenu { background:#161b22; color:#c9d1d9; border:1px solid #30363d; padding:4px; }
QMenu::item { padding:4px 18px; border-radius:4px; }
QMenu::item:selected { background:#1f6feb; color:white; }
QMenu::separator { height:1px; background:#30363d; margin:4px 6px; }
"""


class Dock(QWidget):
    def __init__(self, cfg, capture, media):
        super().__init__()
        self.cfg = cfg
        self.cap = capture
        self.media = media
        self.an = Analyzer(cfg["bars"])
        self.net = NetStats()
        self.logo = QPixmap(cfg["logo"]) if cfg["logo"] and Path(cfg["logo"]).exists() else None  # özel logo (isteğe bağlı)

        self.low, self.high = (31, 111, 235), (160, 200, 255)
        self.media_color = None
        self._ui_key = None
        self.color_actions = {}
        self.track = ""
        self.hovering = False
        self.drag_off = None
        self.last_active = time.monotonic()
        self.last_tick = time.monotonic()
        self.win_opacity = 1.0
        self.wf_cols = 240
        self.wf = np.zeros((cfg["bars"], self.wf_cols), dtype=np.uint8)
        self.wf_acc = 0.0
        self.lut = build_lut(self.low, self.high)
        self._lut_key = None
        self._wf_bytes = b""

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setWindowIcon(logo_icon())
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet(STYLE)
        self.setMinimumSize(340, 130)
        self.apply_flags()

        # --- başlık satırı: Parrot logosu + "Parrot Music Spectrum v1.0" + renk + ayarlar
        self.header = QWidget()
        hl = QHBoxLayout(self.header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        self.logo_lbl = QLabel()
        self.logo_lbl.setPixmap(logo_pixmap(26))
        self.logo_lbl.setFixedSize(26, 26)
        hl.addWidget(self.logo_lbl)
        self.title_lbl = QLabel()
        self.title_lbl.setTextFormat(Qt.RichText)
        hl.addWidget(self.title_lbl)
        hl.addStretch(1)
        self.colorbtn = QToolButton()
        self.colorbtn.setIconSize(QSize(14, 14))
        self.colorbtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.colorbtn.setPopupMode(QToolButton.InstantPopup)
        hl.addWidget(self.colorbtn)
        self.gear = QToolButton()
        self.gear.setText("⚙")
        self.gear.setToolTip("Ayarlar")
        hl.addWidget(self.gear)
        close = QToolButton()
        close.setText("✕")
        close.setToolTip("Kapat")
        close.clicked.connect(QApplication.quit)
        hl.addWidget(close)

        # --- araç çubuğu: modlar + çalan şarkı
        self.toolbar = QWidget()
        tl = QHBoxLayout(self.toolbar)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.mode_btns = {}
        for key, label in MODES:
            b = QToolButton()
            b.setText(label)
            b.setCheckable(True)
            b.setChecked(key == cfg["mode"])
            b.clicked.connect(lambda _c, k=key: self.set_mode(k))
            self.group.addButton(b)
            self.mode_btns[key] = b
            tl.addWidget(b)
        self.label = ElidedLabel()
        self.label.setContentsMargins(10, 0, 6, 0)
        tl.addWidget(self.label, 1)

        self.canvas = Canvas(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(4)
        lay.addWidget(self.header)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self.grip = QSizeGrip(self)
        self.grip.setFixedSize(14, 14)

        self.menu = self.build_menu()
        self.gear.setMenu(self.menu)
        self.gear.setPopupMode(QToolButton.InstantPopup)
        self.colorbtn.setMenu(self.color_menu)
        self.refresh_color_ui()

        self.place_initial()
        self.toolbar.setVisible(not cfg["toolbar_hover"])

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(600)
        self.save_timer.timeout.connect(self.save_config)

        self.net_timer = QTimer(self)
        self.net_timer.timeout.connect(self.net.sample)
        self.net_timer.start(500)
        self.net.sample()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

        media.changed.connect(self.on_media)

    # --- pencere
    def apply_flags(self):
        flags = Qt.FramelessWindowHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus | Qt.NoDropShadowWindowHint
        flags |= Qt.WindowStaysOnTopHint if self.cfg["layer"] == "above" else Qt.WindowStaysOnBottomHint
        self.setWindowFlags(flags)

    def place_initial(self):
        g = self.cfg["geom"]
        screens = QApplication.screens()
        ok = False
        if g and len(g) == 4:
            from PyQt5.QtCore import QRect
            r = QRect(*g)
            ok = any(s.geometry().intersects(r) for s in screens)
        if ok:
            self.setGeometry(*g)
        else:
            ag = QApplication.primaryScreen().availableGeometry()
            w, h = 720, 190
            self.setGeometry(ag.left() + (ag.width() - w) // 2, ag.bottom() - h - 16, w, h)

    def move_preset(self, hx, vy):
        scr = QApplication.screenAt(self.geometry().center()) or QApplication.primaryScreen()
        g = scr.availableGeometry()
        w, h, m = self.width(), self.height(), 16
        x = {"left": g.left() + m, "center": g.left() + (g.width() - w) // 2,
             "right": g.right() - w - m + 1}[hx]
        y = {"top": g.top() + m, "middle": g.top() + (g.height() - h) // 2,
             "bottom": g.bottom() - h - m + 1}[vy]
        self.move(x, y)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(13, 17, 23, int(self.cfg["panel_alpha"])))
        edge = self.qhigh()
        edge.setAlpha(110)
        p.setPen(QPen(edge, 1))
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 14, 14)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.grip.move(self.width() - 16, self.height() - 16)
        self.grip.raise_()
        if hasattr(self, "colorbtn"):
            self.colorbtn.setToolButtonStyle(Qt.ToolButtonIconOnly if self.width() < 470
                                             else Qt.ToolButtonTextBesideIcon)
        self.save_timer.start()

    def moveEvent(self, ev):
        super().moveEvent(ev)
        self.save_timer.start()

    # --- fare: taşı / menü / mod değiştir
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            h = self.windowHandle()
            if h is not None and hasattr(h, "startSystemMove") and h.startSystemMove():
                ev.accept()
                return
            self.drag_off = ev.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev):
        if self.drag_off is not None and ev.buttons() & Qt.LeftButton:
            self.move(ev.globalPos() - self.drag_off)

    def mouseReleaseEvent(self, ev):
        self.drag_off = None

    def mouseDoubleClickEvent(self, ev):
        keys = [k for k, _ in MODES]
        self.set_mode(keys[(keys.index(self.cfg["mode"]) + 1) % len(keys)])

    def contextMenuEvent(self, ev):
        self.menu.popup(ev.globalPos())

    def enterEvent(self, ev):
        self.hovering = True
        if self.cfg["toolbar_hover"]:
            self.toolbar.show()

    def leaveEvent(self, ev):
        self.hovering = False
        if self.cfg["toolbar_hover"]:
            self.toolbar.hide()

    # --- ayarlar
    def set_cfg(self, key, value):
        self.cfg[key] = value
        self.save_config()

    def set_mode(self, mode):
        old = self.cfg["mode"]
        if mode != old and mode_category(mode) != mode_category(old):
            self.remember_geometry()
            self.cfg["mode"] = mode
            self.apply_mode_geometry(mode)       # daire ⇄ şerit: pencere otomatik biçim değiştirir
        self.mode_btns[mode].setChecked(True)
        self.set_cfg("mode", mode)

    def remember_geometry(self):
        g = self.geometry()
        self.cfg.setdefault("geoms", {})[mode_category(self.cfg["mode"])] = [g.x(), g.y(), g.width(), g.height()]

    def apply_mode_geometry(self, mode):
        """Yeni düzenin kayıtlı (yoksa varsayılan) boyutunu, mevcut pencerenin merkezine ortalayarak uygula."""
        cat = mode_category(mode)
        saved = self.cfg.get("geoms", {}).get(cat)
        w, h = (saved[2], saved[3]) if saved and len(saved) == 4 else DEFAULT_SIZE[cat]
        c = self.geometry().center()
        scr = QApplication.screenAt(c) or QApplication.primaryScreen()
        ag = scr.availableGeometry()
        x, y, w, h = clamp_rect(c.x() - w // 2, c.y() - h // 2, w, h,
                                ag.left(), ag.top(), ag.width(), ag.height())
        self.setGeometry(x, y, max(w, 340), max(h, 130))

    def build_menu(self):
        m = QMenu(self)

        def radio(title, items, current, cb):
            sub = m.addMenu(title)
            grp = QActionGroup(sub)
            for key, label in items:
                a = QAction(label, sub, checkable=True)
                a.setChecked(key == current)
                a.triggered.connect(lambda _c, k=key: cb(k))
                grp.addAction(a)
                sub.addAction(a)

        def check(title, key, cb=None):
            a = QAction(title, m, checkable=True)
            a.setChecked(bool(self.cfg[key]))
            a.toggled.connect(lambda v: (self.set_cfg(key, v), cb() if cb else None))
            m.addAction(a)

        radio("Mod", MODES, self.cfg["mode"], self.set_mode)
        radio("Çubuk stili", [("bars", "Normal"), ("mirror", "Ayna")], self.cfg["style"],
              lambda k: self.set_cfg("style", k))
        self.color_menu = m.addMenu("Renk (Linux dağıtımı)")
        cgrp = QActionGroup(self.color_menu)
        for key, label in COLOR_CHOICES:
            a = QAction(swatch_icon(self.base_for(key)), label, self.color_menu, checkable=True)
            a.setChecked(key == self.cfg["color"])
            a.triggered.connect(lambda _c, k=key: self.set_color(k))
            cgrp.addAction(a)
            self.color_menu.addAction(a)
            self.color_actions[key] = a
            if key == DISTROS[-1][0]:
                self.color_menu.addSeparator()
        if not self.media.available:
            hint = m.addAction("Albüm rengi için: sudo apt install playerctl")
            hint.setEnabled(False)
        m.addSeparator()
        check("Tepe işaretçileri", "caps")
        layer = QAction("Pencerelerin üstünde dur", m, checkable=True)
        layer.setChecked(self.cfg["layer"] == "above")
        layer.toggled.connect(self.toggle_layer)
        m.addAction(layer)
        check("Araç çubuğunu yalnız üzerine gelince göster", "toolbar_hover", self.refresh_toolbar)
        check("Sessizlikte sönükleştir", "dim")
        m.addSeparator()
        pos = m.addMenu("Konum")
        for label, hx, vy in (("Sol alt", "left", "bottom"), ("Orta alt", "center", "bottom"),
                              ("Sağ alt", "right", "bottom"), ("Sol üst", "left", "top"),
                              ("Orta üst", "center", "top"), ("Sağ üst", "right", "top"),
                              ("Ekranın ortası", "center", "middle")):
            pos.addAction(label, lambda hx=hx, vy=vy: self.move_preset(hx, vy))
        size = m.addMenu("Boyut")
        for label, w, h in (("Küçük (560×170)", 560, 170), ("Orta (720×230)", 720, 230),
                            ("Büyük (1100×300)", 1100, 300), ("Şerit (1400×140)", 1400, 140),
                            ("Kare kutu / Daire (380×440)", 380, 440)):
            size.addAction(label, lambda w=w, h=h: self.resize(w, h))
        radio("Panel opaklığı", [(90, "Cam"), (190, "Yarı saydam"), (250, "Opak")],
              self.cfg["panel_alpha"], lambda k: self.set_cfg("panel_alpha", k))
        m.addSeparator()
        m.addAction("Öne getir", self.raise_)
        m.addAction("Çıkış", QApplication.quit)
        return m

    def refresh_toolbar(self):
        self.toolbar.setVisible(not self.cfg["toolbar_hover"] or self.hovering)

    def toggle_layer(self, above):
        self.cfg["layer"] = "above" if above else "below"
        geom = self.geometry()
        self.apply_flags()
        self.setGeometry(geom)
        self.show()
        self.save_config()

    def save_config(self):
        g = self.geometry()
        self.cfg["geom"] = [g.x(), g.y(), g.width(), g.height()]
        self.cfg.setdefault("geoms", {})[mode_category(self.cfg["mode"])] = list(self.cfg["geom"])
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(json.dumps(self.cfg, indent=2))
        except OSError:
            pass

    # --- medya / renk
    def on_media(self, status, title, artist, r, g, b):
        self.media_color = (float(r), float(g), float(b)) if r >= 0 else None
        text = ""
        if title:
            text = f"♪ {artist} — {title}" if artist else f"♪ {title}"
            if status == "Paused":
                text += "  (duraklatıldı)"
        self.label.set_full(text)

    def base_for(self, key):
        """Renk seçeneği için örnek (düğme/menü ikonu) rengi; gökkuşağı → None."""
        if key in DISTRO_RGB:
            return DISTRO_RGB[key]
        if key == "rainbow":
            return None
        if key == "album" and self.media_color:
            return self.media_color
        c = QApplication.palette().highlight().color()
        return (c.red(), c.green(), c.blue())

    def set_color(self, key):
        self.set_cfg("color", key)
        self.refresh_color_ui()
        self.canvas.update()

    def refresh_color_ui(self):
        """Başlıktaki renk düğmesi, menü işaretleri ve sürüm yazısının rengi."""
        key = self.cfg["color"]
        label = dict(COLOR_CHOICES).get(key, key)
        self.colorbtn.setIcon(swatch_icon(self.base_for(key)))
        self.colorbtn.setText(SHORT_NAMES.get(key, key))
        self.colorbtn.setToolTip(f"Renk: {label}")
        for k, a in self.color_actions.items():
            a.setChecked(k == key)
            if k in ("album", "accent"):
                a.setIcon(swatch_icon(self.base_for(k)))
        self.title_lbl.setText(
            f'<span style="color:#f0f6fc;font-size:13px;font-weight:600">{APP_NAME}</span> '
            f'<span style="color:{self.qhigh().name()};font-size:10px;font-weight:600">{APP_VERSION}</span>')

    def target_palette(self):
        name = self.cfg["color"]
        if name in FIXED:
            return FIXED[name]
        if name == "album" and self.media_color:
            base = self.media_color
        elif name.startswith("#") and len(name) == 7:
            c = QColor(name)
            base = (c.red(), c.green(), c.blue())
        else:
            c = QApplication.palette().highlight().color()
            base = (c.red(), c.green(), c.blue())
        return tuple(x * 0.45 for x in base), mix(base, (255, 255, 255), 0.25)

    @staticmethod
    def qcolor(t):
        return QColor(int(max(0, min(255, t[0]))), int(max(0, min(255, t[1]))),
                      int(max(0, min(255, t[2]))))

    def qlow(self):
        return self.qcolor(self.low)

    def qhigh(self):
        return self.qcolor(self.high)

    # --- animasyon
    def tick(self):
        now = time.monotonic()
        dt = min(now - self.last_tick, 0.1)
        self.last_tick = now
        mode = self.cfg["mode"]

        tl, th = self.target_palette()
        k = 1.0 - math.exp(-dt / 0.35)
        self.low, self.high = mix(self.low, tl, k), mix(self.high, th, k)
        ukey = (tuple(int(x) // 12 for x in self.high), self.cfg["color"], self.media_color)
        if ukey != self._ui_key:
            self._ui_key = ukey
            self.refresh_color_ui()

        idle = False
        if mode in AUDIO_MODES:
            samples, level = self.cap.snapshot()
            if level > 1e-4:
                self.last_active = now
            idle = (now - self.last_active) > 3.0
            if idle:
                self.an.decay(dt)
            else:
                self.an.update(dt, samples)
            if mode == "waterfall":
                self.wf_acc += dt
                while self.wf_acc >= 1.0 / 40:
                    self.wf_acc -= 1.0 / 40
                    self.wf[:, :-1] = self.wf[:, 1:]
                    self.wf[:, -1] = np.clip(self.an.cur * 255, 0, 255).astype(np.uint8)
                rainbow = self.cfg["color"] == "rainbow"
                key = (tuple(int(x) // 4 for x in (*self.low, *self.high)), rainbow)
                if key != self._lut_key:
                    self._lut_key = key
                    self.lut = build_lut(self.low, self.high, rainbow)
        else:
            self.net.step(dt)

        # sessizlikte paneli sönükleştir (fareyle üzerine gelince geri gelir)
        target = 0.35 if (self.cfg["dim"] and idle and not self.hovering) else 1.0
        self.win_opacity += (target - self.win_opacity) * (1.0 - math.exp(-dt / 0.3))
        if abs(self.windowOpacity() - self.win_opacity) > 0.02:
            self.setWindowOpacity(self.win_opacity)

        if not (idle and mode in AUDIO_MODES and float(self.an.cur.max()) < 1e-3
                and abs(self.win_opacity - target) < 0.01):
            self.canvas.update()
            self.update()


def load_config():
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_FILE.read_text()))
    except (OSError, ValueError):
        pass
    cfg["color"] = LEGACY_COLORS.get(cfg["color"], cfg["color"])
    return cfg


def install_autostart():
    args = [a for a in sys.argv[1:] if a != "--install-autostart"]
    exec_line = " ".join(shlex.quote(x) for x in
                         ["/usr/bin/env", "python3", str(Path(__file__).resolve()), *args])
    d = Path.home() / ".config" / "autostart"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "plasmadock.desktop"
    f.write_text(f"[Desktop Entry]\nType=Application\nName={APP_NAME} {APP_VERSION}\n"
                 f"Exec={exec_line}\nX-KDE-autostart-after=panel\n")
    print(f"Otomatik başlatma eklendi: {f}")


def main():
    ap = argparse.ArgumentParser(description=f"{APP_NAME} {APP_VERSION} - taşınabilir masaüstü görselleştirici")
    ap.add_argument("--mode", choices=[k for k, _ in MODES], help="başlangıç modu")
    ap.add_argument("--bars", type=int, help="çubuk sayısı (varsayılan 48)")
    ap.add_argument("--color", choices=[k for k, _ in COLOR_CHOICES],
                    help="renk: " + ", ".join(k for k, _ in COLOR_CHOICES))
    ap.add_argument("--logo", help="daire modunun ortasındaki logo (varsayılan: yerleşik Parrot logosu)")
    ap.add_argument("--reset-config", action="store_true", help="kayıtlı konum/ayarları sil")
    ap.add_argument("--install-autostart", action="store_true")
    args = ap.parse_args()

    if args.install_autostart:
        install_autostart()
        return
    if args.reset_config and CONFIG_FILE.exists():
        CONFIG_FILE.unlink()

    cfg = load_config()
    if args.mode:
        cfg["mode"] = args.mode
    if args.color:
        cfg["color"] = args.color
    if args.bars:
        cfg["bars"] = max(12, min(args.bars, 128))
    if args.logo:
        cfg["logo"] = os.path.expanduser(args.logo)
    if not cfg["logo"] and (CONFIG_DIR / "logo.png").exists():
        cfg["logo"] = str(CONFIG_DIR / "logo.png")

    cap = Capture()
    cap.start()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("plasmadock")
    app.setWindowIcon(logo_icon())
    app.setQuitOnLastWindowClosed(False)
    app.aboutToQuit.connect(cap.stop)
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    media = MediaWatcher()
    dock = Dock(cfg, cap, media)
    media.start()
    dock.show()

    tray = QSystemTrayIcon(logo_icon(), app)
    tray.setToolTip(f"{APP_NAME} {APP_VERSION}")
    tray.setContextMenu(dock.menu)
    tray.show()

    def check_error():
        if cap.error:
            print(cap.error, file=sys.stderr)
            app.quit()
    err_timer = QTimer()
    err_timer.timeout.connect(check_error)
    err_timer.start(500)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
