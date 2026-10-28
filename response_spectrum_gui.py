#!/usr/bin/env python3
"""
地震加速度反应谱 SA 计算工具 (GUI)

输入:
  * miniSEED (如 TitanSMA 记录, 单位 counts, 默认 2 counts/µg)
  * K-NET / KiK-net ASCII (.EW .NS .UD .EW1 ... , 比例因子从文件头自动读取);
    可直接选 NIED 下载的 .tar.gz 压缩包或文件夹, 多台站时弹窗选择台站
  * 可一次选多个文件/多个分量

计算:
  * 单自由度振子, 阻尼比默认 5%, 周期范围与点数可设
  * Nigam-Jennings (1969) 分段线性精确解法, 对任意采样间隔都无数值误差
  * 输出 PSA(伪谱加速度 = ω²·Sd, 工程常用) 和 SA(绝对加速度最大值),
    以及 SV/SD; 可显示水平两分量几何平均
  * 叠加 GB 50011-2010(2016年版) 设计反应谱对比 (烈度/地震水准/场地类别/
    设计地震分组可选, 按所设阻尼比修正)

运行:  python response_spectrum_gui.py
需要:  pip install obspy numpy scipy matplotlib
"""
import csv
import queue
import re
import threading
import traceback
from pathlib import Path

import numpy as np

G = 980.665  # gal per g

STD_PERIODS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0,
               1.5, 2.0, 3.0, 4.0, 5.0, 6.0]


# ======================================================================
#  读取数据 -> 加速度 (gal)
# ======================================================================
KNET_EXT = re.compile(r"\.(EW|NS|UD)[12]?$", re.I)


def is_knet_file(path):
    p = Path(path)
    if KNET_EXT.search(p.name):
        return True
    try:
        with open(p, "rb") as f:
            return f.read(11) == b"Origin Time"
    except OSError:
        return False


def knet_header(path):
    """只读 K-NET/KiK-net ASCII 的 17 行文件头, 返回 dict"""
    with open(path, encoding="ascii", errors="replace") as f:
        lines = [next(f, "") for _ in range(17)]
    hdr = {ln[:18].strip(): ln[18:].strip() for ln in lines}
    ext = Path(path).suffix.lstrip(".").upper()
    sta = hdr.get("Station Code", Path(path).stem)
    # KiK-net: *1 = 井下, *2 = 地表; K-NET: 无数字 = 地表
    loc = {"1": "井下", "2": "地表"}.get(ext[-1:], "")
    hdr.update(_ext=ext, _sta=sta, _grp=f"{sta}{'(' + loc + ')' if loc else ''}",
               _comp=ext.rstrip("12"), _net="KiK-net" if loc else "K-NET")
    return hdr


def read_knet(path):
    """K-NET/KiK-net ASCII -> [(标签, 采样率, 加速度 gal, 信息, 分量, 分组)]"""
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    hdr = {ln[:18].strip(): ln[18:].strip() for ln in lines[:17]}
    h2 = knet_header(path)
    fs = float(re.sub(r"[^\d.]", "", hdr["Sampling Freq(Hz)"]))
    m = re.match(r"\s*([\d.]+)\(gal\)/([\d.]+)", hdr["Scale Factor"])
    calib = float(m.group(1)) / float(m.group(2))
    counts = np.array(" ".join(lines[17:]).split(), dtype=np.float64)
    acc = counts * calib
    label = f"{h2['_sta']}.{h2['_ext']}"
    info = (f"{label}: {h2['_net']}, fs={fs:g} Hz, n={len(acc)}, 比例因子 {calib:.6e} gal/count, "
            f"文件头 Max.Acc = {hdr.get('Max. Acc. (gal)', '?')} gal, "
            f"发震 {hdr.get('Origin Time', '?')} JST M{hdr.get('Mag.', '?')}, "
            f"台站 ({hdr.get('Station Lat.', '?')}, {hdr.get('Station Long.', '?')})")
    return [(label, fs, acc, info, h2["_comp"], h2["_grp"])]


def extract_archive(path, dest):
    """解压 K-NET/KiK-net 下载包 (.tar.gz / .tgz / .tar / .zip, 可多层嵌套),
    返回其中所有 K-NET ASCII 分量文件路径"""
    import tarfile
    import zipfile
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    found, todo = [], [Path(path)]
    while todo:
        a = todo.pop()
        sub = dest / re.sub(r"[^\w.-]", "_", a.name)
        sub.mkdir(exist_ok=True)
        if zipfile.is_zipfile(a):
            with zipfile.ZipFile(a) as z:
                for n in z.namelist():
                    if ".." in Path(n).parts or Path(n).is_absolute():
                        continue
                    z.extract(n, sub)
        else:
            with tarfile.open(a) as t:
                try:
                    t.extractall(sub, filter="data")
                except TypeError:                     # Python < 3.12
                    t.extractall(sub, members=[m for m in t.getmembers()
                                               if m.isfile() and ".." not in Path(m.name).parts
                                               and not Path(m.name).is_absolute()])
        for f in sub.rglob("*"):
            if not f.is_file():
                continue
            nm = f.name.lower()
            if nm.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
                todo.append(f)
            elif KNET_EXT.search(f.name):
                found.append(str(f))
    return sorted(set(found))


def read_mseed(path, counts_per_ug):
    from obspy import read
    st = read(str(path))
    st.merge(method=1, fill_value="interpolate")
    out = []
    gal_per_count = 1e-6 * G / counts_per_ug        # 1 µg = 1e-6 g
    for tr in st:
        acc = tr.data.astype(np.float64) * gal_per_count
        info = (f"{tr.id}: miniSEED, fs={tr.stats.sampling_rate:g} Hz, n={tr.stats.npts}, "
                f"{tr.stats.starttime} ~ {tr.stats.endtime}, 换算 {counts_per_ug:g} counts/µg")
        out.append((tr.id, tr.stats.sampling_rate, acc, info, tr.stats.channel[-1:], tr.id[:-1]))
    return out


def load_any(path, counts_per_ug):
    if is_knet_file(path):
        return read_knet(path)
    return read_mseed(path, counts_per_ug)


# ======================================================================
#  预处理
# ======================================================================
def preprocess(acc, fs, detrend=True, taper_pct=5.0, band=None, order=4):
    from scipy import signal
    x = acc - acc.mean()
    if detrend:
        x = signal.detrend(x, type="linear")
    if taper_pct > 0:
        n = int(len(x) * taper_pct / 100)
        if n > 1:
            w = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
            x[:n] *= w
            x[-n:] *= w[::-1]
    if band:
        lo, hi = band
        nyq = fs / 2
        if hi and hi >= nyq:
            hi = None
        if lo and hi:
            sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        elif lo:
            sos = signal.butter(order, lo, btype="highpass", fs=fs, output="sos")
        elif hi:
            sos = signal.butter(order, hi, btype="lowpass", fs=fs, output="sos")
        else:
            sos = None
        if sos is not None:
            x = signal.sosfiltfilt(sos, x)
    return x


# ======================================================================
#  反应谱: Nigam-Jennings 精确解
# ======================================================================
def response_spectrum(acc, dt, periods, zeta=0.05):
    """
    acc: 地面加速度 (任意单位), dt: 采样间隔 (s), periods: 周期数组 (s)
    返回 dict: SD(位移, 单位=acc单位·s²), SV, PSA, SA(绝对加速度)
    运动方程: u'' + 2ζω u' + ω² u = -a_g
    """
    T = np.asarray(periods, dtype=np.float64)
    w = 2 * np.pi / T
    wd = w * np.sqrt(1 - zeta ** 2)
    sq = np.sqrt(1 - zeta ** 2)
    e = np.exp(-zeta * w * dt)
    s, c = np.sin(wd * dt), np.cos(wd * dt)

    a11 = e * (zeta / sq * s + c)
    a12 = e * s / wd
    a21 = -w / sq * e * s
    a22 = e * (c - zeta / sq * s)

    k1 = (2 * zeta ** 2 - 1) / (w ** 2 * dt)
    k2 = 2 * zeta / (w ** 3 * dt)
    b11 = e * ((k1 + zeta / w) * s / wd + (k2 + 1 / w ** 2) * c) - k2
    b12 = -e * (k1 * s / wd + k2 * c) - 1 / w ** 2 + k2
    b21 = (e * ((k1 + zeta / w) * (c - zeta / sq * s)
                - (k2 + 1 / w ** 2) * (wd * s + zeta * w * c)) + 1 / (w ** 2 * dt))
    b22 = (-e * (k1 * (c - zeta / sq * s) - k2 * (wd * s + zeta * w * c))
           - 1 / (w ** 2 * dt))

    p = -np.asarray(acc, dtype=np.float64)
    u = np.zeros_like(T)
    v = np.zeros_like(T)
    umax = np.zeros_like(T)
    vmax = np.zeros_like(T)
    amax = np.zeros_like(T)
    c2, c3 = 2 * zeta * w, w ** 2
    for i in range(len(p) - 1):
        pi, pj = p[i], p[i + 1]
        u, v = (a11 * u + a12 * v + b11 * pi + b12 * pj,
                a21 * u + a22 * v + b21 * pi + b22 * pj)
        np.maximum(umax, np.abs(u), out=umax)
        np.maximum(vmax, np.abs(v), out=vmax)
        np.maximum(amax, np.abs(c2 * v + c3 * u), out=amax)   # 绝对加速度
    return {"T": T, "SD": umax, "SV": vmax, "PSA": c3 * umax, "SA": amax}


def compute_all(records, params, log):
    """records: [(label, fs, acc_gal, info, dir, grp)] -> results list"""
    T = np.logspace(np.log10(params["tmin"]), np.log10(params["tmax"]), params["npts"])
    T = np.unique(np.concatenate([T, [t for t in STD_PERIODS
                                       if params["tmin"] <= t <= params["tmax"]]]))
    res = []
    for label, fs, acc, info, d, grp in records:
        log(info)
        x = preprocess(acc, fs, params["detrend"], params["taper"], params["band"])
        rs = response_spectrum(x, 1.0 / fs, T, params["zeta"])
        pga = np.abs(x).max()
        log(f"  预处理后 PGA = {pga:.3f} gal = {pga / G:.4f} g;  "
            f"PSA 峰值 {rs['PSA'].max():.2f} gal @ T = {T[np.argmax(rs['PSA'])]:.3f} s")
        res.append(dict(label=label, dir=d, grp=grp, fs=fs, pga=pga, acc=x, **rs))
    return res


def geo_means(results):
    """每个台站(分组)取两个水平分量的几何平均; 返回列表"""
    groups = {}
    for r in results:
        if str(r["dir"]).upper() in ("Z", "U-D", "UD"):
            continue
        groups.setdefault(r.get("grp", ""), []).append(r)
    out = []
    multi = len({r.get("grp", "") for r in results}) > 1
    for g, hor in groups.items():
        if len(hor) != 2:
            continue
        a, b = hor
        gm = {"label": f"{g} 水平几何平均" if multi else "水平几何平均", "is_gm": True,
              "grp": g, "T": a["T"], "pga": np.sqrt(a["pga"] * b["pga"])}
        for k in ("SD", "SV", "PSA", "SA"):
            gm[k] = np.sqrt(a[k] * b[k])
        out.append(gm)
    return out


# ======================================================================
#  GB 50011-2010 (2016年版) 设计反应谱
# ======================================================================
# 水平地震影响系数最大值 αmax: (多遇, 设防, 罕遇)
#   多遇/罕遇: 表 5.1.4-1; 设防地震: 对应基本地震加速度/g × 2.25
GB_ALPHA_MAX = {
    "6度 (0.05g)": (0.04, 0.12, 0.28),
    "7度 (0.10g)": (0.08, 0.23, 0.50),
    "7度 (0.15g)": (0.12, 0.34, 0.72),
    "8度 (0.20g)": (0.16, 0.45, 0.90),
    "8度 (0.30g)": (0.24, 0.68, 1.20),
    "9度 (0.40g)": (0.32, 0.90, 1.40),
}
GB_LEVELS = ["多遇地震", "设防地震", "罕遇地震"]
GB_SITES = ["I0", "I1", "II", "III", "IV"]
GB_GROUPS = ["第一组", "第二组", "第三组"]
# 特征周期 Tg (s): 表 5.1.4-2
GB_TG = {
    "第一组": (0.20, 0.25, 0.35, 0.45, 0.65),
    "第二组": (0.25, 0.30, 0.40, 0.55, 0.75),
    "第三组": (0.30, 0.35, 0.45, 0.65, 0.90),
}


def gb50011_params(intensity, level, site, group, zeta=0.05, vertical=False):
    amax = GB_ALPHA_MAX[intensity][GB_LEVELS.index(level)]
    tg = GB_TG["第一组" if vertical else group][GB_SITES.index(site)]
    if level == "罕遇地震":
        tg += 0.05                                   # 5.1.4 条: 罕遇地震 Tg 增加 0.05 s
    if vertical:
        amax *= 0.65                                 # 5.3.1 / 5.3.4 条: 竖向取水平的 65%
    gamma = 0.9 + (0.05 - zeta) / (0.3 + 6 * zeta)   # 式 5.1.5-1
    eta1 = max(0.0, 0.02 + (0.05 - zeta) / (4 + 32 * zeta))       # 式 5.1.5-2
    eta2 = max(0.55, 1 + (0.05 - zeta) / (0.08 + 1.6 * zeta))     # 式 5.1.5-3
    return dict(amax=amax, tg=tg, gamma=gamma, eta1=eta1, eta2=eta2)


def gb50011_alpha(T, prm):
    """地震影响系数 α(T) (单位 g); T > 6 s 规范未规定, 返回 nan"""
    T = np.asarray(T, dtype=np.float64)
    a, tg, g_, e1, e2 = prm["amax"], prm["tg"], prm["gamma"], prm["eta1"], prm["eta2"]
    out = np.full_like(T, np.nan)
    m = T < 0.1
    out[m] = (0.45 + (e2 - 0.45) * T[m] / 0.1) * a
    m = (T >= 0.1) & (T <= tg)
    out[m] = e2 * a
    m = (T > tg) & (T <= 5 * tg)
    out[m] = (tg / T[m]) ** g_ * e2 * a
    m = (T > 5 * tg) & (T <= 6.0)
    out[m] = (e2 * 0.2 ** g_ - e1 * (T[m] - 5 * tg)) * a
    return out


# ======================================================================
#  GUI
# ======================================================================
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                              "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    root = tk.Tk()
    root.title("地震反应谱 SA 计算")
    root.geometry("1150x960")
    root.minsize(950, 750)
    msgq = queue.Queue()
    state = {"results": None, "gm": None}

    def log(s):
        msgq.put(("log", s))

    # ---------------- 文件
    f_in = ttk.LabelFrame(root, text="① 输入数据（miniSEED，或 K-NET/KiK-net ASCII 分量文件 / 下载压缩包 / 文件夹）")
    f_in.pack(fill="x", padx=10, pady=(10, 4))
    lf = ttk.Frame(f_in)
    lf.pack(side="left", fill="both", expand=True, padx=8, pady=6)
    lb = tk.Listbox(lf, height=6, selectmode="extended", font=("Consolas", 9))
    sbx = ttk.Scrollbar(lf, orient="horizontal", command=lb.xview)
    lb.configure(xscrollcommand=sbx.set)
    lb.pack(fill="both", expand=True)
    sbx.pack(fill="x")
    files = []
    import tempfile
    tmp_root = Path(tempfile.mkdtemp(prefix="knet_"))

    def describe_file(f):
        if is_knet_file(f):
            try:
                h = knet_header(f)
                return (f"[{h['_net']}] {h['_grp']:<14} {h['_comp']:<3} "
                        f"{h.get('Sampling Freq(Hz)', '?'):>6} {h.get('Duration Time(s)', '?'):>4}s  "
                        f"Max {h.get('Max. Acc. (gal)', '?'):>9} gal   "
                        f"{h.get('Origin Time', '?')} JST  M{h.get('Mag.', '?')}   {f}")
            except Exception:
                pass
        return f"[miniSEED] {f}"

    def push(paths):
        def sort_key(f):
            if is_knet_file(f):
                h = knet_header(f)
                return (0, h["_grp"], "NS EW UD".find(h["_comp"]), f)
            return (1, "", 0, f)
        for f in sorted(paths, key=sort_key):
            if f not in files:
                files.append(f)
                lb.insert("end", describe_file(f))

    def pick_stations(knet_files, title):
        """多个台站时弹窗让用户选择"""
        groups = {}
        for f in knet_files:
            try:
                h = knet_header(f)
            except Exception:
                continue
            g = groups.setdefault(h["_grp"], dict(files=[], comps=[], maxacc=0.0, h=h))
            g["files"].append(f)
            g["comps"].append(h["_comp"])
            try:
                g["maxacc"] = max(g["maxacc"], float(h.get("Max. Acc. (gal)", 0)))
            except ValueError:
                pass
        if len(groups) <= 1:
            return knet_files
        order = sorted(groups, key=lambda k: -groups[k]["maxacc"])
        dlg = tk.Toplevel(root)
        dlg.title(title)
        dlg.geometry("760x480")
        dlg.transient(root)
        h0 = groups[order[0]]["h"]
        ttk.Label(dlg, text=f"共 {len(groups)} 个台站（按最大加速度排序）。发震 {h0.get('Origin Time', '?')} JST，"
                            f"M{h0.get('Mag.', '?')}，震源 ({h0.get('Lat.', '?')}, {h0.get('Long.', '?')}) "
                            f"深 {h0.get('Depth. (km)', '?')} km\n按住 Ctrl / Shift 可多选。").pack(
            anchor="w", padx=10, pady=6)
        fr = ttk.Frame(dlg)
        fr.pack(fill="both", expand=True, padx=10)
        sl = tk.Listbox(fr, selectmode="extended", font=("Consolas", 10))
        sb_ = ttk.Scrollbar(fr, orient="vertical", command=sl.yview)
        sl.configure(yscrollcommand=sb_.set)
        sl.pack(side="left", fill="both", expand=True)
        sb_.pack(side="right", fill="y")
        for k in order:
            g = groups[k]
            sl.insert("end", f"{k:<14} 最大加速度 {g['maxacc']:>9.3f} gal   分量 {' '.join(sorted(g['comps'])):<9}"
                             f"  台站 ({g['h'].get('Station Lat.', '?')}, {g['h'].get('Station Long.', '?')})")
        sl.selection_set(0)
        res = []

        def ok():
            for i in sl.curselection():
                res.extend(groups[order[i]]["files"])
            dlg.destroy()
        bb = ttk.Frame(dlg)
        bb.pack(fill="x", padx=10, pady=8)
        ttk.Button(bb, text="全选", command=lambda: sl.selection_set(0, "end")).pack(side="left")
        ttk.Button(bb, text="确定", command=ok).pack(side="right")
        ttk.Button(bb, text="取消", command=dlg.destroy).pack(side="right", padx=6)
        sl.bind("<Double-Button-1>", lambda e: ok())
        dlg.grab_set()
        root.wait_window(dlg)
        return res

    def add_mseed():
        fs = filedialog.askopenfilenames(
            title="选择 miniSEED 文件",
            filetypes=[("miniSEED", "*.mseed *.miniseed *.msd"), ("所有文件", "*.*")])
        push(list(fs))

    def add_knet():
        fs = filedialog.askopenfilenames(
            title="选择 K-NET/KiK-net ASCII 文件或下载压缩包",
            filetypes=[("K-NET/KiK-net", "*.EW *.NS *.UD *.EW1 *.NS1 *.UD1 *.EW2 *.NS2 *.UD2 "
                                         "*.tar.gz *.tgz *.tar *.zip"),
                       ("所有文件", "*.*")])
        if not fs:
            return
        knet = []
        for f in fs:
            if f.lower().endswith((".tar.gz", ".tgz", ".tar", ".zip")):
                try:
                    got = extract_archive(f, tmp_root / Path(f).name.replace(".", "_"))
                    log(f"解压 {Path(f).name}: 找到 {len(got)} 个 K-NET 分量文件")
                    knet += got
                except Exception as e:
                    messagebox.showerror("解压失败", f"{f}\n{e}")
            elif is_knet_file(f):
                knet.append(f)
            else:
                messagebox.showwarning("提示", f"不是 K-NET/KiK-net ASCII 文件：\n{f}")
        if knet:
            push(pick_stations(knet, "选择台站"))

    def add_knet_dir():
        d = filedialog.askdirectory(title="选择包含 K-NET/KiK-net 文件的文件夹")
        if d:
            knet = [str(f) for f in Path(d).rglob("*") if f.is_file() and KNET_EXT.search(f.name)]
            if not knet:
                messagebox.showwarning("提示", "该文件夹中没有 K-NET/KiK-net ASCII 文件")
                return
            push(pick_stations(knet, "选择台站"))

    def remove_sel():
        for i in sorted(lb.curselection(), reverse=True):
            del files[i]
            lb.delete(i)

    def clear():
        files.clear()
        lb.delete(0, "end")

    bf = ttk.Frame(f_in)
    bf.pack(side="right", padx=8)
    for t_, c_ in [("添加 K-NET 数据…", add_knet), ("添加 K-NET 文件夹…", add_knet_dir),
                   ("添加 miniSEED…", add_mseed), ("移除所选", remove_sel), ("清空", clear)]:
        ttk.Button(bf, text=t_, command=c_, width=17).pack(pady=1)

    # ---------------- 参数
    f_p = ttk.LabelFrame(root, text="② 参数")
    f_p.pack(fill="x", padx=10, pady=4)
    V = {k: tk.StringVar(value=v) for k, v in dict(
        cpu="2", zeta="5", tmin="0.01", tmax="10", npts="200",
        taper="5", flo="0.05", fhi="", unit="g", kind="PSA（伪谱加速度）").items()}
    detrend = tk.BooleanVar(value=True)
    use_band = tk.BooleanVar(value=False)
    show_gm = tk.BooleanVar(value=True)
    logy = tk.BooleanVar(value=True)

    def row(parent):
        r = ttk.Frame(parent)
        r.pack(fill="x", padx=8, pady=3)
        return r

    def ent(r, label, key, w=7, after=""):
        ttk.Label(r, text=label).pack(side="left")
        ttk.Entry(r, textvariable=V[key], width=w).pack(side="left", padx=(3, 2))
        if after:
            ttk.Label(r, text=after).pack(side="left", padx=(0, 14))

    r1 = row(f_p)
    ent(r1, "阻尼比", "zeta", 5, "%")
    ent(r1, "周期范围", "tmin", 6, "~")
    ent(r1, "", "tmax", 6, "s")
    ent(r1, "周期点数", "npts", 6, "（对数等分，另含标准周期点）")
    r2 = row(f_p)
    ent(r2, "miniSEED 换算系数", "cpu", 6, "counts/µg（TitanSMA 为 2；K-NET 自动读取文件头）")
    r3 = row(f_p)
    ttk.Checkbutton(r3, text="去均值 + 去线性趋势", variable=detrend).pack(side="left")
    ent(r3, "   两端余弦尖灭", "taper", 5, "%")
    ttk.Checkbutton(r3, text="带通滤波（4 阶 Butterworth 零相位）", variable=use_band).pack(side="left")
    ent(r3, "  ", "flo", 6, "~")
    ent(r3, "", "fhi", 6, "Hz（留空=不设该端）")
    r4 = row(f_p)
    ttk.Label(r4, text="显示").pack(side="left")
    ttk.Combobox(r4, textvariable=V["kind"], width=20, state="readonly",
                 values=["PSA（伪谱加速度）", "SA（绝对加速度）", "SV（相对速度）",
                         "SD（相对位移）"]).pack(side="left", padx=4)
    ttk.Label(r4, text="加速度单位").pack(side="left", padx=(12, 0))
    ttk.Combobox(r4, textvariable=V["unit"], width=8, state="readonly",
                 values=["g", "gal", "m/s²"]).pack(side="left", padx=4)
    ttk.Checkbutton(r4, text="显示水平两分量几何平均", variable=show_gm,
                    command=lambda: redraw()).pack(side="left", padx=12)
    ttk.Checkbutton(r4, text="纵轴对数", variable=logy,
                    command=lambda: redraw()).pack(side="left")
    for k in ("kind", "unit"):
        V[k].trace_add("write", lambda *a: redraw())

    # ---------------- 规范设计反应谱
    f_c = ttk.LabelFrame(root, text="③ 规范设计反应谱对比（GB 50011-2010，2016年版，5.1.4、5.1.5条；按上面的阻尼比修正）")
    f_c.pack(fill="x", padx=10, pady=4)
    show_code = tk.BooleanVar(value=True)
    show_vert = tk.BooleanVar(value=False)
    C = dict(inten=tk.StringVar(value="8度 (0.20g)"), level=tk.StringVar(value="设防地震"),
             site=tk.StringVar(value="II"), group=tk.StringVar(value="第一组"))
    rc = ttk.Frame(f_c)
    rc.pack(fill="x", padx=8, pady=(4, 2))
    ttk.Checkbutton(rc, text="显示规范谱", variable=show_code,
                    command=lambda: redraw()).pack(side="left", padx=(0, 12))
    for lab, k, vals, w_ in [("设防烈度", "inten", list(GB_ALPHA_MAX), 12),
                             ("地震水准", "level", GB_LEVELS, 9),
                             ("场地类别", "site", GB_SITES, 5),
                             ("设计地震分组", "group", GB_GROUPS, 7)]:
        ttk.Label(rc, text=lab).pack(side="left")
        ttk.Combobox(rc, textvariable=C[k], values=vals, width=w_,
                     state="readonly").pack(side="left", padx=(3, 12))
        C[k].trace_add("write", lambda *a: redraw())
    ttk.Checkbutton(rc, text="同时画竖向谱（0.65×，Tg 按第一组）", variable=show_vert,
                    command=lambda: redraw()).pack(side="left")
    code_info = tk.StringVar(value="")
    ttk.Label(f_c, textvariable=code_info, foreground="#a01010").pack(fill="x", padx=8, pady=(0, 4))
    V["zeta"].trace_add("write", lambda *a: redraw())

    def cur_zeta():
        if state.get("zeta") is not None:
            return state["zeta"]
        try:
            z = float(V["zeta"].get()) / 100
            return z if 0 < z < 1 else 0.05
        except ValueError:
            return 0.05

    def code_curves():
        """返回 [(标签, 参数)]，未勾选时为空"""
        if not show_code.get():
            return []
        z = cur_zeta()
        args = (C["inten"].get(), C["level"].get(), C["site"].get(), C["group"].get(), z)
        h = gb50011_params(*args)
        out = [(f"GB 50011 水平 {args[0]} {args[1]} {args[2]}类 {args[3]}", h)]
        if show_vert.get():
            out.append((f"GB 50011 竖向 {args[0]} {args[1]}", gb50011_params(*args, vertical=True)))
        code_info.set(f"水平：αmax = {h['amax']:.3f} g，Tg = {h['tg']:.2f} s，γ = {h['gamma']:.3f}，"
                      f"η1 = {h['eta1']:.4f}，η2 = {h['eta2']:.3f}（阻尼比 {z*100:g}%）；"
                      f"平台段 η2·αmax = {h['eta2']*h['amax']:.3f} g，T=0 时 0.45αmax = {0.45*h['amax']:.3f} g")
        return out

    # ---------------- 按钮
    f_a = ttk.Frame(root)
    f_a.pack(fill="x", padx=10, pady=4)
    pb = ttk.Progressbar(f_a, mode="indeterminate", length=150)

    def get_params():
        try:
            p = dict(cpu=float(V["cpu"].get()), zeta=float(V["zeta"].get()) / 100,
                     tmin=float(V["tmin"].get()), tmax=float(V["tmax"].get()),
                     npts=int(V["npts"].get()), taper=float(V["taper"].get() or 0),
                     detrend=detrend.get())
            assert 0 < p["zeta"] < 1 and 0 < p["tmin"] < p["tmax"] and p["npts"] >= 10
            band = None
            if use_band.get():
                lo = float(V["flo"].get()) if V["flo"].get().strip() else None
                hi = float(V["fhi"].get()) if V["fhi"].get().strip() else None
                band = (lo, hi)
            p["band"] = band
            return p
        except Exception:
            messagebox.showerror("参数错误", "请检查参数：阻尼比 0~100%，0 < 最短周期 < 最长周期，点数 ≥ 10")
            return None

    def on_calc():
        if not files:
            messagebox.showwarning("提示", "请先添加文件")
            return
        p = get_params()
        if p is None:
            return
        btn.configure(state="disabled")
        pb.pack(side="left", padx=10)
        pb.start(12)
        txt.delete("1.0", "end")

        def worker():
            try:
                recs = []
                for f in files:
                    recs += load_any(f, p["cpu"])
                log(f"阻尼比 {p['zeta']*100:g}%，周期 {p['tmin']:g}~{p['tmax']:g} s，"
                    f"滤波 {p['band'] if p['band'] else '无'}")
                res = compute_all(recs, p, log)
                msgq.put(("done", res))
            except Exception as e:
                log("错误: " + str(e))
                log(traceback.format_exc())
                msgq.put(("error", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    btn = ttk.Button(f_a, text="计算反应谱", command=on_calc)
    btn.pack(side="left")

    def unit_factor():
        return {"g": 1 / G, "gal": 1.0, "m/s²": 0.01}[V["unit"].get()]

    def series():
        out = list(state["results"] or [])
        if out and show_gm.get():
            out = out + geo_means(out)
        return out

    def on_csv():
        if not state["results"]:
            messagebox.showwarning("提示", "请先计算")
            return
        f = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")], initialfile="response_spectrum.csv")
        if not f:
            return
        rs = series()
        uf, un = unit_factor(), V["unit"].get()
        with open(f, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp)
            w.writerow([f"阻尼比 {float(V['zeta'].get()):g}%", "SD 单位 cm, SV 单位 cm/s",
                        f"PSA/SA 单位 {un}"])
            cds = code_curves()
            head = ["T(s)"]
            for r in rs:
                head += [f"{r['label']} PSA", f"{r['label']} SA",
                         f"{r['label']} SV", f"{r['label']} SD"]
            head += [f"{lab} ({un})" for lab, _ in cds]
            w.writerow(head)
            w.writerow(["PGA"] + sum([[f"{r['pga']*uf:.6g}", f"{r['pga']*uf:.6g}", "", ""]
                                      for r in rs], []))
            codev = [gb50011_alpha(rs[0]["T"], prm) * G * uf for _, prm in cds]
            for i, t in enumerate(rs[0]["T"]):
                line = [f"{t:.5g}"]
                for r in rs:
                    line += [f"{r['PSA'][i]*uf:.6g}", f"{r['SA'][i]*uf:.6g}",
                             f"{r['SV'][i]:.6g}", f"{r['SD'][i]:.6g}"]
                line += ["" if np.isnan(c[i]) else f"{c[i]:.6g}" for c in codev]
                w.writerow(line)
        log(f"已导出 {f}")
        messagebox.showinfo("完成", f"已导出：\n{f}")

    def on_png():
        if not state["results"]:
            messagebox.showwarning("提示", "请先计算")
            return
        f = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png")], initialfile="response_spectrum.png")
        if f:
            fig.savefig(f, dpi=150)
            log(f"已保存图片 {f}")

    ttk.Button(f_a, text="导出 CSV…", command=on_csv).pack(side="left", padx=6)
    ttk.Button(f_a, text="保存图片…", command=on_png).pack(side="left")

    # ---------------- 结果区: 图 + 表/日志
    pw = ttk.PanedWindow(root, orient="vertical")
    pw.pack(fill="both", expand=True, padx=10, pady=(4, 10))
    f_fig = ttk.Frame(pw)
    nb = ttk.Notebook(pw)
    pw.add(f_fig, weight=3)
    pw.add(nb, weight=1)

    fig = Figure(figsize=(9, 4.2), dpi=96)
    ax_ts = fig.add_subplot(1, 3, 1)
    ax = fig.add_subplot(1, 3, (2, 3))
    canvas = FigureCanvasTkAgg(fig, master=f_fig)
    tb = NavigationToolbar2Tk(canvas, f_fig, pack_toolbar=False)
    tb.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)

    tab_tab, tab_log = ttk.Frame(nb), ttk.Frame(nb)
    nb.add(tab_tab, text="  标准周期 SA 表  ")
    nb.add(tab_log, text="  日志  ")
    table = ScrolledText(tab_tab, height=8, font=("Consolas", 9), wrap="none")
    table.pack(fill="both", expand=True, padx=4, pady=4)
    txt = ScrolledText(tab_log, height=8, font=("Consolas", 9), wrap="none")
    txt.pack(fill="both", expand=True, padx=4, pady=4)

    def redraw():
        codes = code_curves()
        if not state["results"] and not codes:
            return
        rs = series() if state["results"] else []
        uf, un = unit_factor(), V["unit"].get()
        key = V["kind"].get().split("（")[0]
        key = key if key in ("PSA", "SA", "SV", "SD") else "PSA"
        f_, ylab = (uf, f"{key} ({un})") if key in ("PSA", "SA") else \
            ((1.0, "SV (cm/s)") if key == "SV" else (1.0, "SD (cm)"))

        ax_ts.clear()
        recs = state["results"] or []
        off = 0
        span = max([np.abs(r["acc"]).max() for r in recs] or [1]) * 2.2
        for r in recs:
            t = np.arange(len(r["acc"])) / r["fs"]
            ax_ts.plot(t, r["acc"] * uf - off * uf, lw=0.4)
            ax_ts.text(t[-1], -off * uf, " " + r["label"].split(".")[-1], fontsize=7, va="center")
            off += span
        ax_ts.set_title("加速度时程（预处理后）", fontsize=9)
        ax_ts.set_xlabel("时间 (s)")
        ax_ts.set_yticks([])

        ax.clear()
        for r in rs:
            style = dict(lw=2.4) if r.get("is_gm") else dict(lw=1.0)
            if r.get("is_gm") and len([x for x in rs if x.get("is_gm")]) == 1:
                style["color"] = "k"
            ax.plot(r["T"], r[key] * f_, label=f"{r['label']}  (PGA {r['pga']*uf:.4g} {un})", **style)
        if key in ("PSA", "SA") and codes:
            try:
                t_lo = float(V["tmin"].get())
            except ValueError:
                t_lo = 0.01
            Tc = np.logspace(np.log10(max(t_lo, 1e-3)), np.log10(6.0), 400)
            for (lab, prm), col in zip(codes, ("#d62728", "#9467bd")):
                ax.plot(Tc, gb50011_alpha(Tc, prm) * G * uf, ls="--", lw=2, color=col, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log" if logy.get() else "linear")
        ax.set_xlabel("周期 T (s)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{key} 反应谱，阻尼比 {cur_zeta()*100:g}%", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        canvas.draw_idle()

        # 标准周期表
        table.delete("1.0", "end")
        cds = codes if key in ("PSA", "SA") else []
        hdr = (f"{'T (s)':>8}" + "".join(f"{r['label'][:22]:>24}" for r in rs)
               + "".join(f"{('规范' + lab[9:11]):>22}" for lab, _ in cds)
               + "".join(f"{('实测/规范'):>18}" for _ in cds[:1] if rs))
        table.insert("end", f"{key}，单位 {ylab.split('(')[-1].rstrip(')')}"
                            + ("；实测/规范 取最后一列实测(几何平均或最后一个分量)与水平规范谱之比" if cds and rs else "")
                            + f"\n{hdr}\n")
        T = rs[0]["T"] if rs else None
        if key in ("PSA", "SA"):
            line = f"{'PGA/T=0':>8}" + "".join(f"{r['pga']*uf:>24.5g}" for r in rs)
            v0 = [gb50011_alpha([0.0], prm)[0] * G * uf for _, prm in cds]
            line += "".join(f"{v:>22.5g}" for v in v0)
            if cds and rs:
                line += f"{rs[-1]['pga'] * uf / v0[0]:>18.3f}"
            table.insert("end", line + "\n")
        for t in STD_PERIODS:
            line = f"{t:>8.3g}"
            meas = None
            if rs:
                i = int(np.argmin(np.abs(T - t)))
                if abs(T[i] - t) / t > 1e-6:
                    continue
                line += "".join(f"{r[key][i]*f_:>24.5g}" for r in rs)
                meas = rs[-1][key][i] * f_
            vals = [gb50011_alpha([t], prm)[0] * G * uf for _, prm in cds]
            line += "".join(f"{v:>22.5g}" for v in vals)
            if cds and rs:
                line += f"{meas / vals[0]:>18.3f}" if vals[0] == vals[0] else f"{'':>18}"
            table.insert("end", line + "\n")

    def poll():
        try:
            while True:
                k, s = msgq.get_nowait()
                if k == "log":
                    txt.insert("end", s + "\n")
                    txt.see("end")
                else:
                    pb.stop()
                    pb.pack_forget()
                    btn.configure(state="normal")
                    if k == "done":
                        state["results"] = s
                        try:
                            state["zeta"] = float(V["zeta"].get()) / 100
                        except ValueError:
                            state["zeta"] = 0.05
                        redraw()
                        nb.select(tab_tab)
                    else:
                        nb.select(tab_log)
                        messagebox.showerror("出错", s)
        except queue.Empty:
            pass
        root.after(100, poll)

    poll()
    root.mainloop()


if __name__ == "__main__":
    run_gui()
