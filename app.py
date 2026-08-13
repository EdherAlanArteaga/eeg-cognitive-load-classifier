"""
EEG Dynamic Spectral Persistence
Edher Alan Arteaga Marroquin · 2026

Framework v3 — Dynamic Spectral Persistence. Replaces v1 (quaternionic R, C)
and v2 (C_dyn = λ₂²·G_pure).

Observables, from "Dynamic Spectral Persistence in Heterogeneous Networks:
From Hub Blindness to Heat Kernel Identity":

    V_i      = Σ_{k≥2} v_k(i)²/λ_k          spectral visibility
    V_i      = ∫₀^∞ R_i^⊥(t) dt             exact identity (Theorem 2.1)
    R_i^⊥(t) = Σ_{k≥2} e^{-tλ_k} v_k(i)²    transverse returnability
    τ_i      = M₂/M₁                        persistence timescale
    τ̃_i      = λ₂ · τ_i                     normalized timescale

Validated before publishing:
    heat kernel identity          error 2e-16 … 6e-14   (paper: < 5e-15)
    closed-form R_hub(t)          error 0.00e+00        (paper: 4.34e-15)
    spectral blindness of the hub 0.00e+00 exact
    anti-centrality Star S_12     Pearson −1.0000       (paper: −1.000)
    EEG recurrence networks       Spearman(k,V) −0.90 … −0.99
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample_poly
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from fractions import Fraction
import tempfile, os, re

st.set_page_config(page_title="EEG Dynamic Spectral Persistence",
                   page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#f5f5f0;color:#1a1a1a;}
.stApp{background:#f5f5f0;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;color:#1a1a1a;}
.card{background:#fff;border:1px solid #d0d0c8;border-radius:6px;padding:16px;text-align:center;margin:4px 0;}
.lbl{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#777;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;}
.val{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;color:#1a1a1a;}
.pos{color:#1a6b1a;}.neg{color:#c0392b;}
.hero{background:#fff;border:2px solid #3a7bd5;border-radius:8px;padding:22px;text-align:center;}
.hero-t{font-family:'IBM Plex Mono',monospace;font-size:21px;font-weight:600;margin:8px 0 4px;color:#3a7bd5;}
.hero-s{font-size:13px;color:#555;line-height:1.6;margin-top:8px;}
.hero-l{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#888;text-transform:uppercase;letter-spacing:2px;}
.info{background:#fff;border-left:3px solid #3a7bd5;padding:14px 18px;border-radius:0 6px 6px 0;margin:14px 0;font-size:14px;line-height:1.7;color:#333;}
.warn{background:#fffbf0;border-left:3px solid #e6a817;padding:14px 18px;border-radius:0 6px 6px 0;margin:14px 0;font-size:13px;line-height:1.6;color:#555;}
.ok{background:#f0f8f0;border-left:3px solid #2d8a2d;padding:14px 18px;border-radius:0 6px 6px 0;margin:14px 0;font-size:13px;line-height:1.6;color:#2a4a2a;}
.sec{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:2px;margin:26px 0 10px;border-bottom:1px solid #d0d0c8;padding-bottom:5px;}
.foot{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#aaa;text-align:center;margin-top:48px;border-top:1px solid #d0d0c8;padding-top:14px;}
</style>
""", unsafe_allow_html=True)

FS_OBJETIVO = 100.0
PREF_CANAL = ['fz', 'cz', 'f3', 'c3', 'f4', 'c4', 'pz', 'fp1']
RANGO_MAX = 1000.0     # λmax/λ₂ — por encima, τ̃ deja de converger (barrido BA/ER)
WIN_S, STEP_S, MAX_WINS = 4, 2, 40
EMBED_D, N_MAX, PCT = 4, 40, 20


# ── Canales ──────────────────────────────────────────────────────────────────
def normalizar_label(s):
    s = re.sub(r'\b(eeg|ref|le|re|a1|a2|avg|linked|ears)\b', ' ', s.lower().strip())
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    t = s.split()
    return t[0] if t else s


def elegir_canal(labels):
    norm = [normalizar_label(l) for l in labels]
    for p in PREF_CANAL:
        if p in norm:
            i = norm.index(p)
            return i, labels[i], "by name"
    i = len(labels) // 2
    return i, labels[i], "fallback — unrecognized label"


# ── Preproceso ───────────────────────────────────────────────────────────────
def preprocesar(x, fs, fmin=0.5, fmax=40.0):
    nyq = fs / 2.0
    hi, lo = min(fmax, nyq * .95), max(fmin, .1)
    if hi <= lo:
        return (x - x.mean()) / (x.std() + 1e-10)
    b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
    y = filtfilt(b, a, x)
    return (y - y.mean()) / (y.std() + 1e-10)


def remuestrear(x, fs, fs_obj=FS_OBJETIVO):
    """resample_poly, not decimate: decimate only accepts integer factors and
    250 Hz would give 125 instead of 100, breaking comparability across datasets."""
    if abs(fs - fs_obj) < 1e-6:
        return x, fs
    f = Fraction(fs_obj / fs).limit_denominator(1000)
    if f.numerator < 1 or f.denominator < 1:
        return x, fs
    return resample_poly(x, f.numerator, f.denominator), fs * f.numerator / f.denominator


# ── Núcleo v3 ────────────────────────────────────────────────────────────────
def red_recurrencia(x, d=EMBED_D, n_max=N_MAX, pct=PCT):
    if x.std() < 1e-10:
        return None
    x = np.clip((x - x.mean()) / (x.std() + 1e-10), -4, 4)
    pts = np.array([x[i:i + d] for i in range(len(x) - d + 1)])
    if len(pts) < 6:
        return None
    if len(pts) > n_max:
        pts = pts[np.linspace(0, len(pts) - 1, n_max).astype(int)]
    D = cdist(pts, pts); D0 = D.copy()
    np.fill_diagonal(D, np.inf)
    eps = np.quantile(D[D < np.inf], pct / 100)
    A = (D < eps).astype(float)
    np.fill_diagonal(A, 0)
    for i in np.where(A.sum(1) == 0)[0]:
        j = np.argsort(D0[i])[1]
        A[i, j] = A[j, i] = 1.0
    return A


def dsp(A):
    """V_i, τ_i, τ̃_i and the Laplacian spectrum of a network."""
    A = np.array(A, float); np.fill_diagonal(A, 0)
    N = A.shape[0]; deg = A.sum(1)
    ev, evec = eigh(np.diag(deg) - A)
    k0 = next((k for k in range(1, N) if ev[k] > 1e-8), None)
    if k0 is None:
        return None
    M1 = np.zeros(N); M2 = np.zeros(N)
    for k in range(k0, N):
        if ev[k] < 1e-10:
            continue
        v2 = evec[:, k] ** 2
        M1 += v2 / ev[k]; M2 += v2 / ev[k] ** 2
    lam2 = ev[k0]
    tau = np.where(M1 > 1e-14, M2 / M1, 0.)
    return dict(V=M1, tau=tau, tau_tilde=lam2 * tau, lambda2=lam2,
                lambda_max=ev[-1], rango=ev[-1] / lam2, deg=deg,
                ev=ev, evec=evec, k0=k0, N=N)


def returnability(d, i, t):
    """R_i^⊥(t) = Σ_{k≥2} e^{-tλ_k} v_k(i)²"""
    return float(np.sum(np.exp(-d['ev'][d['k0']:] * t) * d['evec'][i, d['k0']:] ** 2))


def analizar(x, fs):
    """Iterates over windows and aggregates the v3 observables for the full recording."""
    win, step = int(WIN_S * fs), int(STEP_S * fs)
    if len(x) < win + step:
        return None

    tt_m, tt_v, l2, Vh, rho_s, rho_p, rangos = [], [], [], [], [], [], []
    descartadas = 0
    ejemplo = None
    np.random.seed(42)

    for n, s in enumerate(range(0, len(x) - win + 1, step)):
        if n >= MAX_WINS:
            break
        A = red_recurrencia(x[s:s + win])
        if A is None:
            continue
        d = dsp(A)
        if d is None:
            continue
        if d['rango'] > RANGO_MAX:      # τ̃ does not converge, see BA/ER sweep
            descartadas += 1
            continue

        rangos.append(d['rango'])
        tt_m.append(d['tau_tilde'].mean())
        tt_v.append(d['tau_tilde'].var())
        l2.append(d['lambda2'])
        Vh.append(d['V'].max() / (d['V'].min() + 1e-12))
        if np.std(d['deg']) > 1e-12:
            rho_s.append(spearmanr(d['deg'], d['V'])[0])
            rho_p.append(pearsonr(d['deg'], d['V'])[0])
        if ejemplo is None:
            ejemplo = d

    if len(tt_m) < 5:
        return None

    tt_m = np.array(tt_m)
    return dict(tau_tilde=float(tt_m.mean()),
                sigma_tau=float(tt_m.std()),
                disp_tau=float(np.mean(tt_v)),
                lambda2=float(np.mean(l2)),
                heterog_V=float(np.mean(Vh)),
                rho_spearman=float(np.mean(rho_s)) if rho_s else np.nan,
                rho_pearson=float(np.mean(rho_p)) if rho_p else np.nan,
                rango=float(np.mean(rangos)), rango_max=float(np.max(rangos)),
                n_win=len(tt_m), descartadas=descartadas, ejemplo=ejemplo)


def verificar_identidad(d, n_nodos=5):
    """Theorem 2.1 on the subject's real network. Numerical quadrature integration."""
    from scipy.integrate import quad
    errs = []
    for i in range(min(n_nodos, d['N'])):
        integral, _ = quad(lambda t: returnability(d, i, t), 0, np.inf, limit=200)
        errs.append(abs(d['V'][i] - integral))
    return float(np.max(errs))


# ── EDF ──────────────────────────────────────────────────────────────────────
def cargar_edf(b):
    try:
        import pyedflib
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as t:
            t.write(b); ruta = t.name
        f = pyedflib.EdfReader(ruta)
        labels = f.getSignalLabels()
        n_ch = min(f.signals_in_file, 64)
        fss = [f.getSampleFrequency(i) for i in range(n_ch)]
        fs = fss[0]
        val = [i for i in range(n_ch) if abs(fss[i] - fs) < 1e-6]
        n = min(f.getNSamples()[i] for i in val)
        sig = np.zeros((len(val), n))
        for k, i in enumerate(val):
            sig[k, :] = f.readSignal(i, n=n)
        labels = [labels[i] for i in val]
        f.close(); os.unlink(ruta)
        return sig, float(fs), labels, n / fs, None
    except Exception as e:
        return None, None, None, None, str(e)


# ── Gráficas ─────────────────────────────────────────────────────────────────
def fig_returnability(dr, dt_):
    """The signature of v3: timescale separation between persistent and transient nodes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.patch.set_facecolor('#f5f5f0')
    ts = np.logspace(-2, 1.2, 160)
    for ax, d, titulo, col in [(axes[0], dr, 'Rest', '#666'),
                               (axes[1], dt_, 'Task', '#3a7bd5')]:
        ax.set_facecolor('#fff')
        if d is None:
            ax.text(.5, .5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='#aaa', fontfamily='monospace')
            continue
        i_alto = int(np.argmax(d['V']))
        i_bajo = int(np.argmin(d['V']))
        ax.plot(ts, [returnability(d, i_alto, t) for t in ts], color=col, lw=2.2,
                label=f'persistent node  V={d["V"][i_alto]:.2f}  k={d["deg"][i_alto]:.0f}')
        ax.plot(ts, [returnability(d, i_bajo, t) for t in ts], color=col, lw=2.2,
                ls='--', label=f'transient node  V={d["V"][i_bajo]:.2f}  k={d["deg"][i_bajo]:.0f}')
        ax.set_xscale('log')
        ax.set_xlabel('t', color='#555', fontfamily='monospace', fontsize=10)
        ax.set_ylabel('R$_i^⊥$(t)', color='#555', fontfamily='monospace', fontsize=10)
        ax.set_title(f'Returnability — {titulo}', fontfamily='monospace',
                     fontsize=10.5, color='#1a1a1a')
        ax.tick_params(colors='#777')
        for sp in ax.spines.values():
            sp.set_color('#d0d0c8')
        ax.grid(alpha=.22, color='#ccc')
        ax.legend(facecolor='#fff', edgecolor='#d0d0c8', labelcolor='#333', fontsize=8)
    plt.tight_layout()
    return fig


def fig_anticentralidad(d, titulo):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    fig.patch.set_facecolor('#f5f5f0'); ax.set_facecolor('#fff')
    ax.scatter(d['deg'], d['V'], s=70, color='#3a7bd5', alpha=.75,
               edgecolors='white', lw=1.2)
    if np.std(d['deg']) > 1e-12:
        rs = spearmanr(d['deg'], d['V'])[0]
        rp = pearsonr(d['deg'], d['V'])[0]
        z = np.polyfit(d['deg'], d['V'], 1)
        xs = np.linspace(d['deg'].min(), d['deg'].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color='#c0392b', ls='--', lw=1.6)
        ax.set_title(f'{titulo}  ·  Spearman {rs:+.3f}  Pearson {rp:+.3f}',
                     fontfamily='monospace', fontsize=10, color='#1a1a1a')
    ax.set_xlabel('node degree  k', color='#555', fontfamily='monospace', fontsize=10)
    ax.set_ylabel('spectral visibility  V', color='#555',
                  fontfamily='monospace', fontsize=10)
    ax.tick_params(colors='#777')
    for sp in ax.spines.values():
        sp.set_color('#d0d0c8')
    ax.grid(alpha=.22, color='#ccc')
    plt.tight_layout()
    return fig


def fig_barras(r, t):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    fig.patch.set_facecolor('#f5f5f0')
    campos = [('tau_tilde', '⟨τ̃⟩  persistence'),
              ('disp_tau', 'Var(τ̃)  heterogeneity'),
              ('lambda2', 'λ₂  connectivity'),
              ('heterog_V', 'V_max/V_min')]
    for ax, (k, tit) in zip(axes, campos):
        ax.set_facecolor('#fff')
        v = [r[k], t[k]]
        bars = ax.bar(['Rest', 'Task'], v, color=['#777', '#3a7bd5'],
                      alpha=.88, edgecolor='white', lw=1.5)
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(v) * .03,
                    f'{x:.4f}', ha='center', va='bottom',
                    fontfamily='monospace', fontsize=8, color='#1a1a1a')
        ax.set_title(f'{tit}\nΔ = {t[k]-r[k]:+.4f}', fontfamily='monospace',
                     fontsize=8.5, color='#1a1a1a')
        ax.tick_params(colors='#777', labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color('#d0d0c8')
        ax.grid(alpha=.18, color='#ccc', axis='y'); ax.margins(y=.2)
    plt.tight_layout()
    return fig


# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 EEG Dynamic Spectral Persistence")
st.markdown("**Edher Alan Arteaga Marroquin · 2026**")
st.markdown("---")

st.markdown("""
<div class="info">
Analyze <strong>any pair of EDF files</strong> — rest and task — with the
<strong>Dynamic Spectral Persistence</strong> framework.<br><br>
Every 4-second window is turned into a recurrence network in phase space.
From that network's Laplacian we extract the <strong>spectral visibility</strong>
V<sub>i</sub> = Σ<sub>k≥2</sub> v<sub>k</sub>(i)²/λ<sub>k</sub>, which by Theorem 2.1
is exactly the integral of the heat kernel returnability, and the
<strong>persistence timescale</strong> τ̃ = λ₂·M₂/M₁.<br><br>
The question is not which nodes carry the most traffic, but
<strong>which ones sustain energy</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Rest")
    st.caption("Baseline state.")
    f_rep = st.file_uploader("rep", type=['edf'], key='rep', label_visibility='collapsed')
with c2:
    st.markdown("#### Task")
    st.caption("During cognitive demand.")
    f_tar = st.file_uploader("tar", type=['edf'], key='tar', label_visibility='collapsed')

with st.expander("Options"):
    canal_manual = st.text_input("Force channel by name (e.g. Fz)", value="")
    ver_identidad = st.checkbox(
        "Verify the heat kernel identity on this signal", value=True,
        help="Checks V_i = ∫₀^∞ R_i^⊥(t)dt by numerical quadrature. "
             "Should give error < 1e-13. Takes a few seconds.")

st.markdown("---")

if f_rep and f_tar:
    with st.spinner('Building recurrence networks and diagonalizing Laplacians…'):
        sig_r, fs_r, lab_r, dur_r, e1 = cargar_edf(f_rep.read())
        sig_t, fs_t, lab_t, dur_t, e2 = cargar_edf(f_tar.read())
        if sig_r is None or sig_t is None:
            st.error(f"Could not read the EDF file. {e1 or e2}")
            st.stop()

        if canal_manual.strip():
            obj = normalizar_label(canal_manual)
            nr = [normalizar_label(l) for l in lab_r]
            if obj in nr:
                idx_r, nom_r, via = nr.index(obj), lab_r[nr.index(obj)], "forced"
            else:
                idx_r, nom_r, via = elegir_canal(lab_r)
                st.warning(f"'{canal_manual}' is not in the file. Using {nom_r}.")
        else:
            idx_r, nom_r, via = elegir_canal(lab_r)

        nt = [normalizar_label(l) for l in lab_t]
        obj_t = normalizar_label(nom_r)
        idx_t = nt.index(obj_t) if obj_t in nt else min(idx_r, len(lab_t) - 1)

        x_r, fsd_r = remuestrear(preprocesar(sig_r[idx_r], fs_r), fs_r)
        x_t, fsd_t = remuestrear(preprocesar(sig_t[idx_t], fs_t), fs_t)
        R = analizar(x_r, fsd_r)
        T = analizar(x_t, fsd_t)

    if R is None or T is None:
        st.error("Insufficient signal: at least 5 valid 4-second windows are needed.")
        st.stop()

    dtt = T['tau_tilde'] - R['tau_tilde']
    dl2 = T['lambda2'] - R['lambda2']
    dvar = T['disp_tau'] - R['disp_tau']

    # ── Resultado ────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Result</div>', unsafe_allow_html=True)
    if dtt > 0:
        titulo = "The network sustains energy longer during the task"
        detalle = ("The persistence timescale went up. The Laplacian's slow modes "
                   "accumulate more modal energy: the dynamics became "
                   "more persistent.")
    else:
        titulo = "The network dissipates energy faster during the task"
        detalle = ("The persistence timescale went down. Energy disperses "
                   "sooner: the dynamics became more transient.")
    st.markdown(f"""
    <div class="hero">
        <div class="hero-l">Dynamic Spectral Persistence · channel {nom_r}</div>
        <div class="hero-t">{titulo}</div>
        <div class="hero-s">
            Δ⟨τ̃⟩ = {dtt:+.5f} &nbsp;·&nbsp; Δλ₂ = {dl2:+.5f}
            &nbsp;·&nbsp; ΔVar(τ̃) = {dvar:+.5f}<br>{detalle}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Observables ──────────────────────────────────────────────────────
    st.markdown('<div class="sec">v3 observables</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    for col, lab, v in zip([k1, k2, k3, k4],
                           ['⟨τ̃⟩ rest', '⟨τ̃⟩ task', 'λ₂ rest', 'λ₂ task'],
                           [R['tau_tilde'], T['tau_tilde'], R['lambda2'], T['lambda2']]):
        col.markdown(f'<div class="card"><div class="lbl">{lab}</div>'
                     f'<div class="val">{v:.5f}</div></div>', unsafe_allow_html=True)

    k5, k6, k7, k8 = st.columns(4)
    for col, lab, v in zip([k5, k6, k7, k8],
                           ['Var(τ̃) rest', 'Var(τ̃) task',
                            'V_max/V_min rest', 'V_max/V_min task'],
                           [R['disp_tau'], T['disp_tau'],
                            R['heterog_V'], T['heterog_V']]):
        col.markdown(f'<div class="card"><div class="lbl">{lab}</div>'
                     f'<div class="val">{v:.4f}</div></div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    for col, lab, v in zip([d1, d2, d3],
                           ['Δ⟨τ̃⟩', 'Δλ₂', 'ΔVar(τ̃)'], [dtt, dl2, dvar]):
        col.markdown(f'<div class="card"><div class="lbl">{lab}</div>'
                     f'<div class="val {"pos" if v>0 else "neg"}">{v:+.5f}</div>'
                     f'</div>', unsafe_allow_html=True)

    # ── Controles del marco ──────────────────────────────────────────────
    st.markdown('<div class="sec">Framework controls</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    a1.markdown(f'<div class="card"><div class="lbl">ρ(k,V) Spearman rest</div>'
                f'<div class="val">{R["rho_spearman"]:+.3f}</div></div>',
                unsafe_allow_html=True)
    a2.markdown(f'<div class="card"><div class="lbl">ρ(k,V) Spearman task</div>'
                f'<div class="val">{T["rho_spearman"]:+.3f}</div></div>',
                unsafe_allow_html=True)
    a3.markdown(f'<div class="card"><div class="lbl">max λmax/λ₂ range</div>'
                f'<div class="val">{max(R["rango_max"], T["rango_max"]):.0f}</div>'
                f'</div>', unsafe_allow_html=True)

    peor_rho = max(R['rho_spearman'], T['rho_spearman'])
    if peor_rho < -0.7:
        st.markdown(f"""<div class="ok">
        <strong>Anti-centrality confirmed.</strong> Spearman(k, V) =
        {R['rho_spearman']:+.3f} at rest and {T['rho_spearman']:+.3f} during the task.
        This signal's recurrence networks behave like the heterogeneous
        networks in the paper (reference: −0.925 to −0.974 in ER graphs).
        High-degree nodes are the ones that sustain <em>least</em> persistence.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="warn">
        <strong>Weak anti-centrality</strong> (Spearman {peor_rho:+.3f}, expected
        &lt; −0.7). This signal's recurrence network is more degree-homogeneous
        than usual — typical of highly periodic signals. V and τ̃ are still
        valid, but the contrast between persistent and transient nodes is smaller.
        </div>""", unsafe_allow_html=True)

    desc = R['descartadas'] + T['descartadas']
    if desc:
        st.markdown(f"""<div class="warn">{desc} window(s) were discarded with
        λmax/λ₂ &gt; {RANGO_MAX:.0f}. In the BA/ER network sweep, τ̃ stopped
        converging above that range: λ₂ collapses and M₂/M₁ explodes.
        </div>""", unsafe_allow_html=True)

    if ver_identidad:
        with st.spinner('Verifying V_i = ∫₀^∞ R_i^⊥(t)dt by quadrature…'):
            er = verificar_identidad(R['ejemplo'])
            et = verificar_identidad(T['ejemplo'])
        peor = max(er, et)
        clase = "ok" if peor < 1e-11 else "warn"
        st.markdown(f"""<div class="{clase}">
        <strong>Heat kernel identity (Theorem 2.1).</strong>
        Maximum error |V<sub>i</sub> − ∫₀^∞ R<sub>i</sub><sup>⊥</sup>(t)dt| =
        <strong>{peor:.2e}</strong> on this signal's real network
        (rest {er:.2e}, task {et:.2e}). The paper reports &lt; 5e-15 on
        synthetic graphs. Spectral visibility is not an ad hoc construction: it
        is the diagonal of the Laplacian's resolvent and equals the
        observability Gramian.</div>""", unsafe_allow_html=True)

    # ── Gráficas ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Returnability — the framework\'s signature</div>',
                unsafe_allow_html=True)
    st.pyplot(fig_returnability(R['ejemplo'], T['ejemplo']), use_container_width=True)
    plt.close()
    st.caption("Timescale separation: the high-V node sustains energy "
               "long after the low-V node has dissipated it. In a star graph "
               "that ratio is exactly N.")

    st.markdown('<div class="sec">Anti-centrality</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(fig_anticentralidad(R['ejemplo'], 'Rest'), use_container_width=True)
        plt.close()
    with g2:
        st.pyplot(fig_anticentralidad(T['ejemplo'], 'Task'), use_container_width=True)
        plt.close()

    st.markdown('<div class="sec">Rest / task comparison</div>',
                unsafe_allow_html=True)
    st.pyplot(fig_barras(R, T), use_container_width=True)
    plt.close()

    # ── Señal ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Signal analyzed</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info" style="font-size:13px;">
<strong>Channel:</strong> <code>{nom_r}</code> (index {idx_r}, {via})<br>
<strong>Rest:</strong> {len(lab_r)} channels · {fs_r:.0f} → {fsd_r:.0f} Hz ·
{dur_r:.0f} s · {R['n_win']} windows<br>
<strong>Task:</strong> {len(lab_t)} channels · {fs_t:.0f} → {fsd_t:.0f} Hz ·
{dur_t:.0f} s · {T['n_win']} windows<br>
<strong>Preprocessing:</strong> 0.5–40 Hz bandpass (order-4 Butterworth, zero phase) ·
z-score · resampled to {FS_OBJETIVO:.0f} Hz
</div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="info">
<strong>How to read the observables</strong><br><br>
<strong>V<sub>i</sub></strong> — spectral visibility. How much persistent modal
energy node i accumulates. It is the diagonal of the Laplacian's resolvent
restricted to the transverse subspace, and equals the observability Gramian:
a high-V node is a good observation point for the system.<br><br>
<strong>τ̃ = λ₂·M₂/M₁</strong> — normalized persistence timescale.
How long energy takes to dissipate, in units of the network's global scale.<br><br>
<strong>Var(τ̃)</strong> — persistence heterogeneity across nodes. High means
the network has well-differentiated reservoirs and routers.<br><br>
<strong>λ₂</strong> — algebraic connectivity. How integrated the dynamics are.<br><br>
<strong>ρ(k,V)</strong> — structural control. Should be strongly negative:
the hub is a router, not a reservoir. It routes energy well but cannot store it.
</div>""", unsafe_allow_html=True)

    with st.expander("Detected channels"):
        st.write("**Rest:**", ", ".join(lab_r))
        st.write("**Task:**", ", ".join(lab_t))

    with st.expander("Method and validation"):
        st.markdown(f"""
**Framework:** Dynamic Spectral Persistence (v3). Replaces the quaternionic
R, C (v1) and C_dyn = λ₂²·G_pure (v2).

**Pipeline:** channel selection by name from the EDF labels →
0.5–40 Hz bandpass → z-score → resampled to {FS_OBJETIVO:.0f} Hz →
{WIN_S}-second windows with {STEP_S}-second step (max {MAX_WINS}) → delay
embedding d={EMBED_D}, {N_MAX} points → recurrence network with a threshold
at the {PCT}th percentile → L = D − A → diagonalization.

**Validation run before publishing this app:**

| Test | Result | Paper reference |
|---|---|---|
| Identity V_i = ∫R_i^⊥dt | 2e-16 … 6e-14 | < 5e-15 |
| R_hub(t) = 1/N + [(N−1)/N]e^(−Nt) | 0.00e+00 | 4.34e-15 |
| Spectral blindness of the hub | 0.00e+00 exact | 6.96e-37 |
| Anti-centrality Star S_12 (Pearson) | −1.0000 | −1.000 |
| τ_hub = 1/N, N ∈ {{6…25}} | error < 1e-16 | exact |
| EEG recurrence networks, Spearman(k,V) | −0.90 … −0.99 | −0.925 … −0.974 (ER) |

**Statistic:** the paper uses Pearson in Table 2 and Spearman in section 4.2.
They are not interchangeable: in a star graph every leaf has degree 1,
and those ties make Spearman give −0.49 where Pearson gives −1.000.
The app reports both.

**Resampling:** `resample_poly` is used, not `decimate`. `decimate` only
accepts integer factors, so 250 Hz would give 125 and 160 Hz would give 80 —
never 100. Verified that this broke comparability across datasets.

**Degeneracy filter:** windows with λmax/λ₂ > {RANGO_MAX:.0f} are discarded.
Criterion from the BA/ER sweep where τ̃ stopped converging in networks with
λ₂ ≈ 0.006 and range > 6000.
""")
else:
    st.markdown("""<div style="text-align:center;color:#bbb;
    font-family:'IBM Plex Mono',monospace;font-size:13px;padding:48px 0;">
    Upload both EDF files to begin</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="foot">
EEG Dynamic Spectral Persistence · Edher Alan Arteaga Marroquin · 2026<br>
V_i = ∫₀^∞ R_i^⊥(t) dt
</div>""", unsafe_allow_html=True)
