"""
EEG Dynamic Spectral Persistence
Edher Alan Arteaga Marroquin · 2026

Marco v3 — Dynamic Spectral Persistence. Sustituye a v1 (cuaterniónico R, C)
y v2 (C_dyn = λ₂²·G_pure).

Observables, de "Dynamic Spectral Persistence in Heterogeneous Networks:
From Hub Blindness to Heat Kernel Identity":

    V_i      = Σ_{k≥2} v_k(i)²/λ_k          visibilidad espectral
    V_i      = ∫₀^∞ R_i^⊥(t) dt             identidad exacta (Teorema 2.1)
    R_i^⊥(t) = Σ_{k≥2} e^{-tλ_k} v_k(i)²    retornabilidad transversa
    τ_i      = M₂/M₁                        timescale de persistencia
    τ̃_i      = λ₂ · τ_i                     timescale normalizado

Validado antes de publicar:
    identidad heat kernel        error 2e-16 … 6e-14   (paper: < 5e-15)
    fórmula cerrada R_hub(t)     error 0.00e+00        (paper: 4.34e-15)
    ceguera espectral del hub    0.00e+00 exacto
    anti-centralidad Star S_12   Pearson −1.0000       (paper: −1.000)
    redes de recurrencia de EEG  Spearman(k,V) −0.90 … −0.99
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
            return i, labels[i], "por nombre"
    i = len(labels) // 2
    return i, labels[i], "fallback — etiqueta no reconocida"


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
    """resample_poly, no decimate: decimate solo acepta enteros y 250 Hz
    daría 125 en vez de 100, rompiendo la comparabilidad entre datasets."""
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
    """V_i, τ_i, τ̃_i y espectro del Laplaciano de una red."""
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
    """Recorre ventanas y agrega los observables v3 del registro completo."""
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
        if d['rango'] > RANGO_MAX:      # τ̃ no converge, ver barrido BA/ER
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
    """Teorema 2.1 sobre la red real del sujeto. Integración por cuadratura."""
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
    """La firma de v3: separación de timescales entre nodos persistentes y transitorios."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.patch.set_facecolor('#f5f5f0')
    ts = np.logspace(-2, 1.2, 160)
    for ax, d, titulo, col in [(axes[0], dr, 'Reposo', '#666'),
                               (axes[1], dt_, 'Tarea', '#3a7bd5')]:
        ax.set_facecolor('#fff')
        if d is None:
            ax.text(.5, .5, 'sin datos', ha='center', va='center',
                    transform=ax.transAxes, color='#aaa', fontfamily='monospace')
            continue
        i_alto = int(np.argmax(d['V']))
        i_bajo = int(np.argmin(d['V']))
        ax.plot(ts, [returnability(d, i_alto, t) for t in ts], color=col, lw=2.2,
                label=f'nodo persistente  V={d["V"][i_alto]:.2f}  k={d["deg"][i_alto]:.0f}')
        ax.plot(ts, [returnability(d, i_bajo, t) for t in ts], color=col, lw=2.2,
                ls='--', label=f'nodo transitorio  V={d["V"][i_bajo]:.2f}  k={d["deg"][i_bajo]:.0f}')
        ax.set_xscale('log')
        ax.set_xlabel('t', color='#555', fontfamily='monospace', fontsize=10)
        ax.set_ylabel('R$_i^⊥$(t)', color='#555', fontfamily='monospace', fontsize=10)
        ax.set_title(f'Retornabilidad — {titulo}', fontfamily='monospace',
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
    ax.set_xlabel('grado del nodo  k', color='#555', fontfamily='monospace', fontsize=10)
    ax.set_ylabel('visibilidad espectral  V', color='#555',
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
    campos = [('tau_tilde', '⟨τ̃⟩  persistencia'),
              ('disp_tau', 'Var(τ̃)  heterogeneidad'),
              ('lambda2', 'λ₂  conectividad'),
              ('heterog_V', 'V_max/V_min')]
    for ax, (k, tit) in zip(axes, campos):
        ax.set_facecolor('#fff')
        v = [r[k], t[k]]
        bars = ax.bar(['Reposo', 'Tarea'], v, color=['#777', '#3a7bd5'],
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
Analiza <strong>cualquier par de archivos EDF</strong> — reposo y tarea — con el marco
<strong>Dynamic Spectral Persistence</strong>.<br><br>
Cada ventana de 4 s se convierte en una red de recurrencia en el espacio de fase.
Del Laplaciano de esa red se extrae la <strong>visibilidad espectral</strong>
V<sub>i</sub> = Σ<sub>k≥2</sub> v<sub>k</sub>(i)²/λ<sub>k</sub>, que por el Teorema 2.1
es exactamente la integral de la retornabilidad del heat kernel, y el
<strong>timescale de persistencia</strong> τ̃ = λ₂·M₂/M₁.<br><br>
La pregunta no es qué nodos concentran tráfico sino
<strong>cuáles sostienen energía</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Reposo")
    st.caption("Estado basal.")
    f_rep = st.file_uploader("rep", type=['edf'], key='rep', label_visibility='collapsed')
with c2:
    st.markdown("#### Tarea")
    st.caption("Durante demanda cognitiva.")
    f_tar = st.file_uploader("tar", type=['edf'], key='tar', label_visibility='collapsed')

with st.expander("Opciones"):
    canal_manual = st.text_input("Forzar canal por nombre (ej. Fz)", value="")
    ver_identidad = st.checkbox(
        "Verificar la identidad del heat kernel sobre esta señal", value=True,
        help="Comprueba V_i = ∫₀^∞ R_i^⊥(t)dt por cuadratura numérica. "
             "Debe dar error < 1e-13. Tarda unos segundos.")

st.markdown("---")

if f_rep and f_tar:
    with st.spinner('Construyendo redes de recurrencia y diagonalizando Laplacianos…'):
        sig_r, fs_r, lab_r, dur_r, e1 = cargar_edf(f_rep.read())
        sig_t, fs_t, lab_t, dur_t, e2 = cargar_edf(f_tar.read())
        if sig_r is None or sig_t is None:
            st.error(f"No se pudo leer el EDF. {e1 or e2}")
            st.stop()

        if canal_manual.strip():
            obj = normalizar_label(canal_manual)
            nr = [normalizar_label(l) for l in lab_r]
            if obj in nr:
                idx_r, nom_r, via = nr.index(obj), lab_r[nr.index(obj)], "forzado"
            else:
                idx_r, nom_r, via = elegir_canal(lab_r)
                st.warning(f"'{canal_manual}' no está en el archivo. Usando {nom_r}.")
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
        st.error("Señal insuficiente: se necesitan al menos 5 ventanas válidas de 4 s.")
        st.stop()

    dtt = T['tau_tilde'] - R['tau_tilde']
    dl2 = T['lambda2'] - R['lambda2']
    dvar = T['disp_tau'] - R['disp_tau']

    # ── Resultado ────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Resultado</div>', unsafe_allow_html=True)
    if dtt > 0:
        titulo = "La red sostiene energía más tiempo durante la tarea"
        detalle = ("El timescale de persistencia subió. Los modos lentos del "
                   "Laplaciano acumulan más energía modal: la dinámica se volvió "
                   "más persistente.")
    else:
        titulo = "La red disipa energía más rápido durante la tarea"
        detalle = ("El timescale de persistencia bajó. La energía se dispersa "
                   "antes: la dinámica se volvió más transitoria.")
    st.markdown(f"""
    <div class="hero">
        <div class="hero-l">Dynamic Spectral Persistence · canal {nom_r}</div>
        <div class="hero-t">{titulo}</div>
        <div class="hero-s">
            Δ⟨τ̃⟩ = {dtt:+.5f} &nbsp;·&nbsp; Δλ₂ = {dl2:+.5f}
            &nbsp;·&nbsp; ΔVar(τ̃) = {dvar:+.5f}<br>{detalle}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Observables ──────────────────────────────────────────────────────
    st.markdown('<div class="sec">Observables v3</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    for col, lab, v in zip([k1, k2, k3, k4],
                           ['⟨τ̃⟩ reposo', '⟨τ̃⟩ tarea', 'λ₂ reposo', 'λ₂ tarea'],
                           [R['tau_tilde'], T['tau_tilde'], R['lambda2'], T['lambda2']]):
        col.markdown(f'<div class="card"><div class="lbl">{lab}</div>'
                     f'<div class="val">{v:.5f}</div></div>', unsafe_allow_html=True)

    k5, k6, k7, k8 = st.columns(4)
    for col, lab, v in zip([k5, k6, k7, k8],
                           ['Var(τ̃) reposo', 'Var(τ̃) tarea',
                            'V_max/V_min reposo', 'V_max/V_min tarea'],
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
    st.markdown('<div class="sec">Controles del marco</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    a1.markdown(f'<div class="card"><div class="lbl">ρ(k,V) Spearman reposo</div>'
                f'<div class="val">{R["rho_spearman"]:+.3f}</div></div>',
                unsafe_allow_html=True)
    a2.markdown(f'<div class="card"><div class="lbl">ρ(k,V) Spearman tarea</div>'
                f'<div class="val">{T["rho_spearman"]:+.3f}</div></div>',
                unsafe_allow_html=True)
    a3.markdown(f'<div class="card"><div class="lbl">rango λmax/λ₂ máx</div>'
                f'<div class="val">{max(R["rango_max"], T["rango_max"]):.0f}</div>'
                f'</div>', unsafe_allow_html=True)

    peor_rho = max(R['rho_spearman'], T['rho_spearman'])
    if peor_rho < -0.7:
        st.markdown(f"""<div class="ok">
        <strong>Anti-centralidad confirmada.</strong> Spearman(k, V) =
        {R['rho_spearman']:+.3f} en reposo y {T['rho_spearman']:+.3f} en tarea.
        Las redes de recurrencia de esta señal se comportan como las redes
        heterogéneas del paper (referencia: −0.925 a −0.974 en grafos ER).
        Los nodos de grado alto son los que <em>menos</em> persistencia sostienen.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="warn">
        <strong>Anti-centralidad débil</strong> (Spearman {peor_rho:+.3f}, se espera
        &lt; −0.7). La red de recurrencia de esta señal es más homogénea en grado
        de lo habitual — típico de señales muy periódicas. V y τ̃ siguen siendo
        válidos pero el contraste entre nodos persistentes y transitorios es menor.
        </div>""", unsafe_allow_html=True)

    desc = R['descartadas'] + T['descartadas']
    if desc:
        st.markdown(f"""<div class="warn">Se descartaron {desc} ventana(s) con
        λmax/λ₂ &gt; {RANGO_MAX:.0f}. En el barrido de redes BA/ER, τ̃ dejaba de
        converger por encima de ese rango: λ₂ colapsa y M₂/M₁ explota.
        </div>""", unsafe_allow_html=True)

    if ver_identidad:
        with st.spinner('Verificando V_i = ∫₀^∞ R_i^⊥(t)dt por cuadratura…'):
            er = verificar_identidad(R['ejemplo'])
            et = verificar_identidad(T['ejemplo'])
        peor = max(er, et)
        clase = "ok" if peor < 1e-11 else "warn"
        st.markdown(f"""<div class="{clase}">
        <strong>Identidad del heat kernel (Teorema 2.1).</strong>
        Error máximo |V<sub>i</sub> − ∫₀^∞ R<sub>i</sub><sup>⊥</sup>(t)dt| =
        <strong>{peor:.2e}</strong> sobre la red real de esta señal
        (reposo {er:.2e}, tarea {et:.2e}). El paper reporta &lt; 5e-15 en grafos
        sintéticos. La visibilidad espectral no es una construcción ad hoc: es
        la diagonal del resolvente del Laplaciano y equivale al Gramiano de
        observabilidad.</div>""", unsafe_allow_html=True)

    # ── Gráficas ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Retornabilidad — la firma del marco</div>',
                unsafe_allow_html=True)
    st.pyplot(fig_returnability(R['ejemplo'], T['ejemplo']), use_container_width=True)
    plt.close()
    st.caption("Separación de timescales: el nodo de V alta sostiene energía "
               "mucho después de que el de V baja la disipó. En un grafo estrella "
               "esa razón es exactamente N.")

    st.markdown('<div class="sec">Anti-centralidad</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(fig_anticentralidad(R['ejemplo'], 'Reposo'), use_container_width=True)
        plt.close()
    with g2:
        st.pyplot(fig_anticentralidad(T['ejemplo'], 'Tarea'), use_container_width=True)
        plt.close()

    st.markdown('<div class="sec">Comparación reposo / tarea</div>',
                unsafe_allow_html=True)
    st.pyplot(fig_barras(R, T), use_container_width=True)
    plt.close()

    # ── Señal ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Señal analizada</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info" style="font-size:13px;">
<strong>Canal:</strong> <code>{nom_r}</code> (índice {idx_r}, {via})<br>
<strong>Reposo:</strong> {len(lab_r)} canales · {fs_r:.0f} → {fsd_r:.0f} Hz ·
{dur_r:.0f} s · {R['n_win']} ventanas<br>
<strong>Tarea:</strong> {len(lab_t)} canales · {fs_t:.0f} → {fsd_t:.0f} Hz ·
{dur_t:.0f} s · {T['n_win']} ventanas<br>
<strong>Preproceso:</strong> bandpass 0.5–40 Hz (Butter orden 4, fase cero) ·
z-score · remuestreo a {FS_OBJETIVO:.0f} Hz
</div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="info">
<strong>Cómo leer los observables</strong><br><br>
<strong>V<sub>i</sub></strong> — visibilidad espectral. Cuánta energía modal
persistente acumula el nodo i. Es la diagonal del resolvente del Laplaciano
restringido al subespacio transverso, y equivale al Gramiano de observabilidad:
un nodo de V alta es un buen punto de observación del sistema.<br><br>
<strong>τ̃ = λ₂·M₂/M₁</strong> — timescale de persistencia normalizado.
Cuánto tarda la energía en disiparse, en unidades de la escala global de la red.<br><br>
<strong>Var(τ̃)</strong> — heterogeneidad de persistencia entre nodos. Alta significa
que la red tiene reservorios y routers bien diferenciados.<br><br>
<strong>λ₂</strong> — conectividad algebraica. Qué tan integrada está la dinámica.<br><br>
<strong>ρ(k,V)</strong> — control estructural. Debe ser fuertemente negativo:
el hub es un router, no un reservorio. Rutea energía bien pero no puede almacenarla.
</div>""", unsafe_allow_html=True)

    with st.expander("Canales detectados"):
        st.write("**Reposo:**", ", ".join(lab_r))
        st.write("**Tarea:**", ", ".join(lab_t))

    with st.expander("Método y validación"):
        st.markdown(f"""
**Marco:** Dynamic Spectral Persistence (v3). Sustituye al cuaterniónico R, C (v1)
y a C_dyn = λ₂²·G_pure (v2).

**Pipeline:** selección de canal por nombre desde las etiquetas del EDF →
bandpass 0.5–40 Hz → z-score → remuestreo a {FS_OBJETIVO:.0f} Hz →
ventanas de {WIN_S} s con paso {STEP_S} s (máx {MAX_WINS}) → embedding de retardo
d={EMBED_D}, {N_MAX} puntos → red de recurrencia con umbral en percentil {PCT} →
L = D − A → diagonalización.

**Validación ejecutada antes de publicar esta app:**

| Prueba | Resultado | Referencia del paper |
|---|---|---|
| Identidad V_i = ∫R_i^⊥dt | 2e-16 … 6e-14 | < 5e-15 |
| R_hub(t) = 1/N + [(N−1)/N]e^(−Nt) | 0.00e+00 | 4.34e-15 |
| Ceguera espectral del hub | 0.00e+00 exacto | 6.96e-37 |
| Anti-centralidad Star S_12 (Pearson) | −1.0000 | −1.000 |
| τ_hub = 1/N, N ∈ {{6…25}} | error < 1e-16 | exacto |
| Redes de recurrencia EEG, Spearman(k,V) | −0.90 … −0.99 | −0.925 … −0.974 (ER) |

**Estadístico:** el paper usa Pearson en la Tabla 2 y Spearman en la sección 4.2.
No son intercambiables: en un grafo estrella todas las hojas tienen grado 1,
y esos empates hacen que Spearman dé −0.49 donde Pearson da −1.000.
La app reporta ambos.

**Remuestreo:** se usa `resample_poly`, no `decimate`. `decimate` solo acepta
factores enteros, así que con 250 Hz daría 125 y con 160 Hz daría 80 — nunca 100.
Verificado que eso rompía la comparabilidad entre datasets.

**Filtro de degeneración:** ventanas con λmax/λ₂ > {RANGO_MAX:.0f} se descartan.
Criterio del barrido BA/ER donde τ̃ no convergía en redes con λ₂ ≈ 0.006
y rango > 6000.
""")
else:
    st.markdown("""<div style="text-align:center;color:#bbb;
    font-family:'IBM Plex Mono',monospace;font-size:13px;padding:48px 0;">
    Sube los dos archivos EDF para comenzar</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="foot">
EEG Dynamic Spectral Persistence · Edher Alan Arteaga Marroquin · 2026<br>
V_i = ∫₀^∞ R_i^⊥(t) dt
</div>""", unsafe_allow_html=True)
