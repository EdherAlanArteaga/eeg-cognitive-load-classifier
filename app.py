import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import tempfile
import os

st.set_page_config(
    page_title="EEG Cognitive Load — SPG",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #f5f5f0;
    color: #1a1a1a;
}
.stApp { background-color: #f5f5f0; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #1a1a1a; }

.metric-card {
    background: #ffffff;
    border: 1px solid #d0d0c8;
    border-radius: 6px;
    padding: 16px;
    text-align: center;
    margin: 4px 0;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #1a1a1a;
}
.metric-value-pos { color: #1a6b1a; }
.metric-value-neg { color: #c0392b; }

.result-aritmética-G {
    background: #eef8ee;
    border: 2px solid #2d8a2d;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.result-aritmética-B {
    background: #fef0f0;
    border: 2px solid #c0392b;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.result-otro {
    background: #f0f4ff;
    border: 2px solid #3a7bd5;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.result-title-G {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #1a6b1a;
    margin: 8px 0 4px;
}
.result-title-B {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #c0392b;
    margin: 8px 0 4px;
}
.result-title-otro {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #3a7bd5;
    margin: 8px 0 4px;
}
.result-sub {
    font-size: 13px;
    color: #555;
    line-height: 1.5;
    margin-top: 6px;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.info-box {
    background: #ffffff;
    border-left: 3px solid #3a7bd5;
    padding: 14px 18px;
    border-radius: 0 6px 6px 0;
    margin: 14px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #333;
}
.warn-box {
    background: #fffbf0;
    border-left: 3px solid #e6a817;
    padding: 14px 18px;
    border-radius: 0 6px 6px 0;
    margin: 14px 0;
    font-size: 13px;
    line-height: 1.6;
    color: #555;
}
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 24px 0 10px;
    border-bottom: 1px solid #d0d0c8;
    padding-bottom: 5px;
}
.footer-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #aaa;
    text-align: center;
    margin-top: 48px;
    border-top: 1px solid #d0d0c8;
    padding-top: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Constantes del dataset de entrenamiento ──────────────────────────────────
MEDIANA_DELTA_R = -0.005885   # mediana de ΔR en 36 sujetos de aritmética mental
CANALES_REF = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
               'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
IDX_Fz = 16   # índice de Fz en la lista de canales del dataset EEGMAT


# ── Funciones de análisis ────────────────────────────────────────────────────
def motor_omega(sig):
    """R y C via transformada de Hilbert sobre todos los canales."""
    analytic = hilbert(sig)
    phases = np.angle(analytic)
    amplitudes = np.abs(analytic)
    r_val = float(np.abs(np.mean(np.exp(1j * phases))))
    norma = np.sqrt(np.sum(amplitudes**2, axis=0))
    c_val = float(np.mean(amplitudes[0] / (norma + 1e-9)))
    return r_val, c_val


def spg_ventanas(signal_1d, fs, win_s=4, step_s=2, max_wins=40, d=4, N_max=40):
    """
    SPG exacto del notebook: embedding de retardo → red de recurrencia →
    Laplaciano → λ₂ y C_dyn = Var(τ̃).
    Canal único (1D), igual que la implementación original con canal Fz.
    """
    win = int(win_s * fs)
    step = int(step_s * fs)
    Cs, lam2s, Gs = [], [], []
    np.random.seed(42)

    for s in range(0, min(len(signal_1d) - win, max_wins * step), step):
        x = signal_1d[s:s + win]
        if x.std() < 1e-10:
            continue
        x = (x - x.mean()) / (x.std() + 1e-10)
        x = np.clip(x, -4, 4)

        pts = np.array([x[i:i + d] for i in range(len(x) - d + 1)])
        if len(pts) > N_max:
            idx = np.linspace(0, len(pts) - 1, N_max).astype(int)
            pts = pts[idx]

        N = len(pts)
        D = cdist(pts, pts)
        D0 = D.copy()
        np.fill_diagonal(D, np.inf)
        eps = np.quantile(D[D < np.inf], 0.20)

        Adj = (D < eps).astype(float)
        np.fill_diagonal(Adj, 0)
        for i in np.where(Adj.sum(1) == 0)[0]:
            j = np.argsort(D0[i])[1]
            Adj[i, j] = Adj[j, i] = 1.0

        L = np.diag(Adj.sum(1)) - Adj
        ev, evec = eigh(L)
        k0 = next((k for k in range(1, N) if ev[k] > 1e-8), None)
        if k0 is None:
            continue

        lam2 = ev[k0]
        M1 = np.zeros(N)
        M2 = np.zeros(N)
        for k in range(k0, N):
            lk = ev[k]
            if lk < 1e-10:
                continue
            v2 = evec[:, k] ** 2
            M1 += v2 / lk
            M2 += v2 / lk ** 2

        tt = np.where(M1 > 1e-14, lam2 * (M2 / M1), 0)
        C = float(np.var(tt))
        geom = np.where(M1 > 1e-14, M2 / M1, 0)
        G_pure = float(np.var(geom))

        if np.isfinite(C) and C > 0:
            Cs.append(C)
            lam2s.append(lam2)
            Gs.append(G_pure)

    if len(Cs) < 5:
        return None

    lam2s_arr = np.array(lam2s)
    Cs_arr = np.array(Cs)
    rho, _ = spearmanr(lam2s_arr, Cs_arr)

    return {
        'C_dyn': float(np.mean(Cs_arr)),
        'lambda2': float(np.mean(lam2s_arr)),
        'G_pure': float(np.mean(Gs)),
        'CI': abs(float(rho)),
        'n_ventanas': len(Cs)
    }


def cargar_edf(archivo_bytes):
    """Carga EDF completo (sin truncar duración)."""
    try:
        import pyedflib
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
            tmp.write(archivo_bytes)
            tmp_path = tmp.name

        f = pyedflib.EdfReader(tmp_path)
        fs = f.getSampleFrequency(0)
        n_canales = min(f.signals_in_file, 23)
        n_muestras = f.getNSamples()[0]   # lee todo lo disponible

        sig = np.zeros((n_canales, n_muestras))
        for i in range(n_canales):
            sig[i, :] = f.readSignal(i, n=n_muestras)
        f.close()
        os.unlink(tmp_path)
        duracion = n_muestras / fs
        return sig, fs, n_canales, duracion
    except Exception as e:
        return None, None, None, None


def grafica(r_rep, c_rep, r_tar, c_tar, spg_rep, spg_tar, es_aritmetica, grupo):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#f5f5f0')

    color_tar = '#2d8a2d' if (es_aritmetica and grupo == 'G') else \
                '#c0392b' if (es_aritmetica and grupo == 'B') else '#3a7bd5'

    # Panel 1: Motor Omega R-C
    ax = axes[0]
    ax.set_facecolor('#ffffff')
    ax.scatter(r_rep, c_rep, color='#555', s=120, zorder=5,
               marker='o', edgecolors='white', linewidth=1.5, label='Reposo')
    ax.scatter(r_tar, c_tar, color=color_tar, s=180, zorder=6,
               marker='*', edgecolors='white', linewidth=1.5, label='Tarea')
    ax.annotate('', xy=(r_tar, c_tar), xytext=(r_rep, c_rep),
                arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.5))
    ax.set_xlabel('R  (rigidez espectral)', color='#555',
                  fontfamily='monospace', fontsize=10)
    ax.set_ylabel('C  (coherencia dinámica)', color='#555',
                  fontfamily='monospace', fontsize=10)
    ax.set_title('Motor Omega: espacio R-C', fontfamily='monospace',
                 fontsize=11, color='#1a1a1a')
    ax.tick_params(colors='#777')
    for sp in ax.spines.values():
        sp.set_color('#d0d0c8')
    ax.grid(alpha=0.25, color='#ccc')
    ax.legend(facecolor='#fff', edgecolor='#d0d0c8',
              labelcolor='#333', fontsize=9)

    # Panel 2: SPG C_dyn
    ax2 = axes[1]
    ax2.set_facecolor('#ffffff')
    if spg_rep and spg_tar:
        vals = [spg_rep['C_dyn'], spg_tar['C_dyn']]
        bars = ax2.bar(['Reposo', 'Tarea'], vals,
                       color=['#555', color_tar], alpha=0.85,
                       edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(vals) * 0.02,
                     f'{val:.5f}', ha='center', va='bottom',
                     fontfamily='monospace', fontsize=9, color='#1a1a1a')
        delta_cd = spg_tar['C_dyn'] - spg_rep['C_dyn']
        ax2.set_title(f'SPG — C_dyn  (Δ = {delta_cd:+.5f})',
                      fontfamily='monospace', fontsize=10, color='#1a1a1a')
        ax2.set_ylabel('C_dyn = Var(τ̃)', color='#555',
                       fontfamily='monospace', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Señal insuficiente\npara SPG',
                 ha='center', va='center', transform=ax2.transAxes,
                 color='#aaa', fontfamily='monospace', fontsize=11)
        ax2.set_title('SPG — C_dyn', fontfamily='monospace',
                      fontsize=10, color='#1a1a1a')

    ax2.tick_params(colors='#777')
    for sp in ax2.spines.values():
        sp.set_color('#d0d0c8')
    ax2.grid(alpha=0.2, color='#ccc', axis='y')

    plt.tight_layout()
    return fig


# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 EEG Cognitive Load — SPG")
st.markdown("**Edher Alan Arteaga Marroquin · 2026**")
st.markdown("---")

st.markdown("""
<div class="info-box">
Sube dos grabaciones EDF — una en reposo y una durante tarea cognitiva.<br><br>
<strong>Motor Omega:</strong> extrae R y C de todos los canales via transformada de Hilbert.<br>
<strong>SPG:</strong> construye una red de recurrencia en el espacio de fase del canal Fz 
(índice 16), calcula el Laplaciano, y extrae λ₂ y C_dyn = Var(τ̃).<br><br>
La clasificación G/B <em>solo aplica al dataset de aritmética mental de PhysioNet</em> 
(mediana ΔR = −0.005885). Para otros datasets los valores crudos se muestran 
sin clasificar para que los interpretes tú.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Reposo")
    st.caption("Estado basal, sin tarea.")
    archivo_reposo = st.file_uploader("EDF reposo", type=['edf'],
                                       key='reposo', label_visibility='collapsed')
with col2:
    st.markdown("#### Tarea")
    st.caption("Durante tarea cognitiva.")
    archivo_tarea = st.file_uploader("EDF tarea", type=['edf'],
                                      key='tarea', label_visibility='collapsed')

es_aritmetica = st.checkbox(
    "Este archivo viene del dataset de aritmética mental de PhysioNet (EEGMAT)",
    value=False,
    help="Activa esto solo si usas archivos del dataset PhysioNet EEG During Mental Arithmetic Tasks. "
         "La clasificación G/B usa la mediana de ΔR calibrada en ese dataset específico."
)

st.markdown("---")

if archivo_reposo and archivo_tarea:
    with st.spinner('Procesando señales...'):

        sig_rep, fs_rep, n_ch_rep, dur_rep = cargar_edf(archivo_reposo.read())
        sig_tar, fs_tar, n_ch_tar, dur_tar = cargar_edf(archivo_tarea.read())

        if sig_rep is None or sig_tar is None:
            st.error("No se pudieron leer los archivos EDF.")
            st.stop()

        # Motor Omega — todos los canales
        r_rep, c_rep = motor_omega(sig_rep)
        r_tar, c_tar = motor_omega(sig_tar)
        delta_r = r_tar - r_rep
        delta_c = c_tar - c_rep

        # SPG — canal Fz (índice 16) si existe, si no el último disponible
        ch_spg = min(IDX_Fz, sig_rep.shape[0] - 1)
        spg_rep = spg_ventanas(sig_rep[ch_spg], fs_rep)
        spg_tar = spg_ventanas(sig_tar[ch_spg], fs_tar)

        # Clasificación solo si es dataset de aritmética
        grupo = None
        if es_aritmetica:
            grupo = 'G' if delta_r > MEDIANA_DELTA_R else 'B'

        # ── Resultado principal ───────────────────────────────────────────
        st.markdown('<div class="section-title">Resultado</div>',
                    unsafe_allow_html=True)

        if es_aritmetica and grupo == 'G':
            st.markdown(f"""
            <div class="result-aritmética-G">
                <div class="result-label">Dataset EEGMAT · Aritmética mental</div>
                <div class="result-title-G">G — Cerebro Eficiente</div>
                <div class="result-sub">
                    ΔR = {delta_r:+.5f} &gt; mediana (−0.005885)<br>
                    La red se adaptó eficientemente durante la tarea.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif es_aritmetica and grupo == 'B':
            st.markdown(f"""
            <div class="result-aritmética-B">
                <div class="result-label">Dataset EEGMAT · Aritmética mental</div>
                <div class="result-title-B">B — Cerebro Inestable</div>
                <div class="result-sub">
                    ΔR = {delta_r:+.5f} ≤ mediana (−0.005885)<br>
                    La red mostró mayor esfuerzo de reorganización.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-otro">
                <div class="result-label">Dataset externo — valores crudos</div>
                <div class="result-title-otro">Análisis SPG completado</div>
                <div class="result-sub">
                    ΔR = {delta_r:+.5f} · ΔC = {delta_c:+.5f}<br>
                    La clasificación G/B no aplica a este dataset.<br>
                    Interpreta los valores de R, C, λ₂ y C_dyn directamente.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="warn-box">
            La mediana de ΔR (−0.005885) fue calibrada en 36 sujetos del dataset 
            PhysioNet EEGMAT a 500 Hz con 23 canales. Aplicarla a otro dataset 
            (diferente hardware, frecuencia o tarea) produce clasificaciones sin sentido.<br><br>
            Usa los valores de ΔR, ΔC, λ₂ y C_dyn para comparar sujetos 
            <em>dentro del mismo dataset</em>.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # ── Motor Omega ───────────────────────────────────────────────────
        st.markdown('<div class="section-title">Motor Omega — R y C</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in zip(
            [c1, c2, c3, c4],
            ['R Reposo', 'R Tarea', 'C Reposo', 'C Tarea'],
            [r_rep, r_tar, c_rep, c_tar]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val:.5f}</div>
            </div>""", unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        dr_cls = 'metric-value-pos' if delta_r > MEDIANA_DELTA_R else 'metric-value-neg'
        dc_cls = 'metric-value-pos' if abs(delta_c) < 0.05 else 'metric-value-neg'

        c5.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ΔR (Tarea − Reposo)</div>
            <div class="metric-value {dr_cls}">{delta_r:+.5f}</div>
        </div>""", unsafe_allow_html=True)

        c6.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ΔC (Tarea − Reposo)</div>
            <div class="metric-value {dc_cls}">{delta_c:+.5f}</div>
        </div>""", unsafe_allow_html=True)

        # ── SPG ───────────────────────────────────────────────────────────
        st.markdown(f'<div class="section-title">SPG — Canal {ch_spg} '
                    f'{"(Fz)" if ch_spg == 16 else ""} · Laplaciano y geometría espectral</div>',
                    unsafe_allow_html=True)

        if spg_rep and spg_tar:
            s1, s2, s3, s4 = st.columns(4)
            delta_cd = spg_tar['C_dyn'] - spg_rep['C_dyn']
            delta_l2 = spg_tar['lambda2'] - spg_rep['lambda2']

            for col, label, val in zip(
                [s1, s2, s3, s4],
                ['λ₂ Reposo', 'λ₂ Tarea', 'C_dyn Reposo', 'C_dyn Tarea'],
                [spg_rep['lambda2'], spg_tar['lambda2'],
                 spg_rep['C_dyn'], spg_tar['C_dyn']]
            ):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val:.5f}</div>
                </div>""", unsafe_allow_html=True)

            s5, s6 = st.columns(2)
            s5.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔC_dyn (Tarea − Reposo)</div>
                <div class="metric-value">{delta_cd:+.5f}</div>
            </div>""", unsafe_allow_html=True)

            s6.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Δλ₂ (Tarea − Reposo)</div>
                <div class="metric-value">{delta_l2:+.5f}</div>
            </div>""", unsafe_allow_html=True)

            st.caption(
                f"Ventanas — Reposo: {spg_rep['n_ventanas']} · "
                f"Tarea: {spg_tar['n_ventanas']} · "
                f"Duración — Reposo: {dur_rep:.0f}s · Tarea: {dur_tar:.0f}s"
            )
        else:
            st.warning(
                f"Señal insuficiente para SPG en canal {ch_spg}. "
                "Se necesitan al menos 5 ventanas de 4 segundos."
            )

        # ── Gráfica ───────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Visualización</div>',
                    unsafe_allow_html=True)
        fig = grafica(r_rep, c_rep, r_tar, c_tar, spg_rep, spg_tar,
                      es_aritmetica, grupo)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Interpretación ────────────────────────────────────────────────
        st.markdown('<div class="section-title">Interpretación</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
<div class="info-box">
<strong>Motor Omega:</strong> R mide la sincronización global de fases entre 
los {n_ch_rep} canales. C mide qué fracción de la amplitud total corresponde 
al canal 0. Broadband, sin filtro de frecuencia.<br><br>

<strong>SPG — canal {ch_spg}{"  (Fz)" if ch_spg == 16 else ""}:</strong>
Para cada ventana de 4 segundos se construye una red de recurrencia en el 
espacio de fase (embedding d=4, percentil 20). El Laplaciano de esa red 
tiene autovalores λ₁ ≤ λ₂ ≤ … donde λ₂ es la conectividad algebraica. 
C_dyn = Var(τ̃) es la varianza del timescale de persistencia τ̃ = λ₂ · M₂/M₁, 
que mide la heterogeneidad geométrica del atractor.<br><br>

<em>Fun fact:</em> Los cerebros eficientes en aritmética mental trabajan menos, 
no más. Su red no necesita reorganizarse al enfrentar el problema — ya está lista. 
Los inestables hacen más esfuerzo de reorganización y producen menos resultado.
</div>
""", unsafe_allow_html=True)

        with st.expander("Información técnica"):
            st.markdown(f"""
**Archivos:**
- Reposo: `{archivo_reposo.name}` · {n_ch_rep} canales · {fs_rep:.0f} Hz · {dur_rep:.0f}s
- Tarea: `{archivo_tarea.name}` · {n_ch_tar} canales · {fs_tar:.0f} Hz · {dur_tar:.0f}s

**Motor Omega:** Hilbert broadband · primeros 23 canales · señal completa

**SPG:** Canal {ch_spg} · embedding d=4 · red de recurrencia percentil 20 · 
ventanas 4s/paso 2s · máx 40 ventanas

**Clasificación G/B:** Solo válida para dataset PhysioNet EEGMAT.
Mediana de ΔR = −0.005885 (36 sujetos, 500Hz, 23 canales).

**Dataset de referencia:**
https://physionet.org/content/eegmat/1.0.0/
""")

else:
    st.markdown("""
<div style="text-align:center; color:#bbb; font-family:'IBM Plex Mono',monospace;
font-size:13px; padding:48px 0;">
Sube los dos archivos EDF para comenzar
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
EEG Cognitive Load — SPG · Edher Alan Arteaga Marroquin · 2026<br>
Motor Omega · Spectral Persistence Geometry · PhysioNet EEGMAT
</div>
""", unsafe_allow_html=True)
