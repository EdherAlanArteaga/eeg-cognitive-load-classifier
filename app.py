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
    page_title="EEG Cognitive Load Classifier",
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

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #1a1a1a;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #d0d0c8;
    border-radius: 6px;
    padding: 18px;
    text-align: center;
    margin: 6px 0;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #1a1a1a;
}
.result-G {
    background: #eef8ee;
    border: 2px solid #2d8a2d;
    border-radius: 8px;
    padding: 24px;
    text-align: center;
}
.result-B {
    background: #fef0f0;
    border: 2px solid #c0392b;
    border-radius: 8px;
    padding: 24px;
    text-align: center;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.result-value-G {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #1a6b1a;
    margin: 10px 0 6px;
}
.result-value-B {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #c0392b;
    margin: 10px 0 6px;
}
.result-desc {
    font-size: 14px;
    color: #444;
    line-height: 1.5;
}
.info-box {
    background: #ffffff;
    border-left: 3px solid #3a7bd5;
    padding: 16px 20px;
    border-radius: 0 6px 6px 0;
    margin: 16px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #333;
}
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 24px 0 12px;
    border-bottom: 1px solid #d0d0c8;
    padding-bottom: 6px;
}
.footer-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #999;
    text-align: center;
    margin-top: 48px;
    border-top: 1px solid #d0d0c8;
    padding-top: 16px;
}
.stButton > button {
    background: #1a1a1a;
    color: #f5f5f0;
    border: none;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)


# ── Motor Omega ──────────────────────────────────────────────────────────────
def motor_omega(sig):
    analytic = hilbert(sig)
    phases = np.angle(analytic)
    amplitudes = np.abs(analytic)
    r_val = float(np.abs(np.mean(np.exp(1j * phases))))
    norma = np.sqrt(np.sum(amplitudes**2, axis=0))
    c_val = float(np.mean(amplitudes[0] / (norma + 1e-9)))
    return r_val, c_val


# ── SPG: Geometría espectral por ventanas ────────────────────────────────────
def spg_ventanas(signal, fs, win_s=4, step_s=2, max_wins=40, d=4, N_max=40):
    """
    SPG real: embedding de retardo → red de recurrencia → Laplaciano → λ₂, C_dyn
    Implementación exacta del notebook EEG_COGNITIVE_SPG_ARITHMETIC_CON_CSV.
    """
    win = int(win_s * fs)
    step = int(step_s * fs)
    Cs, lam2s, Gs = [], [], []
    np.random.seed(42)

    for s in range(0, min(len(signal) - win, max_wins * step), step):
        x = signal[s:s + win]
        if x.std() < 1e-10:
            continue
        x = (x - x.mean()) / (x.std() + 1e-10)
        x = np.clip(x, -4, 4)

        # Embedding de retardo
        pts = np.array([x[i:i + d] for i in range(len(x) - d + 1)])
        if len(pts) > N_max:
            idx = np.linspace(0, len(pts) - 1, N_max).astype(int)
            pts = pts[idx]

        N = len(pts)
        D = cdist(pts, pts)
        D0 = D.copy()
        np.fill_diagonal(D, np.inf)
        eps = np.quantile(D[D < np.inf], 0.20)

        # Red de recurrencia
        Adj = (D < eps).astype(float)
        np.fill_diagonal(Adj, 0)

        # Conectar nodos aislados
        for i in np.where(Adj.sum(1) == 0)[0]:
            j = np.argsort(D0[i])[1]
            Adj[i, j] = Adj[j, i] = 1.0

        # Laplaciano y espectro
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

        # tau_tilde y C_dyn (varianza del timescale)
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

    return {
        'C_dyn': float(np.mean(Cs)),
        'lambda2': float(np.mean(lam2s)),
        'G_pure': float(np.mean(Gs)),
        'n_ventanas': len(Cs)
    }


def manifold_position(lam2_series, cdyn_series):
    """d_eff y CI desde series temporales de (λ₂, C_dyn)."""
    X = np.column_stack([
        (lam2_series - lam2_series.mean()) / (lam2_series.std() + 1e-12),
        (cdyn_series - cdyn_series.mean()) / (cdyn_series.std() + 1e-12)
    ])
    ev2 = np.maximum(np.linalg.eigvalsh(np.cov(X.T)), 0)
    d_eff = float(ev2.sum() ** 2 / (ev2 ** 2).sum()) if (ev2 ** 2).sum() > 0 else np.nan
    rho, _ = spearmanr(lam2_series, cdyn_series)
    CI = abs(float(rho))
    return {'d_eff': d_eff, 'CI': CI}


def cargar_edf(archivo_bytes, duracion_segundos=62):
    try:
        import pyedflib
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
            tmp.write(archivo_bytes)
            tmp_path = tmp.name

        f = pyedflib.EdfReader(tmp_path)
        fs = f.getSampleFrequency(0)
        n_canales = min(f.signals_in_file, 23)
        n_disp = f.getNSamples()[0]
        n_muestras = min(int(fs * duracion_segundos), n_disp)

        sig = np.zeros((n_canales, n_muestras))
        for i in range(n_canales):
            sig[i, :] = f.readSignal(i, n=n_muestras)
        f.close()
        os.unlink(tmp_path)
        return sig, fs, n_canales
    except Exception as e:
        return None, None, None


def clasificar(delta_r):
    """
    Criterio del experimento: mediana de Delta_R del dataset = -0.005885
    G (Eficiente): Delta_R > mediana
    B (Inestable): Delta_R <= mediana
    """
    MEDIANA_DATASET = -0.005885
    if delta_r > MEDIANA_DATASET:
        return 'G'
    else:
        return 'B'


def grafica(r_rep, c_rep, r_tar, c_tar, grupo, spg_rep, spg_tar):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#f5f5f0')

    # Panel 1: Espacio R-C Motor Omega
    ax = axes[0]
    ax.set_facecolor('#ffffff')
    ax.set_title('Motor Omega: espacio R-C', fontfamily='monospace', fontsize=11, color='#1a1a1a')

    color = '#2d8a2d' if grupo == 'G' else '#c0392b'

    ax.scatter(r_rep, c_rep, color='#3a7bd5', s=120, zorder=5,
               marker='o', edgecolors='white', linewidth=1.5, label='Reposo')
    ax.scatter(r_tar, c_tar, color=color, s=180, zorder=6,
               marker='*', edgecolors='white', linewidth=1.5, label='Tarea')
    ax.annotate('', xy=(r_tar, c_tar), xytext=(r_rep, c_rep),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

    ax.set_xlabel('R  (Rigidez espectral)', color='#444', fontfamily='monospace', fontsize=10)
    ax.set_ylabel('C  (Coherencia dinámica)', color='#444', fontfamily='monospace', fontsize=10)
    ax.tick_params(colors='#666')
    for spine in ax.spines.values():
        spine.set_color('#d0d0c8')
    ax.grid(alpha=0.3, color='#ccc')
    ax.legend(facecolor='#fff', edgecolor='#d0d0c8', labelcolor='#333', fontsize=9)

    # Panel 2: SPG - C_dyn reposo vs tarea
    ax2 = axes[1]
    ax2.set_facecolor('#ffffff')
    ax2.set_title('SPG: C_dyn por estado', fontfamily='monospace', fontsize=11, color='#1a1a1a')

    if spg_rep and spg_tar:
        estados = ['Reposo', 'Tarea']
        valores = [spg_rep['C_dyn'], spg_tar['C_dyn']]
        colores = ['#3a7bd5', color]
        bars = ax2.bar(estados, valores, color=colores, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax2.set_ylabel('C_dyn (varianza del timescale τ)', color='#444',
                       fontfamily='monospace', fontsize=9)

        for bar, val in zip(bars, valores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(valores) * 0.02,
                     f'{val:.4f}', ha='center', va='bottom',
                     fontfamily='monospace', fontsize=10, color='#1a1a1a')

        delta_cdyn = spg_tar['C_dyn'] - spg_rep['C_dyn']
        ax2.set_title(f'SPG: C_dyn por estado  (ΔC_dyn = {delta_cdyn:+.4f})',
                      fontfamily='monospace', fontsize=10, color='#1a1a1a')
    else:
        ax2.text(0.5, 0.5, 'Señal insuficiente\npara SPG', ha='center', va='center',
                 transform=ax2.transAxes, color='#999', fontfamily='monospace')

    for spine in ax2.spines.values():
        spine.set_color('#d0d0c8')
    ax2.tick_params(colors='#666')
    ax2.grid(alpha=0.2, color='#ccc', axis='y')

    plt.tight_layout()
    return fig


# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 EEG Cognitive Load Classifier")
st.markdown("**Edher Alan Arteaga Marroquin · 2026**")
st.markdown("---")

st.markdown("""
<div class="info-box">
Sube dos grabaciones EDF — una en reposo y una durante tarea cognitiva. 
El análisis usa <strong>dos métodos en paralelo</strong>:<br><br>
<strong>Motor Omega</strong> — extrae R y C directamente de la señal EEG multicanal 
via transformada de Hilbert.<br><br>
<strong>SPG (Geometría de Persistencia Espectral)</strong> — construye una red de recurrencia 
en el espacio de fase de la señal, calcula el Laplaciano de esa red, y extrae 
λ₂ (conectividad global) y C_dyn = Var(τ) (varianza del timescale de persistencia).<br><br>
La clasificación G/B sigue el criterio del experimento: 
<strong>ΔR > mediana del dataset → G (Eficiente)</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Grabación en Reposo")
    st.caption("Estado basal, sin tarea.")
    archivo_reposo = st.file_uploader("EDF reposo", type=['edf'],
                                       key='reposo', label_visibility='collapsed')
with col2:
    st.markdown("#### Grabación en Tarea")
    st.caption("Durante tarea cognitiva.")
    archivo_tarea = st.file_uploader("EDF tarea", type=['edf'],
                                      key='tarea', label_visibility='collapsed')

canal_spg = st.selectbox(
    "Canal para SPG (elige el más frontal disponible en tu dataset)",
    options=list(range(23)),
    index=16,
    format_func=lambda x: f"Canal {x} {'(Fz — recomendado para aritmética)' if x == 16 else ''}"
)

st.markdown("---")

if archivo_reposo and archivo_tarea:
    with st.spinner('Procesando con Motor Omega y SPG...'):

        sig_rep, fs_rep, n_ch_rep = cargar_edf(archivo_reposo.read())
        sig_tar, fs_tar, n_ch_tar = cargar_edf(archivo_tarea.read())

        if sig_rep is None or sig_tar is None:
            st.error("No se pudieron leer los archivos EDF.")
        else:
            # Motor Omega
            r_rep, c_rep = motor_omega(sig_rep)
            r_tar, c_tar = motor_omega(sig_tar)
            delta_r = r_tar - r_rep
            delta_c = c_tar - c_rep

            # Clasificación por mediana de Delta_R
            grupo = clasificar(delta_r)

            # SPG por ventanas en canal seleccionado
            ch = min(canal_spg, sig_rep.shape[0] - 1)
            spg_rep = spg_ventanas(sig_rep[ch], fs_rep)
            spg_tar = spg_ventanas(sig_tar[ch], fs_tar)

            # Resultado principal
            st.markdown('<div class="section-title">Clasificación</div>', unsafe_allow_html=True)

            if grupo == 'G':
                st.markdown(f"""
                <div class="result-G">
                    <div class="result-label">Resultado del análisis</div>
                    <div class="result-value-G">G — Cerebro Eficiente</div>
                    <div class="result-desc">
                        ΔR = {delta_r:+.5f} &gt; mediana del dataset (−0.005885)<br>
                        La red neuronal se adaptó eficientemente durante la tarea.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-B">
                    <div class="result-label">Resultado del análisis</div>
                    <div class="result-value-B">B — Cerebro Inestable</div>
                    <div class="result-desc">
                        ΔR = {delta_r:+.5f} ≤ mediana del dataset (−0.005885)<br>
                        La red neuronal mostró mayor esfuerzo de reorganización.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)

            # Métricas Motor Omega
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
            dr_color = '#1a6b1a' if delta_r > -0.005885 else '#c0392b'
            dc_color = '#1a6b1a' if abs(delta_c) < 0.05 else '#c0392b'

            c5.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔR (Tarea − Reposo)</div>
                <div class="metric-value" style="color:{dr_color}">{delta_r:+.5f}</div>
            </div>""", unsafe_allow_html=True)

            c6.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔC (Tarea − Reposo)</div>
                <div class="metric-value" style="color:{dc_color}">{delta_c:+.5f}</div>
            </div>""", unsafe_allow_html=True)

            # Métricas SPG
            st.markdown('<div class="section-title">SPG — Geometría Espectral</div>',
                        unsafe_allow_html=True)

            if spg_rep and spg_tar:
                s1, s2, s3, s4 = st.columns(4)
                delta_cdyn = spg_tar['C_dyn'] - spg_rep['C_dyn']
                delta_lam2 = spg_tar['lambda2'] - spg_rep['lambda2']

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
                    <div class="metric-value">{delta_cdyn:+.5f}</div>
                </div>""", unsafe_allow_html=True)

                s6.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Δλ₂ (Tarea − Reposo)</div>
                    <div class="metric-value">{delta_lam2:+.5f}</div>
                </div>""", unsafe_allow_html=True)

                st.caption(f"Ventanas procesadas — Reposo: {spg_rep['n_ventanas']} · Tarea: {spg_tar['n_ventanas']} · Canal: {ch}")
            else:
                st.warning("Señal insuficiente para calcular SPG. Intenta con un archivo más largo o un canal diferente.")

            # Gráfica
            st.markdown('<div class="section-title">Visualización</div>', unsafe_allow_html=True)
            fig = grafica(r_rep, c_rep, r_tar, c_tar, grupo, spg_rep, spg_tar)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Interpretación
            st.markdown('<div class="section-title">Interpretación</div>', unsafe_allow_html=True)
            st.markdown(f"""
<div class="info-box">
<strong>Motor Omega:</strong><br>
R mide la sincronización global de fases entre los {n_ch_rep} canales EEG.
C mide qué fracción de la amplitud total corresponde al canal de referencia.<br><br>

<strong>SPG — Laplaciano y geometría espectral:</strong><br>
Para cada ventana de 4 segundos, se construye una red de recurrencia en el espacio de fase 
de la señal del canal {ch}. El Laplaciano de esa red tiene autovalores λ₁ ≤ λ₂ ≤ ... 
donde λ₂ mide la conectividad algebraica (qué tan integrada está la red). 
C_dyn = Var(τ) = varianza del timescale de persistencia τ = λ₂ · M₂/M₁, 
que mide la heterogeneidad geométrica del atractor en ese momento.<br><br>

<strong>Criterio de clasificación:</strong><br>
ΔR = {delta_r:+.5f}. Mediana del dataset de entrenamiento = −0.005885.<br>
{'ΔR > mediana → red neuronal más fluida durante la tarea → G (Eficiente).' if grupo == 'G' 
else 'ΔR ≤ mediana → red neuronal más rígida durante la tarea → B (Inestable).'}
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="info-box">
<em>Fun fact:</em> Uno pensaría que alguien bueno en aritmética mental trabajaría 
más el cerebro. Los datos muestran lo contrario: los cerebros eficientes trabajan menos 
y producen más. Los inestables hacen más esfuerzo de reorganización con menos resultado. 
Esto se debe a cómo se comunica el cerebro entre nodos — el cerebro eficiente no necesita 
reorganizarse al enfrentar el problema porque su red ya está lista.
</div>
""", unsafe_allow_html=True)

            with st.expander("Información técnica"):
                st.markdown(f"""
**Archivos:**
- Reposo: `{archivo_reposo.name}` · {n_ch_rep} canales · {fs_rep:.0f} Hz
- Tarea: `{archivo_tarea.name}` · {n_ch_tar} canales · {fs_tar:.0f} Hz

**Motor Omega:** Transformada de Hilbert · broadband · primeros 23 canales

**SPG:** Embedding de retardo (d=4) → red de recurrencia (percentil 20) → 
Laplaciano → λ₂ y C_dyn = Var(τ) · Canal {ch} · Ventanas 4s/2s

**Clasificador:** ΔR > −0.005885 → G · Criterio: mediana de ΔR en 36 sujetos 
(EEG During Mental Arithmetic Tasks, PhysioNet)

**Dataset:** https://physionet.org/content/eegmat/1.0.0/
""")

else:
    st.markdown("""
<div style="text-align:center; color:#999; font-family:'IBM Plex Mono',monospace; 
font-size:13px; padding:48px 0;">
Sube los dos archivos EDF para comenzar
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
EEG Cognitive Load Classifier · Edher Alan Arteaga Marroquin · 2026<br>
Motor Omega · SPG Framework · PhysioNet EEGMAT Dataset
</div>
""", unsafe_allow_html=True)
