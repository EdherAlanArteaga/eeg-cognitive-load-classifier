"""
COMPARACIÓN COMPLETA — 36 sujetos, v2 (C_dyn) vs v3 (Dynamic Spectral Persistence)
====================================================================================
Corre en Google Colab, en la MISMA sesión donde ya tengas:
  - eeg_data/SubjectNN_1.edf y _2.edf (los 72 archivos de PhysioNet EEGMAT)
  - df_final_real cargado (Sujeto, Grupo, Restas_por_minuto)

Pipeline: v5 del notebook original — hasta 180s de reposo / 62s de tarea,
canal Fz (índice 16), sin decimar, 40 ventanas máx, embedding d=4, percentil 20.
Es el que más señal aprovecha de cada sujeto.

Calcula, por sujeto:
  v2 (spg_orig):  C_dyn, lambda2, G_pure, CI, d_eff   (reposo, tarea, delta)
  v3 (SPG class): V (visibilidad), tau, tau_tilde, anti-centralidad rho(k,V)
                  (agregados: media de V, media de tau_tilde, Var(tau_tilde))

Guarda comparacion_36_sujetos.csv y muestra las correlaciones de TODO
contra restas_por_minuto, más el crosstab contra el grupo G/B real.
"""

import numpy as np
import pandas as pd
import pyedflib
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.stats import spearmanr, pearsonr

CANALES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
           'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
PREF = ['Fz','FZ','F3','Cz','CZ','C3']
IDX_CH = CANALES.index('Fz')   # 16, igual que el notebook original


# ── Carga de EDF, igual que v5 (trunca a lo disponible) ─────────────────────
def cargar_edf(ruta, duracion_segundos=180):
    f = pyedflib.EdfReader(ruta)
    fs = f.getSampleFrequency(0)
    n_disp = f.getNSamples()[0]
    n = min(int(fs * duracion_segundos), n_disp)
    n_ch = min(f.signals_in_file, 23)
    sig = np.zeros((n_ch, n))
    for i in range(n_ch):
        sig[i, :] = f.readSignal(i, n=n)
    f.close()
    return sig, fs


# ── v2 — spg_orig, EXACTO al notebook celda 46 ──────────────────────────────
def spg_orig(signal, fs, win_s=4, step_s=2, max_wins=40, d=4, N_max=40):
    win = int(win_s * fs); step = int(step_s * fs)
    Cs, lam2s, Gs = [], [], []
    np.random.seed(42)
    for s in range(0, min(len(signal) - win, max_wins * step), step):
        x = signal[s:s + win]
        if x.std() < 1e-10:
            continue
        x = (x - x.mean()) / (x.std() + 1e-10)
        x = np.clip(x, -4, 4)
        pts = np.array([x[i:i + d] for i in range(len(x) - d + 1)])
        if len(pts) > N_max:
            idx = np.linspace(0, len(pts) - 1, N_max).astype(int)
            pts = pts[idx]
        N = len(pts); D = cdist(pts, pts); D0 = D.copy()
        np.fill_diagonal(D, np.inf)
        eps = np.quantile(D[D < np.inf], 0.20)
        Adj = (D < eps).astype(float); np.fill_diagonal(Adj, 0)
        for i in np.where(Adj.sum(1) == 0)[0]:
            j = np.argsort(D0[i])[1]
            Adj[i, j] = Adj[j, i] = 1
        L = np.diag(Adj.sum(1)) - Adj
        ev, evec = eigh(L)
        k0 = next((k for k in range(1, N) if ev[k] > 1e-8), None)
        if k0 is None:
            continue
        lam2 = ev[k0]; M1 = np.zeros(N); M2 = np.zeros(N)
        for k in range(k0, N):
            lk = ev[k]
            if lk < 1e-10:
                continue
            v2 = evec[:, k] ** 2
            M1 += v2 / lk; M2 += v2 / lk ** 2
        tt = np.where(M1 > 1e-14, lam2 * (M2 / M1), 0)
        C = float(np.var(tt))
        geom = np.where(M1 > 1e-14, M2 / M1, 0)
        G = float(np.var(geom))
        if np.isfinite(C) and C > 0:
            Cs.append(C); lam2s.append(lam2); Gs.append(G)
    if len(Cs) < 5:
        return None
    r, _ = spearmanr(lam2s, Cs)
    X = np.column_stack([
        (np.array(lam2s) - np.mean(lam2s)) / (np.std(lam2s) + 1e-12),
        (np.array(Cs) - np.mean(Cs)) / (np.std(Cs) + 1e-12)])
    ev2 = np.maximum(np.linalg.eigvalsh(np.cov(X.T)), 0)
    d_eff = float(ev2.sum() ** 2 / (ev2 ** 2).sum()) if (ev2 ** 2).sum() > 0 else np.nan
    return dict(C_mean=float(np.mean(Cs)), L_mean=float(np.mean(lam2s)),
                G_pure=float(np.mean(Gs)), CI=abs(float(r)),
                d_eff=d_eff, n_wins=len(Cs))


# ── v3 — clase SPG, EXACTA a Comparación_V,Tau_contra_literatura celda 3 ────
class SPG:
    """V = M1 (visibilidad espectral) ; tau = M2/M1 ; tau_tilde = lambda2*tau.
    Convención verificada contra el control de la estrella: tau_hub=1/N."""
    def __init__(self, A):
        A = np.array(A, float); np.fill_diagonal(A, 0)
        N = A.shape[0]; deg = A.sum(1)
        ev, evec = eigh(np.diag(deg) - A)
        k0 = next((k for k in range(1, N) if ev[k] > 1e-8), None)
        if k0 is None:
            raise ValueError("grafo desconectado")
        M1 = np.zeros(N); M2 = np.zeros(N)
        for k in range(k0, N):
            if ev[k] < 1e-10:
                continue
            v2 = evec[:, k] ** 2
            M1 += v2 / ev[k]; M2 += v2 / ev[k] ** 2
        self.V = M1
        self.tau = np.where(M1 > 1e-14, M2 / M1, 0)
        self.tau_tilde = ev[k0] * self.tau
        self.degree = deg
        self.lambda2 = ev[k0]
        self.N = N


def red_recurrencia(x, d=4, N_max=40, pct=20):
    x = np.clip((x - x.mean()) / (x.std() + 1e-10), -4, 4)
    pts = np.array([x[i:i + d] for i in range(len(x) - d + 1)])
    if len(pts) > N_max:
        pts = pts[np.linspace(0, len(pts) - 1, N_max).astype(int)]
    D = cdist(pts, pts); D0 = D.copy()
    np.fill_diagonal(D, np.inf)
    eps = np.quantile(D[D < np.inf], pct / 100)
    A = (D < eps).astype(float); np.fill_diagonal(A, 0)
    for i in np.where(A.sum(1) == 0)[0]:
        j = np.argsort(D0[i])[1]
        A[i, j] = A[j, i] = 1.0
    return A


def spg_v3_serie(signal, fs, win_s=4, step_s=2, max_wins=40):
    """Aplica la clase SPG (v3) ventana por ventana y agrega."""
    win = int(win_s * fs); step = int(step_s * fs)
    V_means, tt_means, tt_vars, lam2s, rhos = [], [], [], [], []
    for s in range(0, min(len(signal) - win, max_wins * step), step):
        x = signal[s:s + win]
        if x.std() < 1e-10:
            continue
        A = red_recurrencia(x)
        try:
            sp = SPG(A)
        except ValueError:
            continue
        V_means.append(sp.V.mean())
        tt_means.append(sp.tau_tilde.mean())
        tt_vars.append(sp.tau_tilde.var())
        lam2s.append(sp.lambda2)
        if np.std(sp.degree) > 1e-12:
            rhos.append(spearmanr(sp.degree, sp.V)[0])
    if len(tt_means) < 5:
        return None
    return dict(V_mean=float(np.mean(V_means)),
                tau_tilde_mean=float(np.mean(tt_means)),
                tau_tilde_var=float(np.mean(tt_vars)),
                lambda2_mean=float(np.mean(lam2s)),
                rho_kV=float(np.mean(rhos)) if rhos else np.nan,
                n_wins=len(tt_means))


# ── Loop principal sobre los 36 sujetos ──────────────────────────────────────
print("Procesando 36 sujetos — v2 (spg_orig) y v3 (clase SPG) — canal Fz, 180s/62s")
print("=" * 78)

filas = []
for suj in df_final_real.index:
    try:
        s_rep, fs = cargar_edf(f'eeg_data/{suj}_1.edf', 180)
        s_tar, _ = cargar_edf(f'eeg_data/{suj}_2.edf', 62)

        ch_rep, ch_tar = s_rep[IDX_CH], s_tar[IDX_CH]

        # v2
        a2 = spg_orig(ch_rep, fs)
        b2 = spg_orig(ch_tar, fs)

        # v3
        a3 = spg_v3_serie(ch_rep, fs)
        b3 = spg_v3_serie(ch_tar, fs)

        if None in (a2, b2, a3, b3):
            print(f"  ✗ {suj}: señal insuficiente en alguna versión")
            continue

        filas.append({
            'Sujeto': suj,
            # v2 — C_dyn / manifold
            'v2_Cdyn_rest': a2['C_mean'], 'v2_Cdyn_task': b2['C_mean'],
            'v2_Delta_Cdyn': b2['C_mean'] - a2['C_mean'],
            'v2_Delta_deff': b2['d_eff'] - a2['d_eff'],
            'v2_Delta_CI': b2['CI'] - a2['CI'],
            # v3 — Dynamic Spectral Persistence
            'v3_V_rest': a3['V_mean'], 'v3_V_task': b3['V_mean'],
            'v3_Delta_V': b3['V_mean'] - a3['V_mean'],
            'v3_tautilde_rest': a3['tau_tilde_mean'], 'v3_tautilde_task': b3['tau_tilde_mean'],
            'v3_Delta_tautilde': b3['tau_tilde_mean'] - a3['tau_tilde_mean'],
            'v3_Delta_var_tautilde': b3['tau_tilde_var'] - a3['tau_tilde_var'],
            'v3_rho_kV_rest': a3['rho_kV'], 'v3_rho_kV_task': b3['rho_kV'],
        })
        print(f"  ✓ {suj}")
    except Exception as e:
        print(f"  ✗ {suj}: {type(e).__name__}: {e}")

df_cmp = pd.DataFrame(filas).set_index('Sujeto')
df_cmp = df_cmp.join(df_final_real[['Grupo', 'Restas_por_minuto']])
df_cmp.to_csv('comparacion_36_sujetos.csv')
print(f"\nGuardado: comparacion_36_sujetos.csv  ({len(df_cmp)} sujetos)")

# ── Correlación de TODO contra restas_por_minuto ────────────────────────────
print("\n" + "=" * 78)
print("CORRELACIÓN CON RENDIMIENTO (restas/min) — Spearman")
print("=" * 78)
cols = [c for c in df_cmp.columns if c not in ('Grupo', 'Restas_por_minuto')]
print(f"{'Observable':<26}{'r':>8}{'p':>10}   {'marco'}")
print("-" * 60)
for c in cols:
    sub = df_cmp[[c, 'Restas_por_minuto']].dropna()
    if len(sub) < 5:
        continue
    r, p = spearmanr(sub[c], sub['Restas_por_minuto'])
    marco = 'v2 (C_dyn/manifold)' if c.startswith('v2') else 'v3 (DSP)'
    marca = ' *' if p < 0.05 else ''
    print(f"{c:<26}{r:>8.3f}{p:>10.4f}{marca}   {marco}")

# ── ¿Algo separa G de B directamente? Mann-Whitney ──────────────────────────
from scipy.stats import mannwhitneyu
print("\n" + "=" * 78)
print("¿SEPARA G vs B? — Mann-Whitney U")
print("=" * 78)
print(f"{'Observable':<26}{'U':>10}{'p':>10}   {'marco'}")
print("-" * 60)
g = df_cmp[df_cmp['Grupo'].astype(str).str.startswith('G')]
b = df_cmp[df_cmp['Grupo'].astype(str).str.startswith('B')]
for c in cols:
    sub_g = g[c].dropna(); sub_b = b[c].dropna()
    if len(sub_g) < 3 or len(sub_b) < 3:
        continue
    u, p = mannwhitneyu(sub_g, sub_b, alternative='two-sided')
    marco = 'v2 (C_dyn/manifold)' if c.startswith('v2') else 'v3 (DSP)'
    marca = ' *' if p < 0.05 else ''
    print(f"{c:<26}{u:>10.1f}{p:>10.4f}{marca}   {marco}")

# ── Correlación parcial ΔC_dyn | reposo, la única con soporte previo ────────
def parcial(x, y, z):
    rxy, _ = spearmanr(x, y); rxz, _ = spearmanr(x, z); ryz, _ = spearmanr(y, z)
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))

print("\n" + "=" * 78)
print("CONTROL: replicando el único resultado con p<0.05 documentado")
print("=" * 78)
sub = df_cmp[['v2_Delta_Cdyn', 'Restas_por_minuto', 'v2_Cdyn_rest']].dropna()
rp = parcial(sub['v2_Delta_Cdyn'], sub['Restas_por_minuto'], sub['v2_Cdyn_rest'])
print(f"ΔC_dyn | ctrl REST: r_parcial = {rp:.3f}  (documento: +0.335, p=0.046)")
print(f"n = {len(sub)}")

print("\nHecho. Revisa comparacion_36_sujetos.csv para el detalle sujeto por sujeto.")
