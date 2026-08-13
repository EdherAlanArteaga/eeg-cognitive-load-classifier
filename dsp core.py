"""
NÚCLEO v3 — Dynamic Spectral Persistence
=========================================
Implementa los observables de "Dynamic Spectral Persistence in Heterogeneous
Networks: From Hub Blindness to Heat Kernel Identity".

  V_i   = Σ_{k≥2} v_k(i)² / λ_k          visibilidad espectral
  V_i   = ∫₀^∞ R_i^⊥(t) dt               identidad exacta (Teorema 2.1)
  R_i^⊥(t) = Σ_{k≥2} e^{-tλ_k} v_k(i)²   retornabilidad transversa
  τ_i   = M₂/M₁                          timescale de persistencia
  τ̃_i   = λ₂ · τ_i                       timescale normalizado

Sustituye a v1 (cuaterniónico R,C) y v2 (C_dyn = λ₂²·G_pure).
"""
import numpy as np
from scipy.linalg import eigh, expm
from scipy.integrate import quad
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr


def dsp(A):
    """
    Dynamic Spectral Persistence de una matriz de adyacencia.
    Devuelve V_i, τ_i, τ̃_i por nodo más los agregados espectrales.
    """
    A = np.array(A, float)
    np.fill_diagonal(A, 0)
    N = A.shape[0]
    deg = A.sum(1)
    L = np.diag(deg) - A
    ev, evec = eigh(L)
    k0 = next((k for k in range(1, N) if ev[k] > 1e-8), None)
    if k0 is None:
        return None                      # grafo desconectado

    M1 = np.zeros(N)
    M2 = np.zeros(N)
    for k in range(k0, N):
        if ev[k] < 1e-10:
            continue
        v2 = evec[:, k] ** 2
        M1 += v2 / ev[k]
        M2 += v2 / ev[k] ** 2

    lam2 = ev[k0]
    V = M1                                # visibilidad espectral (sin normalizar)
    tau = np.where(M1 > 1e-14, M2 / M1, 0.0)
    tau_tilde = lam2 * tau

    return dict(V=V, V_norm=V / V.sum(), tau=tau, tau_tilde=tau_tilde,
                lambda2=lam2, lambda_max=ev[-1], rango=ev[-1] / lam2,
                deg=deg, ev=ev, evec=evec, k0=k0, N=N)


def returnability(d, i, t):
    """R_i^⊥(t) = Σ_{k≥2} e^{-tλ_k} v_k(i)²  — diagonal del heat kernel."""
    ev, evec, k0 = d['ev'], d['evec'], d['k0']
    return float(np.sum(np.exp(-ev[k0:] * t) * evec[i, k0:] ** 2))


def verificar_identidad(d, nodos=None):
    """Teorema 2.1: V_i == ∫₀^∞ R_i^⊥(t) dt. Debe dar error < 1e-13."""
    if nodos is None:
        nodos = range(min(d['N'], 6))
    errores = []
    for i in nodos:
        integral, _ = quad(lambda t: returnability(d, i, t), 0, np.inf, limit=200)
        errores.append(abs(d['V'][i] - integral))
    return float(np.max(errores))


def anti_centralidad(d):
    """
    Resultado central del paper: grado y persistencia anti-correlacionan.
    Referencia: Spearman(grado, V) ≈ -0.96 en redes heterogéneas.
    """
    if np.std(d['deg']) < 1e-12:
        return np.nan, np.nan
    rho_V, _ = spearmanr(d['deg'], d['V'])
    rho_tt, _ = spearmanr(d['deg'], d['tau_tilde'])
    return float(rho_V), float(rho_tt)


def red_recurrencia(x, d=4, N_max=40, pct=20):
    """Señal 1D -> red de recurrencia en espacio de fase (embedding de retardo)."""
    x = np.asarray(x, float)
    if x.std() < 1e-10:
        return None
    x = np.clip((x - x.mean()) / (x.std() + 1e-10), -4, 4)
    pts = np.array([x[i:i + d] for i in range(len(x) - d + 1)])
    if len(pts) < 6:
        return None
    if len(pts) > N_max:
        pts = pts[np.linspace(0, len(pts) - 1, N_max).astype(int)]
    D = cdist(pts, pts)
    D0 = D.copy()
    np.fill_diagonal(D, np.inf)
    eps = np.quantile(D[D < np.inf], pct / 100)
    A = (D < eps).astype(float)
    np.fill_diagonal(A, 0)
    for i in np.where(A.sum(1) == 0)[0]:
        j = np.argsort(D0[i])[1]
        A[i, j] = A[j, i] = 1.0
    return A


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import networkx as nx

    print("=" * 74)
    print("V3-T1 — Identidad del heat kernel  V_i = ∫₀^∞ R_i^⊥(t) dt")
    print("=" * 74)
    print("  Paper: error < 5e-15, Pearson = 1.000000 para N = 8,12,16,20")
    print(f"  {'grafo':<24} {'N':>4} {'error máx':>12}")
    for nombre, G in [("Star S_8", nx.star_graph(7)),
                      ("Star S_16", nx.star_graph(15)),
                      ("ER p=0.3 N=12", nx.erdos_renyi_graph(12, .3, seed=1)),
                      ("ER p=0.6 N=20", nx.erdos_renyi_graph(20, .6, seed=1)),
                      ("BA m=2 N=20", nx.barabasi_albert_graph(20, 2, seed=1))]:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        d = dsp(nx.to_numpy_array(G))
        err = verificar_identidad(d)
        print(f"  {nombre:<24} {d['N']:>4} {err:>12.2e}")

    print()
    print("=" * 74)
    print("V3-T2 — Fórmula cerrada del hub  R_hub(t) = 1/N + [(N-1)/N]e^{-Nt}")
    print("=" * 74)
    print("  Paper Teorema 3.1: error máx 4.34e-15")
    N = 16
    d = dsp(nx.to_numpy_array(nx.star_graph(N - 1)))
    print(f"  {'t':>6} {'R_hub medido':>14} {'R_hub teórico':>15} {'error':>11}")
    errs = []
    for t in [0.1, 0.5, 1.0, 2.0, 5.0]:
        medido = returnability(d, 0, t) + 1 / N     # + modo cero
        teorico = 1 / N + ((N - 1) / N) * np.exp(-N * t)
        errs.append(abs(medido - teorico))
        print(f"  {t:>6.1f} {medido:>14.9f} {teorico:>15.9f} {abs(medido-teorico):>11.2e}")
    print(f"  error máximo = {max(errs):.2e}")

    print()
    print("=" * 74)
    print("V3-T3 — Ceguera espectral: contribución de Fiedler al hub = 0")
    print("=" * 74)
    ev, evec, k0 = d['ev'], d['evec'], d['k0']
    fiedler = [k for k in range(k0, d['N']) if abs(ev[k] - 1.0) < 1e-9]
    amp_hub = np.abs(evec[0, fiedler]).max()
    print(f"  Modos de Fiedler (λ=1): {len(fiedler)}")
    print(f"  Amplitud máx del hub en esos modos: {amp_hub:.2e}")
    print(f"  -> {'PASA (cero algebraico)' if amp_hub < 1e-12 else 'FALLA'}")

    print()
    print("=" * 74)
    print("V3-T4 — Anti-centralidad  Spearman(grado, V)")
    print("=" * 74)
    print("  Paper Tabla 2: Star = -1.000, ER = -0.90 a -0.97")
    print(f"  {'grafo':<24} {'ρ(k,V)':>10} {'ρ(k,τ̃)':>10}")
    for nombre, G in [("Star S_12", nx.star_graph(11)),
                      ("ER p=0.3 N=12", nx.erdos_renyi_graph(12, .3, seed=2)),
                      ("ER p=0.6 N=12", nx.erdos_renyi_graph(12, .6, seed=2)),
                      ("BA m=2 N=100", nx.barabasi_albert_graph(100, 2, seed=2))]:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        dd = dsp(nx.to_numpy_array(G))
        rv, rt = anti_centralidad(dd)
        print(f"  {nombre:<24} {rv:>10.3f} {rt:>10.3f}")

    print()
    print("=" * 74)
    print("V3-T5 — ¿Las redes de recurrencia de EEG cumplen lo mismo?")
    print("=" * 74)
    rng = np.random.default_rng(5)
    fs = 100.0
    t = np.arange(int(4 * fs)) / fs

    señales = {
        "alfa 10 Hz + ruido": np.sin(2*np.pi*10*t) + .5*rng.standard_normal(len(t)),
        "ruido rosa":         np.cumsum(rng.standard_normal(len(t))) * .05,
        "ruido blanco":       rng.standard_normal(len(t)),
        "mezcla alfa+beta":   (np.sin(2*np.pi*10*t) + .6*np.sin(2*np.pi*22*t)
                               + .3*rng.standard_normal(len(t))),
    }
    print(f"  {'señal':<22} {'N':>4} {'ident.err':>11} {'ρ(k,V)':>9} "
          f"{'λ₂':>8} {'⟨τ̃⟩':>8} {'Var(τ̃)':>9}")
    for nombre, x in señales.items():
        A = red_recurrencia(x)
        dd = dsp(A)
        if dd is None:
            print(f"  {nombre:<22} desconectada")
            continue
        err = verificar_identidad(dd, nodos=range(5))
        rv, _ = anti_centralidad(dd)
        print(f"  {nombre:<22} {dd['N']:>4} {err:>11.2e} {rv:>9.3f} "
              f"{dd['lambda2']:>8.4f} {dd['tau_tilde'].mean():>8.4f} "
              f"{dd['tau_tilde'].var():>9.5f}")

    print()
    print("  Si ρ(k,V) es fuertemente negativo, la red de recurrencia se comporta")
    print("  como las redes heterogéneas del paper y v3 aplica a EEG.")
