# Dynamic Spectral Persistence — Especificación de implementación

**Autor del marco:** Edher Alan Arteaga Marroquin
**Propósito de este documento:** especificación completa para que una IA implemente
el marco sobre cualquier red — colaboración musical, conectomas, redes metabólicas,
transacciones, infraestructura — o sobre cualquier serie temporal.

> **Cómo usar este documento.** Pégalo completo como contexto. Contiene las
> definiciones, el algoritmo, la batería de validación con números esperados, y
> los ocho errores que ya se cometieron y están documentados. Si tu implementación
> no reproduce la sección 6 exactamente, está mal — no sigas hasta arreglarla.

---

## Abstract (EN)

Spectral visibility V_i of a node equals the integrated diagonal of the heat kernel
restricted to the transverse subspace: V_i = ∫₀^∞ R_i^⊥(t) dt. This identity connects
a static spectral object to a dynamical quantity, and places it within observability
and diffusion geometry theory. The central empirical claim is anti-centrality: in
heterogeneous networks, node degree and dynamic persistence are systematically
anti-correlated. Classical centralities identify *transit* nodes; spectral visibility
identifies *persistence* nodes. The hub is a router, not a reservoir.

---

## 1. Objetos matemáticos

Sea G = (V, E) un grafo **conexo**, **no dirigido**, **sin auto-lazos**, con N nodos.

### 1.1 Laplaciano combinatorio

```
L = D − A
```

donde A es la matriz de adyacencia y D = diag(k₁, …, k_N) con kᵢ el grado del nodo i.

Espectro: `0 = λ₁ < λ₂ ≤ … ≤ λ_N`, autovectores ortonormales `v₁, …, v_N`.

λ₂ es la **conectividad algebraica**. El subespacio transverso es
`H_⊥ = {x : ⟨x, 1⟩ = 0}`, de dimensión N−1. Todo lo que sigue vive ahí: el modo
cero (λ₁ = 0) se excluye siempre.

> **Crítico:** debe ser el Laplaciano **combinatorio**. Ver §7.1 — con el
> normalizado el fenómeno central desaparece.

### 1.2 Momentos espectrales

```
M₁(i) = Σ_{k≥2} v_k(i)² / λ_k
M₂(i) = Σ_{k≥2} v_k(i)² / λ_k²
```

### 1.3 Los cuatro observables

| Símbolo | Definición | Qué mide |
|---|---|---|
| `V_i` | `M₁(i)` | Visibilidad espectral — energía modal persistente acumulada |
| `τ_i` | `M₂(i) / M₁(i)` | Timescale de persistencia |
| `τ̃_i` | `λ₂ · τ_i` | Timescale normalizado por la escala global de la red |
| `R_i^⊥(t)` | `Σ_{k≥2} e^{−t λ_k} v_k(i)²` | Retornabilidad transversa (diagonal del heat kernel) |

V normalizada, cuando se quiera comparar entre redes:
`V_i^norm = V_i / Σ_j V_j`

### 1.4 El teorema central

```
V_i = ∫₀^∞ R_i^⊥(t) dt          (identidad exacta)
```

**Demostración.** Cálculo directo:

```
∫₀^∞ R_i^⊥(t) dt = Σ_{k≥2} v_k(i)² ∫₀^∞ e^{−λ_k t} dt
                 = Σ_{k≥2} v_k(i)² / λ_k
                 = V_i
```

Converge porque λ_k > 0 para todo k ≥ 2. ∎

Tres consecuencias:

1. `V_i` es la diagonal del resolvente del Laplaciano `(L⁺)_ii` restringido al
   subespacio transverso — no es una construcción ad hoc.
2. Equivale a la diagonal del **Gramiano de observabilidad** del sistema
   `dα/dt = −Lα` observado en el nodo i. V alta ⟹ buen punto de medición.
3. Los nodos de V alta son exactamente los que acumulan energía modal
   sobre tiempo infinito.

### 1.5 Anti-centralidad

La afirmación empírica central:

> En redes heterogéneas, **grado y persistencia anti-correlacionan**.

Las centralidades clásicas (grado, betweenness, PageRank, Katz, random walk)
miden **tránsito**: cuánto flujo pasa por un nodo. V mide **persistencia**:
cuánta energía sostiene. En redes con grados heterogéneos son
sistemáticamente opuestas.

Interpretación física: **el hub es un router, no un reservorio.** Su alta
conectividad hace que cualquier energía depositada en él se disperse de
inmediato por el modo de eigenvalor alto. Los nodos periféricos participan
en modos de decaimiento lento y sostienen energía.

---

## 2. Algoritmo — redes directas

Entrada: lista de aristas o matriz de adyacencia.

```
1. Construir grafo no dirigido
2. Eliminar auto-lazos
3. Colapsar aristas duplicadas (binarizar)
4. Tomar la componente conexa mayor          ← obligatorio, ver §7.3
5. A ← matriz de adyacencia, diagonal a cero
6. deg ← A.sum(axis=1)
7. L ← diag(deg) − A
8. λ, v ← eigh(L)                            ← simétrico: usar eigh, no eig
9. k₀ ← primer índice con λ[k] > 1e-8        ← detecta desconexión residual
10. M₁, M₂ ← acumular sobre k ≥ k₀
11. V ← M₁ ; τ ← M₂/M₁ ; τ̃ ← λ₂·τ
12. Verificar rango λ_max/λ₂                 ← ver §7.2
```

### Implementación de referencia

```python
import numpy as np
from scipy.linalg import eigh

def dsp(A):
    """Dynamic Spectral Persistence de una matriz de adyacencia."""
    A = np.array(A, float)
    np.fill_diagonal(A, 0)
    N = A.shape[0]
    deg = A.sum(1)
    ev, evec = eigh(np.diag(deg) - A)

    k0 = next((k for k in range(1, N) if ev[k] > 1e-8), None)
    if k0 is None:
        raise ValueError("grafo desconectado: λ₂ ≈ 0")

    M1 = np.zeros(N)
    M2 = np.zeros(N)
    for k in range(k0, N):
        if ev[k] < 1e-10:
            continue
        v2 = evec[:, k] ** 2
        M1 += v2 / ev[k]
        M2 += v2 / ev[k] ** 2

    lam2 = ev[k0]
    tau = np.where(M1 > 1e-14, M2 / M1, 0.0)

    return dict(V=M1, V_norm=M1 / M1.sum(),
                tau=tau, tau_tilde=lam2 * tau,
                lambda2=lam2, lambda_max=ev[-1],
                rango=ev[-1] / lam2, deg=deg,
                ev=ev, evec=evec, k0=k0, N=N)


def returnability(d, i, t):
    """R_i^⊥(t) — diagonal del heat kernel, modo cero excluido."""
    return float(np.sum(np.exp(-d['ev'][d['k0']:] * t)
                        * d['evec'][i, d['k0']:] ** 2))
```

---

## 3. Algoritmo — series temporales

Para aplicar el marco a una señal (EEG, precio, sismograma, audio) hay que
convertirla primero en red mediante **embedding de retardo** y
**red de recurrencia**.

```
1. Preprocesar: filtrar banda de interés, z-score
2. Remuestrear a una frecuencia fija               ← ver §7.4
3. Ventanear: ventanas de W segundos, paso S
4. Por ventana:
     a. z-score y clip a ±4σ
     b. embedding de retardo dimensión d:
          pts[i] = x[i : i+d]
     c. submuestrear a N_max puntos si hay más
     d. D ← matriz de distancias euclidianas entre puntos
     e. ε ← percentil P de las distancias fuera de la diagonal
     f. A ← (D < ε), diagonal a cero
     g. conectar nodos aislados a su vecino más cercano
     h. aplicar dsp(A)
5. Agregar sobre ventanas: media de ⟨τ̃⟩, λ₂, Var(τ̃), ρ(k,V)
```

Parámetros que funcionan como punto de partida:

| Parámetro | Valor | Nota |
|---|---|---|
| ventana W | 4 s | |
| paso S | 2 s | solape del 50 % |
| máx ventanas | 40 | costo O(N³) por ventana |
| dimensión d | 4 | embedding de retardo |
| N_max | 40 | nodos por red |
| percentil P | 20 | densidad de la red resultante |
| frecuencia | 100 Hz | fija, ver §7.4 |

```python
from scipy.spatial.distance import cdist

def red_recurrencia(x, d=4, n_max=40, pct=20):
    x = np.asarray(x, float)
    if x.std() < 1e-10:
        return None                       # canal muerto
    x = np.clip((x - x.mean()) / (x.std() + 1e-10), -4, 4)

    pts = np.array([x[i:i+d] for i in range(len(x) - d + 1)])
    if len(pts) < 6:
        return None
    if len(pts) > n_max:
        pts = pts[np.linspace(0, len(pts)-1, n_max).astype(int)]

    D = cdist(pts, pts)
    D0 = D.copy()
    np.fill_diagonal(D, np.inf)
    eps = np.quantile(D[D < np.inf], pct / 100)

    A = (D < eps).astype(float)
    np.fill_diagonal(A, 0)
    for i in np.where(A.sum(1) == 0)[0]:   # conectar aislados
        j = np.argsort(D0[i])[1]
        A[i, j] = A[j, i] = 1.0
    return A
```

---

## 4. Observables agregados por red

Para comparar redes o estados entre sí:

| Observable | Fórmula | Interpretación |
|---|---|---|
| `⟨τ̃⟩` | media de τ̃ sobre nodos | persistencia típica |
| `Var(τ̃)` | varianza de τ̃ sobre nodos | heterogeneidad reservorio/router |
| `λ₂` | conectividad algebraica | integración de la dinámica |
| `V_max/V_min` | razón | contraste de persistencia |
| `ρ(k, V)` | Spearman **y** Pearson | control estructural, ver §7.5 |

En series temporales, cada uno se promedia sobre ventanas, y `σ_τ` (desviación
de ⟨τ̃⟩ **entre** ventanas) mide plasticidad temporal.

---

## 5. Fórmulas cerradas del grafo estrella S_N

Sirven como control analítico exacto. S_N tiene un hub y N−1 hojas.

**Espectro:**
```
λ₁ = 0
λ₂ = … = λ_{N−1} = 1        (subespacio de Fiedler, multiplicidad N−2)
λ_N = N                      (modo alto)
```

**Hecho estructural.** Los autovectores de Fiedler cumplen `(v_k)_hub = 0` para
todo k = 2…N−1. El hub tiene amplitud **exactamente cero** en todos los modos
de Fiedler. Es consecuencia de la simetría S_{N−1}: el hub transforma
trivialmente mientras que el subespacio de Fiedler porta la representación
estándar, y el lema de Schur garantiza ortogonalidad entre representaciones
irreducibles distintas.

**Retornabilidad, incluyendo el modo cero:**
```
R_hub(t)  = 1/N + [(N−1)/N] · e^{−Nt}
R_leaf(t) = 1/N + e^{−t}/(N−1) − e^{−Nt}/N     (promediado sobre hojas)
```

**Timescales:**
```
τ_hub  = 1/N
τ_leaf = 1
```

Separación de N veces. Para N = 20 el hub alcanza el equilibrio 20 veces
más rápido que las hojas.

---

## 6. Batería de validación

Tu implementación **debe** reproducir esto. Números medidos con
`numpy` + `scipy.linalg.eigh`, doble precisión.

### T1 — Identidad del heat kernel

`V_i` vs `∫₀^∞ R_i^⊥(t) dt` por cuadratura numérica (`scipy.integrate.quad`,
límite superior infinito, `limit=200`).

| Grafo | N | Error máximo esperado |
|---|---|---|
| Star S_8 | 8 | 5.66e-15 |
| Star S_16 | 16 | 2.22e-16 |
| ER p=0.3 | 12 | 1.18e-14 |
| ER p=0.6 | 20 | 7.95e-15 |
| BA m=2 | 20 | 5.85e-14 |

Criterio de aprobación: **error < 1e-13** en todos.

### T2 — Fórmula cerrada del hub, S_16

`returnability(d, hub, t) + 1/N` contra `1/N + [(N−1)/N]e^{−Nt}`:

| t | R_hub |
|---|---|
| 0.1 | 0.251777986 |
| 0.5 | 0.062814496 |
| 1.0 | 0.062500106 |
| 2.0 | 0.062500000 |
| 5.0 | 0.062500000 |

Error esperado: **0.00e+00**.

### T3 — Ceguera espectral

En S_16, amplitud máxima del hub sobre los 14 modos con λ = 1:
**0.00e+00 exacto**.

### T4 — Escalamiento τ_hub = 1/N

| N | τ_hub | 1/N | error |
|---|---|---|---|
| 6 | 0.166667 | 0.166667 | 2.78e-17 |
| 8 | 0.125000 | 0.125000 | 0.00e+00 |
| 12 | 0.083333 | 0.083333 | 0.00e+00 |
| 16 | 0.062500 | 0.062500 | 0.00e+00 |
| 20 | 0.050000 | 0.050000 | 6.94e-18 |
| 25 | 0.040000 | 0.040000 | 6.94e-18 |

Y `τ_leaf = 0.999868` en S_20 (teoría: 1.0).

### T5 — Anti-centralidad

| Grafo | Pearson(k,V) | Spearman(k,V) |
|---|---|---|
| Star S_12 | **−1.0000** | −0.4864 |
| ER p=0.3 N=12 | — | −0.967 |
| ER p=0.6 N=12 | — | −0.984 |
| BA m=2 N=100 | — | −0.932 |

La discrepancia en Star entre los dos estadísticos **no es un error**. Ver §7.5.

### T6 — Redes de recurrencia desde series temporales

Señales sintéticas a 100 Hz, ventana de 4 s, parámetros de §3:

| Señal | N | error identidad | Spearman(k,V) | λ₂ |
|---|---|---|---|---|
| seno 10 Hz + ruido | 40 | 5.20e-14 | −0.965 | 0.414 |
| ruido rosa | 40 | 2.97e-15 | −0.958 | 0.358 |
| ruido blanco | 40 | 8.82e-14 | −0.994 | 0.673 |
| 10 Hz + 22 Hz + ruido | 40 | 1.28e-15 | −0.895 | 0.285 |

Esto establece que las redes de recurrencia se comportan como las redes
heterogéneas del marco. Si tu Spearman no sale fuertemente negativo,
revisa el paso de conexión de nodos aislados.

### T7 — Valores de referencia en redes reales

De la corrida sobre redes públicas, con Laplaciano combinatorio:

| Red | N | Spearman(k, L⁺ᵢᵢ) | R²(V, 1/k) | densidad |
|---|---|---|---|---|
| C. elegans metabólica | 453 | −0.962 | 0.772 | 0.020 |
| C. elegans neuronal | 297 | −0.998 | 0.999 | 0.049 |
| Jazz (colaboración) | 198 | −0.998 | 0.951 | 0.141 |
| Food web Little Rock | 183 | −0.996 | 1.000 | 0.146 |
| Bitcoin alpha | 3775 | — | 0.873 | 0.002 |
| Fly larva | 2952 | — | 0.930 | 0.022 |
| Genetic | 683 | — | 0.491 | 0.007 |

Nota que biológicas y no biológicas dan lo mismo:
**el efecto es topológico, no biológico.**

---

## 7. Los ocho errores documentados

Todos se cometieron. Todos están medidos.

### 7.1 El Laplaciano normalizado destruye el fenómeno

Con `L_sym = I − D^{−1/2} A D^{−1/2}` o con el random-walk
`L_rw = I − D^{−1} A`, la anti-centralidad **colapsa**:

| Laplaciano | Spearman(V, grado) | R²(V, 1/k) |
|---|---|---|
| combinatorio | −0.9623 | 0.7715 |
| simétrico normalizado | −0.1818 | 0.0336 |
| random walk | −0.0899 | 0.0307 |

Barrido continuo `L_γ = D^{−γ} L D^{−γ}` con γ ∈ [0, 0.5]: la correlación
se mantiene ≈ −0.96 hasta γ ≈ 0.4 y se desploma cerca de γ = 0.5.

**Usa el combinatorio. Sin excepción.**

### 7.2 Redes con rango espectral degenerado

Cuando `λ_max/λ₂` es muy grande, λ₂ colapsa, M₂/M₁ explota y τ̃ deja de
converger. Medido:

| Red | λ₂ | rango | ¿converge? |
|---|---|---|---|
| BA m=1 N=300 | 0.0063 | 6392.6 | **no** |
| BA m=2 N=300 | 0.574 | 78.7 | sí |
| ER p=0.05 N=300 | 5.31 | 5.7 | sí |

**Umbral práctico: descarta redes o ventanas con λ_max/λ₂ > 1000.**
Reporta cuántas descartaste.

### 7.3 Grafos desconectados

Si el grafo no es conexo, λ₂ = 0 y todo diverge. Toma siempre la
componente conexa mayor **antes** de calcular. Verifica además cuántos
autovalores quedan por debajo de 1e-6 — si hay más de uno, la red sigue
casi desconectada aunque formalmente sea conexa.

### 7.4 `decimate` no sirve para remuestrear

`scipy.signal.decimate` solo acepta factores enteros. Con `fs = 250` y
objetivo 100 Hz calcula `q = round(2.5) = 2` y devuelve **125 Hz**, no 100.
Con `fs = 160` devuelve 80 Hz.

Medido: la misma onda a distintas fs daba d_eff de 1.20 a 1.52 por este
motivo. Usa `resample_poly` con `Fraction`:

```python
from fractions import Fraction
from scipy.signal import resample_poly

f = Fraction(fs_objetivo / fs).limit_denominator(1000)
y = resample_poly(x, f.numerator, f.denominator)
```

### 7.5 Pearson y Spearman no son intercambiables

En un grafo estrella todas las hojas tienen grado 1 — son N−1 empates.
Spearman los penaliza, Pearson no:

```
Star S_12:  Pearson(k, V) = −1.0000
            Spearman(k, V) = −0.4864
```

Ambos son correctos. La literatura reporta Pearson para estrella y Spearman
para ER. **Reporta los dos siempre** y di cuál estás usando.

### 7.6 V ≈ 1/k explica gran parte de la varianza — cuánta depende de la densidad

La objeción obvia: "V solo es 1/grado, es trivial". Medido sobre 11 redes,
el R² del ajuste `V ~ a/k + b` va de **0.38 a 1.00**, y correlaciona con la
densidad de la red (Spearman R² vs densidad = +0.60).

En redes densas V ≈ 1/k casi perfecto. En redes esparsas y modulares queda
un residuo estructural sustancial — hasta 25 % de |V| en la metabólica de
C. elegans. Ese residuo correlaciona con betweenness (+0.62) y core number
(+0.43).

**Siempre reporta R²(V, 1/k) junto con V.** Sin ese número no se puede
saber si tu resultado es geometría o solo grado disfrazado.

### 7.7 El efecto aparece hasta con ruido blanco

Probado sobre estrella S_16: el ratio anti-preferencial aparece con ruido
blanco, ruido rosa, AR(1), Ornstein-Uhlenbeck y fases de Kuramoto — todos
alrededor de 120–130×.

Esto **no invalida** el marco. Confirma que la anti-centralidad es una
propiedad de la **topología** proyectada sobre la señal, no de la señal.
Cualquier señal que viva en el subespacio transverso mostrará supresión
del hub, porque el subespacio de Fiedler tiene componente cero ahí.

Pero significa que **no puedes concluir nada sobre tu señal** a partir de
la anti-centralidad sola. Es un control estructural, no un hallazgo.

### 7.8 El observable no es causal ni depende de fase

Por Parseval, `∫|α_i(t)|² dt = ∫|F_i(ω)|² dω`. El lado derecho depende solo
del espectro de potencia. Consecuencia medida:

- Permutación temporal cambia el resultado en +8.5 %
- Aleatorización de fase lo cambia en +22 %
- Cualquier filtro lineal que preserve el espectro de potencia: ratio 1.000×

No es un fallo. Define qué es el observable: **contenido de energía modal**,
no causalidad temporal ni coherencia de fase. No lo vendas como lo segundo.

---

## 8. Qué afirmar y qué no

### Se puede afirmar

- `V_i = ∫R_i^⊥dt` — identidad exacta, demostrada y verificada numéricamente
- Fórmulas cerradas de la estrella — exactas
- Anti-centralidad en estrella y ER — empíricamente fuerte, con base teórica
- Dependencia de banda — la detección del hub la controla un solo modo
- Colocación de sensores — V supera a las centralidades clásicas en test
  sin fuga (Pearson +0.931 contra +señal ODE, mientras random walk da −0.680)

### No se puede afirmar

- Teorema general de anti-centralidad para toda familia de grafos —
  falta demostración
- Aplicaciones a sistemas biológicos sin validación específica del dominio
- Teoría universal de detección — los resultados aplican a difusión
  laplaciana, no a cualquier dinámica
- Que V sea novedoso como objeto: `V_i = L⁺_ii` es la diagonal de la
  pseudoinversa del Laplaciano, conocida desde Klein–Randić (1993) como
  resistencia efectiva y estudiada por Van Mieghem (2017). **La contribución
  es la identidad con el heat kernel, el análisis de timescales y la
  validación empírica sistemática, no el objeto en sí.** Cítalos.

---

## 9. Plantilla mínima para una app nueva

```python
"""
1. Cargar la red
   - CSV de aristas: pd.read_csv(f, comment='#', header=None).iloc[:, :2]
   - nx.Graph(), remove_edges_from(nx.selfloop_edges(G))
   - componente conexa mayor
2. A = nx.to_numpy_array(G)
3. d = dsp(A)
4. Controles obligatorios:
   - d['rango'] < 1000                      → si no, avisar
   - error de identidad < 1e-13             → si no, bug
   - reportar Pearson Y Spearman de (k, V)
   - reportar R²(V, 1/k)                    → §7.6
5. Salidas:
   - ranking de nodos por V
   - curvas R_i^⊥(t) del nodo más y menos persistente
   - scatter grado vs V con ambas correlaciones
   - ⟨τ̃⟩, Var(τ̃), λ₂, V_max/V_min
"""
```

### Preguntas que el marco responde bien

- ¿Dónde coloco sensores para observar la dinámica lenta del sistema?
  → nodos de V alta, no los de grado alto
- ¿Qué nodos almacenan versus cuáles enrutan?
  → V alta almacena, grado alto enruta
- ¿Cómo cambia la organización de la red entre dos condiciones?
  → Δ⟨τ̃⟩, Δλ₂, ΔVar(τ̃)

### Preguntas que no responde

- Qué está procesando el sistema (solo cómo)
- Contenido causal o dirección de flujo
- Predicción a nivel de individuo — el marco es robusto a nivel de grupo

---

## 10. Referencias que debes citar

- Klein, D. J., & Randić, M. (1993). Resistance distance.
  *Journal of Mathematical Chemistry*, 12, 81–95.
- Van Mieghem, P., Devriendt, K., & Cetinay, H. (2017). Pseudoinverse of the
  Laplacian and best spreader node in a network. *Physical Review E*, 96, 032311.
- Brandes, U., & Fleischer, D. (2005). Centrality measures based on current flow.
  *STACS 2005*.
- Chung, F. R. K. (1997). *Spectral Graph Theory*. AMS.
- Arteaga Marroquin, E. A. (2025). Dynamic Spectral Persistence in Heterogeneous
  Networks: From Hub Blindness to Heat Kernel Identity. Preprint.
