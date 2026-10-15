Density matrix embedding theory (DMET) with **COMPASS** (*COMmutativity
Pre-screened Automated Selection of Scatterers*), a dynamically built,
compact UCC-type ansatz used as the quantum high-level solver of every
embedded subsystem.

This repository implements the algorithm of

> D. Mondal, A. K. Patra, R. Maitra, *Advancing practical quantum embedding
> simulations via operator commutativity-based state preparation for complex
> chemical systems*, **Chem. Sci.** (2026). DOI: [10.1039/d6sc03132c](https://doi.org/10.1039/d6sc03132c)

and ships the paper's flagship example, cyclo[10]carbon (C10, 100 qubits in
STO-3G), which DMET reduces to 12-qubit embedded problems.

---

## 1. The algorithm

### 1.1 DMET: the embedding

| Step | What happens | Code |
|---|---|---|
| (a) | RHF of the full molecule. | `Molecule` |
| (b) | Partition the molecule into atom-centred fragments `A_y` and build meta-Löwdin localized orbitals. | `DMETCompass.__init__` |
| (c) | Diagonalize the environment block of the mean-field 1-RDM. Eigenvalues strictly between 0 and 2 (threshold 1e-13) define the entangled **bath** orbitals `B_y`. Fragment + bath = embedded subsystem *x*. | `DMETCompass.build_fragment` |
| (d) | Project the full Hamiltonian onto *x* and add the global chemical potential on the fragment orbitals, **H**ˣ_emb(μ) = **H**ˣ_emb − μ Σ_{p∈A} a†_p a_p (eqn 5). An embedded RHF gives the reference \|φˣ_ref⟩. A (6e, 6o) active space around HOMO/LUMO is kept, i.e. 12 qubits. | `build_fragment`, `select_frozen_orbitals` |
| (d) | Solve **H**ˣ_emb(μ) with COMPASS-VQE (see 1.2). | `CompassVQE.solve` |
| (e) | Measure the 1- and 2-RDMs of the optimized state. Compute the fragment energy Eˣ (eqn 6) and the fragment electron number Nˣ = Σ_{p∈A} D_pp. | `measure_rdms`, `fragment_energy` |
| (f) | Total energy E = Σ_x Eˣ + E_nuc (eqn 7). Update μ with the Newton–secant method until N_act − Σ_x Nˣ(μ) ≈ 0. Every μ update rebuilds the embedded Hamiltonians and **re-constructs the ansatz**. | `DMETCompass.macro_cycle`, `DMETCompass.run` |

### 1.2 COMPASS: dynamic ansatz construction for one embedded subsystem

Plain UCCSD misses the higher-order (triples and up) correlation that matters
in strongly correlated regions, and explicit triples are far too expensive.
COMPASS produces their leading effect from **two-body operators only**.

A double `σ_I = T_I − T_I†` is paired with a non-commuting two-body
**scatterer** `ς_n`, and the disentangled product `e^{ς_n} e^{σ_I}` contains
`[ς_n, σ_I] ∝ σ³` (eqn 10).

Scatterers are generalized two-body operators with an effective excitation
rank of one (eqn 9):

* **S_h** (hole scatterer): one occupied→virtual excitation plus one
  occupied→occupied scattering into a *contractible set of orbitals* (CSO).
* **S_p** (particle scatterer): one occupied→virtual excitation plus one
  virtual→virtual scattering out of a CSO.


The ansatz is built in four steps:

1. **One-parameter micro-cycles (eqn 11).** Every double `T_I` is optimized
   alone on \|φ_ref⟩. Doubles with `|E_ref − E_I| > 1e-6` Ha are kept and
   placed in operator blocks in **descending order of energy gain**: the most
   important double acts on the reference first.
2. **Commutativity pre-screening.** For each kept double, list the scatterers
   that share a CSO with it (and so do not commute with it) and share no other
   index.
3. **Two-parameter micro-cycles (eqn 13).** Optimize `e^{ς_n} e^{σ_I}`,
   starting from the step-1 optimum of `σ_I`. Scatterers with
   `|E_I − E_In| > 1e-7` Ha join the block of `σ_I`. The block order is not
   changed.
4. **Global VQE (eqns 14, 15).** All blocks plus every single excitation are
   optimized together with SLSQP, starting from the micro-cycle parameters
   (singles start at 0):

## 7. Citation

```bibtex
@article{mondal2026dmetcompass,
  title   = {Advancing practical quantum embedding simulations via operator
             commutativity-based state preparation for complex chemical systems},
  author  = {Mondal, Dibyendu and Patra, Ashish Kumar and Maitra, Rahul},
  journal = {Chemical Science},
  year    = {2026},
  doi     = {10.1039/d6sc03132c}
}
```
