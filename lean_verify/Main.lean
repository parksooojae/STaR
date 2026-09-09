import Mathlib

open MeasureTheory ProbabilityTheory
open scoped BigOperators ProbabilityTheory

/-- Algebraic core of the whiteboard derivation.
If E[X^2] = σ² + μ² and E[X̄^2] = σ²/N + μ²,
then E[X^2] - E[X̄^2] = ((N-1)/N) σ². -/
theorem whiteboard_core
    (N : ℕ) (hN : 0 < N)
    (μ σ2 EX2 Ebar2 : ℝ)
    (hX2 : EX2 = σ2 + μ ^ 2)
    (hbar2 : Ebar2 = σ2 / (N : ℝ) + μ ^ 2) :
    EX2 - Ebar2 = (((N : ℝ) - 1) / (N : ℝ)) * σ2 := by
  have hN0 : (N : ℝ) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt hN)
  rw [hX2, hbar2]
  field_simp [hN0]
  ring

/-- Pointwise identity behind the sample-variance calculation. -/
theorem mean_sq_deviation_identity
    {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) :
    (∑ i, (x i - (∑ j, x j) / (Fintype.card ι : ℝ)) ^ 2) /
        (Fintype.card ι : ℝ) =
      (∑ i, (x i) ^ 2) / (Fintype.card ι : ℝ) -
        ((∑ j, x j) / (Fintype.card ι : ℝ)) ^ 2 := by
  classical
  have hn : (Fintype.card ι : ℝ) ≠ 0 := by
    exact_mod_cast (Fintype.card_ne_zero : Fintype.card ι ≠ 0)
  let a : ℝ := (∑ j, x j) / (Fintype.card ι : ℝ)
  have hsum :
      (∑ i, (x i - a) ^ 2) =
        (∑ i, (x i) ^ 2) - (2 * a) * (∑ i, x i) +
          (Fintype.card ι : ℝ) * a ^ 2 := by
    calc
      _ = ∑ i, ((x i) ^ 2 - (2 * a) * x i + a ^ 2) := by
        apply Finset.sum_congr rfl
        intro i hi
        ring
      _ = (∑ i, (x i) ^ 2) - (∑ i, (2 * a) * x i) + (∑ _i : ι, a ^ 2) := by
        rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
      _ = (∑ i, (x i) ^ 2) - (2 * a) * (∑ i, x i) +
          (Fintype.card ι : ℝ) * a ^ 2 := by
        rw [← Finset.mul_sum]
        simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  change (∑ i, (x i - a) ^ 2) / (Fintype.card ι : ℝ) =
      (∑ i, (x i) ^ 2) / (Fintype.card ι : ℝ) - a ^ 2
  rw [hsum]
  dsimp [a]
  field_simp [hn]
  ring

/-- For a finite nonempty i.i.d. family of square-integrable real random variables,
the variance of the sample mean is the common variance divided by the sample size. -/
theorem variance_iid_sample_mean
    {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι] [Nonempty ι]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : ι → Ω → ℝ) (k : ι)
    (hXk : MemLp (X k) 2 μ)
    (hident : ∀ i, IdentDistrib (X i) (X k) μ μ)
    (hindep : Pairwise (fun i j ↦ IndepFun (X i) (X j) μ)) :
    Var[fun ω ↦ (∑ i, X i ω) / (Fintype.card ι : ℝ); μ] =
      Var[X k; μ] / (Fintype.card ι : ℝ) := by
  classical
  have hcard : (Fintype.card ι : ℝ) ≠ 0 := by
    exact_mod_cast (Fintype.card_ne_zero : Fintype.card ι ≠ 0)
  have hXi : ∀ i, MemLp (X i) 2 μ := fun i ↦ (hident i).memLp_iff.2 hXk
  have hsum :
      Var[(∑ i, X i); μ] = ∑ i, Var[X i; μ] := by
    simpa using
      (IndepFun.variance_sum
        (s := (Finset.univ : Finset ι)) (X := X)
        (fun i _ ↦ hXi i)
        (fun i _ j _ hij ↦ hindep hij))
  have hvarsum :
      (∑ i, Var[X i; μ]) = (Fintype.card ι : ℝ) * Var[X k; μ] := by
    simp_rw [(hident _).variance_eq]
    simp
  have hscale :
      Var[fun ω ↦ (∑ i, X i ω) / (Fintype.card ι : ℝ); μ] =
        (1 / (Fintype.card ι : ℝ)) ^ 2 * Var[(∑ i, X i); μ] := by
    rw [show (fun ω ↦ (∑ i, X i ω) / (Fintype.card ι : ℝ)) =
        (fun ω ↦ (1 / (Fintype.card ι : ℝ)) * (∑ i, X i) ω) by
      funext ω
      simp [div_eq_mul_inv, mul_comm]]
    rw [variance_const_mul]
  rw [hscale, hsum, hvarsum]
  field_simp [hcard]

#check ProbabilityTheory.variance_eq_sub
#check ProbabilityTheory.variance_const_mul
#check ProbabilityTheory.IndepFun.variance_sum

example : (7 : ℝ) - 3 = 4 := by norm_num
#eval 7 - 3
