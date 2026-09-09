import Mathlib

open scoped BigOperators

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

#check ProbabilityTheory.variance_eq_sub
#check ProbabilityTheory.variance_const_mul
#check ProbabilityTheory.IndepFun.variance_sum

example : (7 : ℝ) - 3 = 4 := by norm_num

#eval 7 - 3
