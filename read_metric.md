# Reading the training metrics

What every column in `runs/<env>/<run>/metrics.csv` means, and what a bad value
looks like. Written against the LunarLander runs; the same reading applies to
the G1 walking task at Stage 5.

## The one rule

**`policy_loss` and `value_loss` are not progress metrics.**

Policy loss hovers near zero by construction and can rise while the agent
improves. Value loss falls because returns shrink as episodes shorten, not
because the critic got better — `explained_variance` is the metric for that.
If you only watch the losses you are blind.

The metric that answers "is this working" is `charts/episodic_return`.

## The table

| metric | what it means | healthy | trouble |
|---|---|---|---|
| `charts/episodic_return` | **the only success metric** | rising, then plateaus high | flat from the start; or rises then collapses |
| `charts/episodic_length` | *how* it's succeeding | env-specific — read with return | see "the two-metric reading" |
| `charts/episodic_return_std` | spread over the 100-episode window | shrinking as the policy sharpens | stays as large as the mean = inconsistent policy |
| `charts/episodes_total` | episodes finished so far | — | not rising = episodes never end |
| `charts/SPS` | samples per second | — | dropping over time = a leak or video capture left on |
| `losses/explained_variance` | is the critic any good | climbing toward 0.8–0.95 | stuck near 0 = critic useless; negative = worse than a constant |
| `losses/approx_kl` | how far the policy moved per update | 0.003–0.02 | > 0.05 = updates too large, lower `lr`; ≈ 0 = not learning |
| `losses/clip_fraction` | fraction of samples hitting the clip | 0.05–0.2 | > 0.3 = lower `lr`; ≈ 0 = updates too timid |
| `losses/entropy` | how random the policy still is | declining slowly | crashes to ≈ 0 early = premature determinism, raise `ent_coeff` |
| `losses/value_loss` | critic fit error | falls fast, then wanders | read as a ratio to the return scale, never absolutely |
| `losses/grad_norm` | pre-clip gradient size | anything; watch for spikes | sustained ≫ `max_grad_norm` = value loss dominating the shared gradient |
| `losses/early_stopped` | did `target_kl` fire this iteration | rare | frequent = `lr` too high |
| `losses/epochs_ran` | epochs completed before the KL stop | `update_epochs` | consistently below it = same as above |

Entropy has a ceiling worth knowing: for a uniform policy over `n` discrete
actions it is `ln n` (1.386 for LunarLander's 4). A freshly initialised policy
should start there — if it doesn't, the output layer init is wrong.

## The two-metric reading

Return alone is ambiguous. Return *plus* episode length names the failure mode.

| length | return | reading |
|---|---|---|
| ↑ | ↑ | learning to survive. Normal early phase. |
| ↓ | ↑ | learning to *finish*. This is what solving looks like. |
| high, flat | mid, flat | **hovering** — banking the shaping reward, never committing to the goal |
| pinned at the time limit | low | the agent found a way to stall. Check the reward for a term it can farm. |
| ↓ | ↓ | dying faster. Something is actively wrong. |

For the humanoid at Stage 3 the same logic reads as: length ↑ with return ↑ is
learning not to fall; length flat with return ↑ is learning to actually walk.

## Diagnosing a plateau

When return stops improving, check these in order — each one you rule out
narrows the problem:

1. **`explained_variance` low?** The critic is the problem. Advantages are noise.
   Lower `gae_lambda` (1.0 → 0.95 helps a lot), or give the critic more capacity.
2. **`approx_kl` high / `clip_fraction` > 0.3?** Updates are too aggressive.
   Lower `lr`, or lower `update_epochs`.
3. **`entropy` collapsed?** The policy went deterministic before it found the
   good behaviour. Raise `ent_coeff` (0.0 → 0.01).
4. **All three healthy and still stuck?** The machinery is fine — it's the
   objective. Either the reward has a local optimum (hovering, stalling), or
   `update_epochs` is high enough that each batch is being overfitted.

## Iteration count, when changing `num_envs`

```
num_iterations = total_steps // (num_envs * num_steps)
```

Raising `num_envs` at fixed `total_steps` **divides your number of policy
updates**. 1 env × 2048 steps over 400k = 195 updates; 8 envs × 2048 over the
same 400k = 24. Same samples, one-eighth the optimisation, much worse result.

To use more envs as a speedup, hold the batch size fixed:
`--num-envs 8 --num-steps 256`. To use them for lower-variance updates, raise
`total_steps` to match.
