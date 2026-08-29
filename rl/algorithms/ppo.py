import numpy as np
import torch
import torch.nn as nn

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    '''Measures how much your value network (critic) explains true returns.
    - 1.0 means high precision
    - 0.0 or negative means critic is bad.
    '''
    var_y = torch.var(y_true)
    return float(1 - torch.var(y_true - y_pred) / var_y) if var_y > 0 else float('nan')

def ppo_update(agent, optimizer, batch: dict, cfg) -> dict:
    '''cfg is PPOConfig, bath is buf.flatten()'''
    obs, actions = batch["obs"], batch["actions"]
    old_logprobs, old_values = batch["logprobs"], batch["values"]
    advantages, returns = batch["advantages"], batch["returns"]

    n = obs.shape[0]
    mb_size = max(1, n // cfg.num_minibatches)
    inds = np.arange(n)

    clipfracs, kls = [], []
    pg_loss = v_loss = entropy = grad_norm = torch.tensor(0.0)

    stop = False

    for epoch in range(1, cfg.update_epochs + 1):
        np.random.shuffle(inds)
        for start in range(0, n, mb_size):
            mb = inds[start:start + mb_size]
            mb_obs, mb_actions = obs[mb], actions[mb]
            mb_old_logprobs = old_logprobs[mb]
            mb_advantages, mb_returns = advantages[mb], returns[mb]

            if cfg.norm_adv and mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            mb_logprob, mb_entropy, mb_value = agent.evaluate_actions(mb_obs, mb_actions)
            logratio = mb_logprob - mb_old_logprobs
            ratio = torch.exp(logratio)
            surr1 = -ratio * mb_advantages
            surr2 = -torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_advantages

            pg_loss = torch.max(surr1, surr2).mean()  # actor loss
            v_loss = 0.5 * ((mb_value - mb_returns) ** 2).mean()  # critic loss
            entropy = mb_entropy.mean()  # entropy over action distribution

            loss = pg_loss + cfg.vf_coeff * v_loss - cfg.ent_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # For logging purposes
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                kls.append(approx_kl.item())
                clipfracs.append(((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item())

        if cfg.target_kl is not None and kls[-1] > cfg.target_kl:
            stop = True
            break

    return {
        "losses/policy_loss": pg_loss.item(),
        "losses/value_loss": v_loss.item(),
        "losses/entropy": entropy.item(),
        "losses/approx_kl": float(np.mean(kls)),
        "losses/clip_fraction": float(np.mean(clipfracs)),
        "losses/grad_norm": float(grad_norm),
        "losses/explained_variance": explained_variance(old_values, returns),
        "losses/epochs_ran": epoch,
        "losses/early_stopped": float(stop),
    }
