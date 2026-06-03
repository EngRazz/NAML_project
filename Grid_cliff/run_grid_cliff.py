from Grid_cliff.grid_cliff_algorithms import (
    evaluate,
    evaluate_recorded,
    plot_comparison,
    register_cliff_env,
    train_q_learning_recorded,
    train_sarsa_recorded,
)

# ── Environment config ────────────────────────────────────────────────────────

register_cliff_env()

# ── Hyperparameters ───────────────────────────────────────────────────────────

NUM_EPISODES = 5000
RECORD_EVERY = 500
LEARNING_RATE = 0.1
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1   # keep a floor > 0: persistent exploration is what makes Q-learning hug the cliff (risky) and SARSA back off (safe)
DISCOUNT_FACTOR = 0.99
EPS_DECAY_STEPS = 3000
TRAIN_SEED = 0
EVAL_EPISODES = 1
EVAL_SEED = 1234

# ══════════════════════════════════════════════════════════════════════════════
#  Q-LEARNING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Q-LEARNING  ")
print("=" * 70)

Q, rets, lens, succ, eps = train_q_learning_recorded(
    episodes=NUM_EPISODES,
    alpha=LEARNING_RATE,
    gamma=DISCOUNT_FACTOR,
    eps_start=INITIAL_EPSILON,
    eps_end=FINAL_EPSILON,
    eps_decay_steps=EPS_DECAY_STEPS,
    seed=TRAIN_SEED,
    log_every=500,
    video_folder="videos/cliff_training/qlearning",
    record_every=RECORD_EVERY,
    name_prefix="qlearning_video",
)

eval_rets, eval_lens, eval_succ = evaluate(Q, episodes=200, seed=EVAL_SEED)
print(
    "\n[Q-learning] Evaluation (greedy, no render): "
    f"mean_return={eval_rets.mean():.2f}, "
    f"mean_len={eval_lens.mean():.2f}, "
    f"success_rate={eval_succ.mean()*100:.1f}%"
)

video_rets, video_lens, video_succ = evaluate_recorded(
    Q,
    episodes=EVAL_EPISODES,
    seed=EVAL_SEED,
    video_folder="videos/cliff_evaluation/qlearning",
    name_prefix="qlearning_video",
)
print(
    "[Q-learning] Recorded evaluation: "
    f"return={float(video_rets.mean()):.2f}, "
    f"len={float(video_lens.mean()):.2f}, "
    f"success_rate={float(video_succ.mean())*100:.1f}%"
)

# ══════════════════════════════════════════════════════════════════════════════
#  SARSA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SARSA  ")
print("=" * 70)

Q_sarsa, rets_s, lens_s, succ_s, eps_s = train_sarsa_recorded(
    episodes=NUM_EPISODES,
    alpha=LEARNING_RATE,
    gamma=DISCOUNT_FACTOR,
    eps_start=INITIAL_EPSILON,
    eps_end=FINAL_EPSILON,
    eps_decay_steps=EPS_DECAY_STEPS,
    seed=TRAIN_SEED,
    log_every=500,
    video_folder="videos/cliff_training/sarsa",
    record_every=RECORD_EVERY,
    name_prefix="sarsa_video",
)

eval_rets_s, eval_lens_s, eval_succ_s = evaluate(Q_sarsa, episodes=200, seed=EVAL_SEED)
print(
    "\n[SARSA] Evaluation (greedy, no render): "
    f"mean_return={eval_rets_s.mean():.2f}, "
    f"mean_len={eval_lens_s.mean():.2f}, "
    f"success_rate={eval_succ_s.mean()*100:.1f}%"
)

video_rets_s, video_lens_s, video_succ_s = evaluate_recorded(
    Q_sarsa,
    episodes=EVAL_EPISODES,
    seed=EVAL_SEED,
    video_folder="videos/cliff_evaluation/sarsa",
    name_prefix="sarsa_video",
)
print(
    "[SARSA] Recorded evaluation: "
    f"return={float(video_rets_s.mean()):.2f}, "
    f"len={float(video_lens_s.mean()):.2f}, "
    f"success_rate={float(video_succ_s.mean())*100:.1f}%"
)

# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

plot_comparison(
    q_stats=(rets, lens, succ, eps),
    sarsa_stats=(rets_s, lens_s, succ_s, eps_s),
    smooth_window=100,
    save_path="images/cliff_qlearning_vs_sarsa.png",
)
