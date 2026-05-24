from Grid_cliff.grid_cliff_Qlearning import (
    evaluate,
    evaluate_recorded,
    plot_curves,
    register_cliff_env,
    train_q_learning_recorded,
)

# ── Environment config ────────────────────────────────────────────────────────

register_cliff_env()

# ── Hyperparameters ───────────────────────────────────────────────────────────

NUM_EPISODES = 10_000
RECORD_EVERY = 500
LEARNING_RATE = 0.1
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.0
DISCOUNT_FACTOR = 0.99
EPS_DECAY_STEPS = 8_000
TRAIN_SEED = 0
EVAL_EPISODES = 1
EVAL_SEED = 1234

# ── Training ──────────────────────────────────────────────────────────────────

Q, rets, lens, succ, eps = train_q_learning_recorded(
    episodes=NUM_EPISODES,
    alpha=LEARNING_RATE,
    gamma=DISCOUNT_FACTOR,
    eps_start=INITIAL_EPSILON,
    eps_end=FINAL_EPSILON,
    eps_decay_steps=EPS_DECAY_STEPS,
    seed=TRAIN_SEED,
    log_every=500,
    video_folder="videos/cliff_training",
    record_every=RECORD_EVERY,
)

plot_curves(rets, lens, succ, eps, smooth_window=100)

# ── Evaluation ────────────────────────────────────────────────────────────────

eval_rets, eval_lens, eval_succ = evaluate(Q, episodes=200, seed=EVAL_SEED)
print(
    "\nEvaluation (greedy, no render): "
    f"mean_return={eval_rets.mean():.2f}, "
    f"mean_len={eval_lens.mean():.2f}, "
    f"success_rate={eval_succ.mean()*100:.1f}%"
)

video_rets, video_lens, video_succ = evaluate_recorded(
    Q,
    episodes=EVAL_EPISODES,
    seed=EVAL_SEED,
    video_folder="videos/cliff_evaluation",
    name_prefix="eval",
)
print(
    "Recorded evaluation: "
    f"return={float(video_rets.mean()):.2f}, "
    f"len={float(video_lens.mean()):.2f}, "
    f"success_rate={float(video_succ.mean())*100:.1f}%"
)
