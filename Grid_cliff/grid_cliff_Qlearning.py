# grid_cliff_Qlearning.py
import numpy as np
import matplotlib.pyplot as plt
import time
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from Grid_cliff.grid_cliff_env import ENV_ID, MAP_SEED, MAX_STEPS, N_CLIFFS, make_cliff_env


def epsilon_greedy(Q, s, eps, n_actions, rng):
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    return int(np.argmax(Q[s]))


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1:
        return x.copy()
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid").astype(np.float32)


def train_q_learning(
    episodes=10000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.00,
    eps_decay_steps=8000,
    seed=0,
    log_every=500,
    env=None,
):
    if env is None:
        env = make_cliff_env(render_mode=None)
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    goal_state = env.unwrapped.goal[0] * env.unwrapped.cols + env.unwrapped.goal[1]

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    returns = np.zeros(episodes, dtype=np.float32)
    lengths = np.zeros(episodes, dtype=np.int32)
    success = np.zeros(episodes, dtype=np.int32)
    epsilons = np.zeros(episodes, dtype=np.float32)

    def eps_schedule(ep: int) -> float:
        t = min(ep, eps_decay_steps)
        return float(eps_start + (eps_end - eps_start) * (t / float(eps_decay_steps)))

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0

        eps = eps_schedule(ep)
        epsilons[ep] = eps

        terminated = False
        truncated = False

        while not done:
            a = epsilon_greedy(Q, s, eps, n_actions, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            td_target = r + (0.0 if done else gamma * float(np.max(Q[s2])))
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s2
            ep_return += float(r)
            ep_len += 1

        returns[ep] = ep_return
        lengths[ep] = ep_len
        success[ep] = 1 if (terminated and (s == goal_state)) else 0

        if log_every and (ep + 1) % log_every == 0:
            rr = returns[ep + 1 - log_every : ep + 1]
            ss = success[ep + 1 - log_every : ep + 1]
            ll = lengths[ep + 1 - log_every : ep + 1]
            print(
                f"Episode {ep+1:5d}/{episodes} | eps={eps:.3f} | "
                f"avg_return={rr.mean():7.2f} | avg_len={ll.mean():6.2f} | "
                f"success_rate={ss.mean()*100:5.1f}% | "
                f"best={rr.max():6.0f} worst={rr.min():6.0f}"
            )

    env.close()
    return Q, returns, lengths, success, epsilons


def evaluate(Q, episodes=200, seed=999):
    env = make_cliff_env(render_mode=None)
    env.action_space.seed(seed)

    n_states = env.observation_space.n
    goal_state = env.unwrapped.goal[0] * env.unwrapped.cols + env.unwrapped.goal[1]

    returns = np.zeros(episodes, dtype=np.float32)
    lengths = np.zeros(episodes, dtype=np.int32)
    success = np.zeros(episodes, dtype=np.int32)

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        terminated = False
        truncated = False

        while not done:
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_return += float(r)
            ep_len += 1

        returns[ep] = ep_return
        lengths[ep] = ep_len
        success[ep] = 1 if (terminated and (s == goal_state)) else 0

    env.close()
    return returns, lengths, success


def register_cliff_env():
    try:
        gym.spec(ENV_ID)
    except gym.error.Error:
        gym.register(
            id=ENV_ID,
            entry_point="Grid_cliff.grid_cliff_env:RandomCliffWalkingEnv",
            kwargs={
                "map_seed": MAP_SEED,
                "n_cliffs": N_CLIFFS,
                "max_steps": MAX_STEPS,
            },
        )


def train_q_learning_recorded(
    episodes=10000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.0,
    eps_decay_steps=8000,
    seed=0,
    log_every=500,
    video_folder="videos/cliff_training",
    record_every=500,
):
    register_cliff_env()
    base_env = gym.make(ENV_ID, render_mode="rgb_array")
    train_env = RecordVideo(
        base_env,
        video_folder=video_folder,
        episode_trigger=lambda ep: (ep + 1) % record_every == 0,
        disable_logger=True,
    )
    train_env = RecordEpisodeStatistics(train_env)

    try:
        return train_q_learning(
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            seed=seed,
            log_every=log_every,
            env=train_env,
        )
    finally:
        train_env.close()


def evaluate_recorded(Q, episodes=1, seed=1234, video_folder="videos/cliff_evaluation", name_prefix="eval"):
    register_cliff_env()
    eval_env = gym.make(ENV_ID, render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,
        name_prefix=name_prefix,
        disable_logger=True,
    )
    eval_env = RecordEpisodeStatistics(eval_env)

    goal_state = eval_env.unwrapped.goal[0] * eval_env.unwrapped.cols + eval_env.unwrapped.goal[1]
    returns = np.zeros(episodes, dtype=np.float32)
    lengths = np.zeros(episodes, dtype=np.int32)
    success = np.zeros(episodes, dtype=np.int32)

    try:
        eval_env.action_space.seed(seed)
        for ep in range(episodes):
            s, _ = eval_env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            ep_len = 0
            terminated = False
            truncated = False

            while not done:
                a = int(np.argmax(Q[s]))
                s, r, terminated, truncated, _ = eval_env.step(a)
                done = terminated or truncated
                ep_return += float(r)
                ep_len += 1

            returns[ep] = ep_return
            lengths[ep] = ep_len
            success[ep] = 1 if (terminated and (s == goal_state)) else 0
    finally:
        eval_env.close()

    return returns, lengths, success


def render_final_episode(Q, seed=2026, step_delay=0.12, max_steps=1000):
    """
    Shows ONE episode in a human-render window.
    Keeps the same action selection you use in evaluation: greedy argmax(Q[s]).
    """
    env = make_cliff_env(render_mode="human")
    s, _ = env.reset(seed=seed)

    done = False
    steps = 0
    ep_return = 0.0

    while not done and steps < max_steps:
        a = int(np.argmax(Q[s]))
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_return += float(r)
        steps += 1
        time.sleep(step_delay)

    # keep the final frame visible briefly
    time.sleep(1.0)
    env.close()
    print(f"[Final episode] return={ep_return:.0f}, steps={steps}, done={done}")


def plot_curves(returns, lengths, success, epsilons, smooth_window=200):
    episodes = np.arange(len(returns))

    # Smoothed curves
    ret_sm = rolling_mean(returns, smooth_window)
    len_sm = rolling_mean(lengths.astype(np.float32), smooth_window)
    suc_sm = rolling_mean(success.astype(np.float32), smooth_window)
    eps_sm = epsilons  # epsilon is already smooth by design

    # X axes aligned with "valid" conv length
    x_sm = np.arange(len(ret_sm)) + (smooth_window - 1)

    plt.figure()
    plt.plot(x_sm, ret_sm, linewidth=2)
    plt.title(f"Training: Return (rolling mean, w={smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)

    plt.figure()
    plt.plot(x_sm, len_sm, linewidth=2)
    plt.title(f"Training: Episode Length (rolling mean, w={smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)

    plt.figure()
    plt.plot(x_sm, suc_sm, linewidth=2)
    plt.title(f"Training: Success Rate (rolling mean, w={smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("P(success)")
    plt.ylim(0.0, 1.0)
    plt.grid(True)

    plt.figure()
    plt.plot(episodes, eps_sm, linewidth=2)
    plt.title("Epsilon Schedule")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    register_cliff_env()
    Q, rets, lens, succ, eps = train_q_learning(
        episodes=10000,
        alpha=0.1,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay_steps=8000,
        seed=0,
        log_every=500,
    )

    plot_curves(rets, lens, succ, eps, smooth_window=100)

    eval_rets, eval_lens, eval_succ = evaluate(Q, episodes=200, seed=1234)
    print(
        "\nEvaluation (greedy, no render): "
        f"mean_return={eval_rets.mean():.2f}, "
        f"mean_len={eval_lens.mean():.2f}, "
        f"success_rate={eval_succ.mean()*100:.1f}%"
    )

    #Show ONE final episode with human rendering
    render_final_episode(Q, seed=2026, step_delay=0.12)
