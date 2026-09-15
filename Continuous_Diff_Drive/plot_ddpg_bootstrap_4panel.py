# ============================================================
# DDPG: 4-panel training plot (Return, Length, Critic loss, Actor loss)
# with moving-block bootstrap CI, read ONLY from metrics.csv.
# No training / no environment / no agent needed.
# ============================================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Load the metrics CSV
# ------------------------------------------------------------

PROJECT_ROOT = Path.cwd()

csv_path = PROJECT_ROOT / "artifacts" / "continuous" / "ddpg" / \
    "20260915T150100546310Z_seed0_8d99d48d" / "metrics.csv"

if not csv_path.exists():
    csv_path = PROJECT_ROOT.parent / "artifacts" / "continuous" / "ddpg" / \
        "20260915T150100546310Z_seed0_8d99d48d" / "metrics.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"File CSV non trovato in: {csv_path}")

df = pd.read_csv(csv_path)

print("File caricato con successo:")
print(csv_path)
print("\nColonne trovate nel file CSV:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. Automatically identify the relevant columns
# ------------------------------------------------------------

def find_column(columns, keywords):
    for c in columns:
        cl = c.lower()
        if any(k in cl for k in keywords):
            return c
    return None


episode_col = find_column(df.columns, ["episode"])
return_col = find_column(df.columns, ["episode_return", "return", "reward"])
length_col = find_column(df.columns, ["episode_length", "length", "steps"])
critic_loss_col = find_column(df.columns, ["critic_loss", "loss_critic"])
actor_loss_col = find_column(df.columns, ["actor_loss", "loss_actor"])

print("\nColonne identificate:")
print("episode     :", episode_col)
print("return      :", return_col)
print("length      :", length_col)
print("critic_loss :", critic_loss_col)
print("actor_loss  :", actor_loss_col)

if episode_col is None or return_col is None:
    raise RuntimeError(
        "Impossibile identificare le colonne episode e return nel CSV. "
        "Verifica i nomi delle colonne stampati sopra."
    )

episodes = df[episode_col].to_numpy()


# ------------------------------------------------------------
# 3. Moving-block bootstrap function (identica a quella del notebook)
# ------------------------------------------------------------

def moving_block_bootstrap_ci(
    values,
    window=100,
    block_size=10,
    n_bootstrap=1000,
    confidence=0.95,
    seed=42
):
    values = np.asarray(values, dtype=float)
    n = len(values)

    means = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    n_blocks = int(np.ceil(window / block_size))

    for i in range(window - 1, n):
        x = values[i - window + 1:i + 1]
        x = x[np.isfinite(x)]

        if len(x) < window * 0.8:
            continue

        starts = np.arange(0, len(x) - block_size + 1)

        sampled_starts = rng.choice(starts, size=(n_bootstrap, n_blocks))
        blocks = np.array([x[s:s + block_size] for s in sampled_starts.flat])
        blocks = blocks.reshape(n_bootstrap, n_blocks * block_size)[:, :window]

        bootstrap_means = np.mean(blocks, axis=1)

        means[i] = np.mean(x)
        lower[i] = np.quantile(bootstrap_means, alpha / 2)
        upper[i] = np.quantile(bootstrap_means, 1 - alpha / 2)

    return means, lower, upper


# ------------------------------------------------------------
# 4. Parameters (stessi usati per la curva di return bootstrap)
# ------------------------------------------------------------

WINDOW = 50
BLOCK_SIZE = 10
N_BOOTSTRAP = 200


def plot_panel(ax, values, title, ylabel, raw_alpha=0.15):
    rolling, low, high = moving_block_bootstrap_ci(
        values, window=WINDOW, block_size=BLOCK_SIZE,
        n_bootstrap=N_BOOTSTRAP, seed=42
    )
    ax.plot(episodes, values, alpha=raw_alpha, color="tab:blue", label="Episode value")
    ax.plot(episodes, rolling, linewidth=2, color="tab:orange",
             label=f"Rolling mean (window={WINDOW})")
    ax.fill_between(episodes, low, high, alpha=0.35, color="tab:blue",
                     label="95% bootstrap CI")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.2)


# ------------------------------------------------------------
# 5. Build the 4-panel figure
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("DDPG (bootstrap CI)", fontsize=16)

plot_panel(axes[0, 0], df[return_col].to_numpy(dtype=float),
           "Episode return", "Return")

if length_col is not None:
    plot_panel(axes[0, 1], df[length_col].to_numpy(dtype=float),
               "Episode length", "Steps")
else:
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Episode length: colonna non trovata nel CSV")

if critic_loss_col is not None:
    plot_panel(axes[1, 0], df[critic_loss_col].to_numpy(dtype=float),
               "Critic loss", "Loss")
else:
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Critic loss: colonna non trovata nel CSV")

if actor_loss_col is not None:
    plot_panel(axes[1, 1], df[actor_loss_col].to_numpy(dtype=float),
               "Actor loss", "Loss")
else:
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Actor loss: colonna non trovata nel CSV")

plt.tight_layout()
plt.savefig("ddpg_training_curves_bootstrap.png", dpi=150)
plt.show()
plt.close()

print("\nSalvato: ddpg_training_curves_bootstrap.png")
