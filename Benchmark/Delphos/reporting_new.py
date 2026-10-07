"""Optional plots; reporting never estimates models."""
from pathlib import Path


def plot_training(agent):
    import matplotlib.pyplot as plt
    import pandas as pd
    root = Path(agent.subfolder)
    training = pd.DataFrame(agent.training_log)
    if training.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(training['episode'], training['reward'].rolling(200, min_periods=1).mean())
    ax.set(xlabel='Episode', ylabel='Mean reward')
    fig.tight_layout()
    fig.savefig(root / 'learning_curve_new.png')
    plt.close(fig)
    history = pd.DataFrame(agent.current_best_history)
    if history.empty:
        return
    for metric, rows in history.groupby('metric'):
        fig, ax = plt.subplots()
        x = rows['estimation'].tolist() + [agent.store.attempts]
        y = rows['value'].tolist() + [rows['value'].iloc[-1]]
        ax.step(x, y, where='post')
        ax.set(xlabel='Estimated models (attempts)', ylabel=metric)
        fig.tight_layout()
        fig.savefig(root / f'best_{metric}_new.png')
        plt.close(fig)
