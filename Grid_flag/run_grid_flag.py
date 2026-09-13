"""Run from the project root: python -m Grid_flag.run_grid_flag --help."""
import argparse
from Grid_flag.grid_flag_env import GridFlagEnv
from Grid_flag.grid_flag_agent import GridFlagAgent

FLAG_CELLS = [(1, 2), (3, 7), (6, 1), (7, 8), (9, 4)]
ENV_KWARGS = dict(grid_size=(10, 10), max_step=100, agent_start=(5, 5),
                  flag_value=10, flag_cells=FLAG_CELLS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-root')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args(argv)
    for algorithm in ('qlearning', 'sarsa'):
        agent = GridFlagAgent(GridFlagEnv(**ENV_KWARGS), learning_rate=.1,
                              initial_epsilon=1., final_epsilon=.05,
                              epsilon_decay=.95/5000, discount_factor=.95, seed=args.seed)
        result = agent.train(args.episodes, algorithm=algorithm, output_root=args.output_root,
                             record_every=500 if args.record else None)
        print(result.artifacts)
        print(agent.evaluate(output_root=args.output_root, record=args.record).episodes)


if __name__ == '__main__':
    main()
