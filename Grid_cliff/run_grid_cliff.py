"""Run from the project root: python -m Grid_cliff.run_grid_cliff --help."""
import argparse
from Grid_cliff.grid_cliff_algorithms import train, plot_comparison
from rl_project.runtime import Run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-root')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args(argv)
    results = []
    for algorithm in ('qlearning', 'sarsa'):
        agent, result = train(algorithm, args.episodes, args.seed, args.output_root,
                              record_every=500 if args.record else None)
        results.append(result)
        print(result.artifacts)
        print(agent.evaluate(seed=1234, output_root=args.output_root, record=args.record).episodes)
    comparison = Run('cliff', 'comparison', args.seed,
                     {'kind': 'comparison', 'source_runs': [str(r.run_dir) for r in results]}, args.output_root)
    plot_comparison(*results, comparison.path / 'plots/cliff_qlearning_vs_sarsa.png')
    comparison.finish()


if __name__ == '__main__':
    main()
