from __future__ import annotations

from misc import solver_threshold_ablation


def test_explicit_state_size_grid_parses_without_changing_defaults() -> None:
    args = solver_threshold_ablation.build_parser().parse_args(
        ["--state-sizes", "400", "600", "900"]
    )

    assert args.state_sizes == [400, 600, 900]
    assert args.rank == 10
    assert args.repeats == 3
