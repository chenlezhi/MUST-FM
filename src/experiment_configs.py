from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    data_file: str
    dim: int
    delta: float
    batch_size: int
    n_epoch: int = 3000
    hidden_dim: int = 256
    n_hiddens: int = 5
    seed: int = 11
    lr_v: float = 0.005
    lr_g: float = 0.005
    eta_min: float = 1e-5
    hold_out: float | None = None
    use_mini_batch: bool = False  # do not change this parameter for MUST-FM
    chunk_size: int = 2000        # do not change this parameter for MUST-FM
    ema_decay: float | None = 0.995
    apply_ema_for_eval: bool = True
    plot_comparison: bool = True
    plot_growth_per_time: bool = False
    reducer_path: str | None = None
    plot_transparent: bool = True
    report_growth_correlation: bool = False
    independent: bool = True
    use_supervised_prior: bool = True


EXPERIMENTS = {
    "eb_50d": ExperimentConfig(
        experiment_name="eb_50d",
        data_file="data/eb_50d.csv",
        dim=50,
        delta=25.0,
        batch_size=512,
        independent=False,
        use_supervised_prior=False,
    ),
    "simulation_2d": ExperimentConfig(
        experiment_name="simulation_2d",
        data_file="data/simulation_gene_data_2d.csv",
        dim=2,
        delta=1.5,
        batch_size=400,
        report_growth_correlation=True,
        independent=False,
        use_supervised_prior=False,
    ),
    "gaussian_1000d": ExperimentConfig(
        experiment_name="gaussian_1000d",
        data_file="data/gaussian_1000d.csv",
        dim=1000,
        delta=1.4,
        batch_size=256,
        lr_v=1e-3,
        lr_g=1e-3,
        ema_decay=None,
        apply_ema_for_eval=False,
        independent=False,
        use_supervised_prior=False,
    ),
    "weinreb_2d": ExperimentConfig(
        experiment_name="weinreb_2d",
        data_file="data/Weinreb_2d.csv",
        dim=2,
        delta=15.0,
        batch_size=512,
        independent=False,
        use_supervised_prior=False,
    ),
    "multiscale_2d": ExperimentConfig(
        experiment_name="multiscale_2d",
        data_file="data/multiscale_simulation_data_2d.csv",
        dim=2,
        delta=5.0,
        batch_size=256,
        seed=42,
        independent=False,
        use_supervised_prior=False,
    ),
    "MOCA_10d": ExperimentConfig(
        experiment_name="MOCA_10d",
        data_file="data/MOCA_10d.csv",
        dim=10,
        delta=1.0,
        batch_size=256,
        seed=42,
        independent=True,
        use_supervised_prior=True,
    ),
}
