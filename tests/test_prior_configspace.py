from ConfigSpace.hyperparameters import NormalFloatHyperparameter, NormalIntegerHyperparameter

from priorbai.benchmarks import LCBenchBenchmark


def test_create_prior_configspace():
    bench = LCBenchBenchmark(dataset_id=0, seed=0, priorband=True, n_prior_construction=10)

    cs = bench.prior_configspace
    hps = cs.get_hyperparameters()

    assert len(hps) > 0, "prior_configspace has no hyperparameters"
    for hp in hps:
        assert isinstance(hp, (NormalFloatHyperparameter, NormalIntegerHyperparameter)), (
            f"{hp.name} is {type(hp)}, expected Normal"
        )
        original_hp = bench.configuration_space.get_hyperparameter(hp.name)
        assert hp.lower == original_hp.lower
        assert hp.upper == original_hp.upper
        assert original_hp.lower <= hp.mu <= original_hp.upper, (
            f"{hp.name}: mu={hp.mu} outside [{original_hp.lower}, {original_hp.upper}]"
        )

    print(f"OK — {len(hps)} hyperparameters, all Normal, mu within bounds")


if __name__ == "__main__":
    test_create_prior_configspace()
