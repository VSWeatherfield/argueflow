def test_imports():
    from argueflow.cli.commands import CLI
    from argueflow.eval.eval import evaluate
    from argueflow.infer.infer import inference
    from argueflow.train.train import train
    from argueflow.utils.dvc_utils import download_data
    from argueflow.utils.logging_utils import setup_logging_from_cfg

    # CLI, train, evaluate, inference, download_data, setup_logging_from_cfg
    assert all(
        [CLI, train, evaluate, inference, download_data, setup_logging_from_cfg]
    ), "Import issues detected"

    assert True, "Import issues detected"
