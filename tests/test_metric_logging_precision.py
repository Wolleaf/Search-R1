from verl.utils.logger.aggregate_logger import concat_dict_to_str
from verl.trainer.ppo.ray_trainer import _validation_metrics


def test_console_metrics_keep_checkpoint_selection_precision():
    line = concat_dict_to_str({'val/utility/nq': 0.1234564}, step=20)

    assert line == 'step:20 - val/utility/nq:0.123456'


def test_validation_reports_em_and_posthoc_utility_separately():
    metrics = _validation_metrics(
        ['nq', 'nq'],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.975, -0.025],
    )

    assert metrics['val/test_score/nq'] == 0.5
    assert metrics['val/em/nq'] == 0.5
    assert metrics['val/utility/nq'] == 0.475
