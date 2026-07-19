from verl.utils.logger.aggregate_logger import concat_dict_to_str


def test_console_metrics_keep_checkpoint_selection_precision():
    line = concat_dict_to_str({'val/utility/nq': 0.1234564}, step=20)

    assert line == 'step:20 - val/utility/nq:0.123456'
