from verl.trainer.ppo.ray_trainer import _next_training_step


def test_exactly_n_one_based_training_steps_execute():
    total_steps = 60
    step = 1
    executed = []

    while step is not None:
        executed.append(step)
        step = _next_training_step(step, total_steps)

    assert executed == list(range(1, total_steps + 1))


def test_final_step_has_no_successor():
    assert _next_training_step(60, 60) is None
