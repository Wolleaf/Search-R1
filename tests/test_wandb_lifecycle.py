import sys
from types import SimpleNamespace

import pytest

from verl.trainer.main_ppo import _run_trainer_with_tracking
from verl.utils.tracking import Tracking


class _FakeRun:

    def __init__(self):
        self.log_calls = []
        self.finish_calls = []
        self.finish_error = None

    def log(self, **kwargs):
        self.log_calls.append(kwargs)

    def finish(self, exit_code):
        self.finish_calls.append(exit_code)
        if self.finish_error is not None:
            raise self.finish_error


def _install_fake_wandb(monkeypatch, run):
    module = SimpleNamespace(init_calls=[], login_calls=[])

    def init(**kwargs):
        module.init_calls.append(kwargs)
        return run

    module.init = init
    module.login = lambda **kwargs: module.login_calls.append(kwargs)
    monkeypatch.setitem(sys.modules, 'wandb', module)
    return module


def test_tracking_logs_to_initialized_run_and_finishes_once(monkeypatch):
    run = _FakeRun()
    wandb = _install_fake_wandb(monkeypatch, run)
    tracking = Tracking('project',
                        'experiment',
                        default_backend='wandb',
                        config={'seed': 42})

    tracking.log({'loss': 0.5}, step=1)
    tracking.finish(exit_code=0)
    tracking.finish(exit_code=1)

    assert tracking.logger['wandb'] is run
    assert wandb.init_calls == [{
        'project': 'project',
        'name': 'experiment',
        'config': {
            'seed': 42
        }
    }]
    assert run.log_calls == [{
        'data': {
            'loss': 0.5
        },
        'step': 1
    }]
    assert run.finish_calls == [0]


def test_tracking_finish_can_retry_after_failure(monkeypatch):
    run = _FakeRun()
    run.finish_error = RuntimeError('finish failed')
    _install_fake_wandb(monkeypatch, run)
    tracking = Tracking('project', 'experiment', default_backend='wandb')

    with pytest.raises(RuntimeError, match='finish failed'):
        tracking.finish(exit_code=0)

    run.finish_error = None
    tracking.finish(exit_code=0)

    assert run.finish_calls == [0, 0]


class _RecordingLogger:

    def __init__(self, events, finish_error=None):
        self.events = events
        self.finish_error = finish_error

    def finish(self, exit_code):
        self.events.append(('finish', exit_code))
        if self.finish_error is not None:
            raise self.finish_error


class _RecordingTrainer:

    def __init__(self, *, init_error=None, fit_error=None, finish_error=None):
        self.events = []
        self.init_error = init_error
        self.fit_error = fit_error
        self.logger = _RecordingLogger(self.events, finish_error)

    def init_workers(self):
        self.events.append('init_workers')
        if self.init_error is not None:
            raise self.init_error

    def fit(self):
        self.events.append('fit')
        if self.fit_error is not None:
            raise self.fit_error


def test_ray_owner_finishes_successful_training_with_zero():
    trainer = _RecordingTrainer()

    _run_trainer_with_tracking(trainer)

    assert trainer.events == ['init_workers', 'fit', ('finish', 0)]


@pytest.mark.parametrize('failure_stage', ['init_workers', 'fit'])
def test_ray_owner_best_effort_finishes_failures_without_masking(
        failure_stage):
    original_error = ValueError(f'{failure_stage} failed')
    trainer = _RecordingTrainer(
        init_error=original_error if failure_stage == 'init_workers' else None,
        fit_error=original_error if failure_stage == 'fit' else None,
        finish_error=RuntimeError('finish failed'))

    with pytest.raises(ValueError,
                       match=f'{failure_stage} failed') as exc_info:
        _run_trainer_with_tracking(trainer)

    assert exc_info.value is original_error
    assert trainer.events[-1] == ('finish', 1)


def test_ray_owner_propagates_successful_training_finish_failure():
    trainer = _RecordingTrainer(finish_error=RuntimeError('finish failed'))

    with pytest.raises(RuntimeError, match='finish failed'):
        _run_trainer_with_tracking(trainer)

    assert trainer.events == ['init_workers', 'fit', ('finish', 0)]
