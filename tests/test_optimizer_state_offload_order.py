import subprocess
import sys
from pathlib import Path


def test_optimizer_callbacks_follow_backward_and_survive_step_failure():
    script = r'''
import sys
import types
from types import SimpleNamespace

import torch

try:
    import ray  # noqa: F401
except ModuleNotFoundError:
    ray = types.ModuleType('ray')
    ray.ObjectRef = type('ObjectRef', (), {})
    ray.get = lambda value: value
    sys.modules['ray'] = ray

from verl.workers.actor.dp_actor import DataParallelPPOActor


class RecordingAdamW(torch.optim.AdamW):

    def __init__(self, params, events):
        super().__init__(params, lr=0.1)
        self.events = events
        self.fail_next_step = False

    def step(self, closure=None):
        self.events.append('step')
        if self.fail_next_step:
            raise RuntimeError('expected optimizer failure')
        return super().step(closure=closure)


events = []
module = torch.nn.Linear(1, 1, bias=False)
optimizer = RecordingAdamW(module.parameters(), events)
module.weight.register_hook(lambda grad: events.append('backward'))

actor = object.__new__(DataParallelPPOActor)
actor.config = SimpleNamespace(grad_clip=1.0)
actor.actor_module = module
actor.actor_optimizer = optimizer
actor.optimizer_state_load_fn = lambda: events.append('load')
actor.optimizer_state_offload_fn = lambda: events.append('offload')

# Two micro-batches accumulate gradients before the first optimizer step.
module(torch.tensor([[1.0]])).sum().backward()
module(torch.tensor([[2.0]])).sum().backward()
assert events == ['backward', 'backward']
actor._optimizer_step()

optimizer.zero_grad()
module(torch.tensor([[3.0]])).sum().backward()
actor._optimizer_step()

assert events == [
    'backward',
    'backward',
    'load',
    'step',
    'offload',
    'backward',
    'load',
    'step',
    'offload',
]

optimizer.zero_grad()
module(torch.tensor([[4.0]])).sum().backward()
optimizer.fail_next_step = True
try:
    actor._optimizer_step()
except RuntimeError as exc:
    assert str(exc) == 'expected optimizer failure'
else:
    raise AssertionError('optimizer failure was not propagated')

assert events[-4:] == ['backward', 'load', 'step', 'offload']
'''

    subprocess.run([sys.executable, '-c', script],
                   cwd=Path(__file__).resolve().parents[1],
                   check=True,
                   capture_output=True,
                   text=True)
