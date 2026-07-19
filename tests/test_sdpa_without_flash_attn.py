import subprocess
import sys
from pathlib import Path


def test_data_parallel_workers_import_without_flash_attn():
    script = r'''
import builtins
import sys
import types

try:
    import ray  # noqa: F401
except ModuleNotFoundError:
    ray = types.ModuleType("ray")
    ray.ObjectRef = type("ObjectRef", (), {})
    ray.get = lambda value: value
    sys.modules["ray"] = ray

real_import = builtins.__import__

def import_without_flash_attn(name, *args, **kwargs):
    if name == "flash_attn" or name.startswith("flash_attn."):
        raise ModuleNotFoundError("flash_attn intentionally unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_flash_attn
import verl.workers.actor.dp_actor
import verl.workers.critic.dp_critic
'''

    subprocess.run([sys.executable, '-c', script],
                   cwd=Path(__file__).resolve().parents[1],
                   check=True,
                   capture_output=True,
                   text=True)
