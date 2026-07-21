import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "export_gated_training.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("export_gated_training", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ExportGatedTrainingTest(unittest.TestCase):

    def test_renders_all_registered_panels_without_plot_dependencies(self):
        rows = []
        for step in (1, 2):
            rows.append({field: step / 10 for field, _ in MODULE.PANELS})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "curves.svg"
            MODULE.render_svg(output, rows)
            rendered = output.read_text(encoding="utf-8")
        self.assertIn("C-gated training dynamics", rendered)
        self.assertEqual(rendered.count('<polyline class="line"'), 6)
        self.assertIn("Post-hoc utility", rendered)


if __name__ == "__main__":
    unittest.main()
