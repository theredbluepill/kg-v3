import importlib.util
import json
import sys
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location('r2_generator', root / 'scripts/kaggriculture_parity/generate_traces.py')
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)
pin = g.pinned_engine()
kaggle, module, digest = g.load_pinned_kaggle(pin)
print(json.dumps({'installed_version': g.importlib.metadata.version('kaggle-environments'), 'engine_path': module.__file__, 'sha256': digest, 'core_path': kaggle.core.__file__}))
bad_pin = g.EnginePin(pin.version, '0' * 64)
g.pinned_engine = lambda: bad_pin
out = root / '.codex-tmp/verify-1.1b-r2/guard-refused-output'
code = g.main(['--preset', 'committed', '--out', str(out)])
print(json.dumps({'main_return_code': code, 'out_exists': out.exists()}))
assert code == 2 and not out.exists()
