import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

root = Path.cwd()
path = root / 'ops/rebuild-2026-09-29/merge-1.2/update_trim_manifest.py'
spec = importlib.util.spec_from_file_location('updater_under_verification', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
snapshots = {rev: json.loads(subprocess.check_output(['git', 'show', f'{rev}:engine_rs/TRIM_MANIFEST.json'])) for rev in [module.BASE, module.OURS, module.THEIRS]}
report = []
for key in ('schema_version', 'reference_commit', 'retained', 'excluded'):
    copies = copy.deepcopy(snapshots)
    if key == 'schema_version':
        copies[module.THEIRS][key] = 2
    elif key == 'reference_commit':
        copies[module.THEIRS][key] = '0' * 40
    else:
        copies[module.THEIRS][key][0]['reference_sha256'] = '0' * 64
    with tempfile.TemporaryDirectory(prefix='kg-merge84-updater-', dir='/private/tmp') as directory:
        module.ROOT = Path(directory)
        module.load = copies.__getitem__
        try:
            module.main()
        except SystemExit as error:
            assert str(error) == f'{key} differs between sides; vendored inventory drift'
            assert not list(module.ROOT.iterdir())
            report.append({'mutation': key + ' differs on theirs', 'result': 'rejected before write', 'error': str(error)})
        else:
            raise AssertionError(key + ' drift accepted')
(root / 'ops/rebuild-2026-09-29/codex/verify-merge-84dd76ad/updater-controls.json').write_text(json.dumps(report, indent=2) + '\n')
print('All 4 updater drift controls rejected before writing output')
