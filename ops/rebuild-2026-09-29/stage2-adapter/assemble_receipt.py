"""Derive a complete command ledger, plain-text handoff and source custody."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
records = []
for line in (OUT / 'commands.jsonl').read_text().splitlines():
    records.append(json.loads(line))
for line in (OUT / 'native/commands.jsonl').read_text().splitlines():
    row = json.loads(line)
    row['name'] = 'native/' + Path(row['log']).stem
    row['log'] = 'native/' + row['log']
    records.append(row)
for folder in ('tables', 'eval'):
    for path in sorted((OUT / folder).glob('*.json')):
        row = json.loads(path.read_text())
        if 'command' in row and 'exit_code' in row:
            row['name'] = folder + '/' + path.stem
            row['log'] = folder + '/' + path.stem + '.log'
            records.append(row)
for row in records:
    if isinstance(row['command'], list):
        row['command'] = shlex.join(row['command'])
    content = (OUT / row['log']).read_text()
    row['test_summaries'] = [
        line.strip() for line in content.splitlines()
        if (re.search(r'\d+ (?:passed|failed|skipped|deselected).* in ', line)
            or line.startswith('test result:'))
    ]
    row['result_note'] = '; '.join(row['test_summaries']) or (
        'budget terminated before a complete test summary' if row.get('stop_reason')
        else 'not a test-count command; see log diagnostics'
    )
(OUT / 'command-ledger.json').write_text(json.dumps(records, indent=2) + '\n')
text = ['# Complete Stage 2 command ledger', '',
        'All commands use the offline/thread environment in results.md. Counts are per invocation, not additive; repeated red/green, umbrella and component runs overlap.', '',
        '| Receipt | Command | Actual exit | Test counts / limits |',
        '| --- | --- | ---: | --- |']
plain = ['COMPLETE VERIFICATION COMMAND LEDGER (per invocation, overlapping counts):']
for row in records:
    command = row['command'].replace('|', '\\|')
    note = row['result_note'].replace('|', '\\|')
    text.append(f"| {row['log']} | `{command}` | {row['exit_code']} | {note} |")
    plain.append(f"{row['log']}: {row['command']}\n  exit={row['exit_code']}; {row['result_note']}")
text += ['', 'The successful `rs-prepare` recipe also records these nested commands, each exit 0 (just would stop on any nonzero exit):',
         '', '- `uv run python scripts/check_engine_trim.py`', '- `cargo fmt`',
         '- `cargo fmt --manifest-path engine_rs/Cargo.toml --check`',
         '- `cargo clippy --all-targets -- -D warnings`',
         '- `cargo clippy --manifest-path engine_rs/Cargo.toml --all-targets --locked -- -D warnings -A clippy::too_many_arguments -A clippy::collapsible_if -A clippy::needless_range_loop`',
         '- `cargo test`: 274 passed, 5 ignored, 0 failed',
         '- `cargo test --manifest-path engine_rs/Cargo.toml --locked`: 69 passed, 0 failed, 0 ignored',
         '- `uv run python scripts/check_doc_freshness.py`', '',
         'The preparation attempts also completed (exit 0): `uv run maturin develop`; `uvx ruff check python/ scripts/ tests/ --select I --fix`; `uvx ruff format python/ scripts/ tests/`; `uv run python scripts/check_python_311_syntax.py`; `uvx ruff check python/ scripts/ tests/`; `uvx pymarkdownlnt scan *.md`; `uvx pymarkdownlnt scan --recurse python/ scripts/ tests/ src/ docs/`; `uv run mypy python/ scripts/` (69 source files). `uv run pytest tests/` / `uv run pytest tests/ -m "not slow"` did not finish before the watchdog killed their process group.']
(OUT / 'commands.md').write_text('\n'.join(text) + '\n')
report = (OUT / 'results.md').read_text()
body, verdict = report.rsplit('\nVERDICT:', 1)
body = re.sub(r'^#+ ', '', body, flags=re.MULTILINE).replace('`', '').replace('**', '')
(OUT / 'final-report.txt').write_text(body + '\n\n' + '\n'.join(plain) + '\n\nNested successful preparation commands: see commands.md and rs-prepare.log; their exit codes and counts are enumerated there.\n\nVERDICT:' + verdict)
paths = subprocess.check_output(['git', 'diff', '--name-only', '558ac3c'], cwd=ROOT, text=True).splitlines()
paths += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
source_paths = sorted(p for p in set(paths) if not p.startswith('ops/rebuild-2026-09-29/stage2-adapter/'))
manifest = {'base': '558ac3c2ffd475276c8221476c1a95f25229ddd8',
            'files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}}
(OUT / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
paths += ['ops/rebuild-2026-09-29/stage2-adapter/source-manifest.json', 'ops/rebuild-2026-09-29/stage2-adapter/files.txt']
(OUT / 'files.txt').write_text('\n'.join(sorted(set(paths))) + '\n')
print(f'{len(records)} logged verification invocations; {len(source_paths)} changed source/doc/test files; complete plain-text report generated')
