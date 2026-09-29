"""Apply one named mutation to scripts/run_ppo.py (exactly one match required)."""

import sys
from pathlib import Path

MUTATIONS = {
    "promote-strict-gt": (
        "                    >= LAST_BEST_WIN_RATE_THRESHOLD\n",
        "                    > LAST_BEST_WIN_RATE_THRESHOLD\n",
    ),
    "skip-last-best-refresh": (
        "                    _refresh_eval_model_from_weights(\n"
        "                        last_best_model,\n",
        "                    (lambda *_a: None)(\n"
        "                        last_best_model,\n",
    ),
    "skip-promoted-checkpoint": (
        "                            last_best_checkpoint_path,\n"
        "                            env_steps=env_steps,\n",
        "                            run_dir / 'discarded_last_best.bin',\n"
        "                            env_steps=env_steps,\n",
    ),
    "skip-teacher-activation": (
        '                    if cfg.rl.teacher_mode == "last_best":\n'
        "                        trainer.set_teacher_model(\n"
        "                            last_best_model,\n",
        '                    if cfg.rl.teacher_mode == "never":\n'
        "                        trainer.set_teacher_model(\n"
        "                            last_best_model,\n",
    ),
    "final-checkpoint-step-zero": (
        "            run_dir / CHECKPOINT_FINAL,\n            env_steps=env_steps,\n",
        "            run_dir / CHECKPOINT_FINAL,\n            env_steps=0,\n",
    ),
    "replay-dir-always": (
        "                            if cfg.rl.eval_replay_games > 0\n",
        "                            if True\n",
    ),
}

old, new = MUTATIONS[sys.argv[1]]
path = Path("scripts/run_ppo.py")
text = path.read_text()
if text.count(old) != 1:
    raise SystemExit(f"{sys.argv[1]}: expected one match, got {text.count(old)}")
path.write_text(text.replace(old, new))
