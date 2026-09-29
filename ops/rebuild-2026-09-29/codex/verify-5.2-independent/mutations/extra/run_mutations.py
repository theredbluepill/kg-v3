from pathlib import Path
import hashlib, json, os, subprocess, time

EVIDENCE=Path(__file__).resolve().parent
INFO=json.loads((EVIDENCE/'scratch.json').read_text())
SCRATCH=Path(INFO['scratch'])
REPO=Path(INFO['repo'])
PYTHON=INFO['python']
BC='python/owl/train/bc.py'
DATA='python/owl/kaggriculture/bc_data.py'
SCRIPT='scripts/train_bc.py'
ALL='tests/kaggriculture/test_bc.py'
CASES=[]
def case(name,file,old,new,test=''):
    CASES.append(dict(name=name,file=file,old=old,new=new,test=test))
# Each existing test has at least one mutant; compound guard tests split below.
case('round_trip_restore_int64',DATA,'return values.to(torch.int64) if values.dtype in _COMPACT_INTEGERS else values','return values','shards_round_trip')
case('tampered_hash',DATA,'if _sha256(path) != record.shard_sha256:','if False:','tampered')
case('duplicate_episode',DATA,'if duplicates:','if False:','tampered')
case('old_schema',DATA,'if "schema" not in names or str(shard["schema"][()]) != SHARD_SCHEMA:','if False:','foreign_shard')
case('missing_split',DATA,'if not records:','if False:','both_splits')
case('unpaired_rows',DATA,'if not bool(obs.still_playing.all()):','if False:','both_splits')
case('integer_range',DATA,'if values.numel() and (\n        int(values.min()) < info.min or int(values.max()) > info.max\n    ):','if False:','range_checks')
case('rank_partition',DATA,'mine = slice(rank, None, world_size)','mine = slice(0, None, world_size)','fake_ranks')
case('winner_draw',BC,'torch.where(final_banks[:, 0] < final_banks[:, 1], 0.0, 0.5)','torch.where(final_banks[:, 0] < final_banks[:, 1], 0.0, 0.0)','winner_targets')
case('winner_seat_order',BC,'self_wins = torch.stack((seat0, 1.0 - seat0), dim=1)','self_wins = torch.stack((seat0, seat0), dim=1)','winner_targets')
case('loss_sign',BC,'return terms.turn_nll.mean() + value_coef * terms.value_ce.mean()','return -terms.turn_nll.mean() - value_coef * terms.value_ce.mean()','loss_decreases')
case('critic_omitted',BC,'return terms.turn_nll.mean() + value_coef * terms.value_ce.mean()','return terms.turn_nll.mean()','every_parameter')
case('rank_seed',BC,'np.random.default_rng([seed, epoch, rank])','np.random.default_rng([seed, epoch, 0])','sharding_is_deterministic')
case('epoch_seed',BC,'np.random.default_rng([seed, epoch, rank])','np.random.default_rng([seed, 0, rank])','sharding_is_deterministic')
case('small_partition',BC,'if self.steps_per_epoch < 1:','if False:','sharding_is_deterministic')
case('min_delta',BC,'if nll < self.best_nll - self.min_delta:','if nll < self.best_nll:','held_out_selection')
case('eval_step_advance',BC,'if step <= self.last_step:','if False:','held_out_selection')
case('finite_nll',BC,'if not math.isfinite(nll):','if False:','held_out_selection')
case('patience_stop',BC,'return self.evals_since_best >= self.patience_evals','return False','held_out_selection or training_keeps')
case('best_save_unconditional',BC,'            if improved:\n                _save_best(','            if True:\n                _save_best(','training_keeps')
case('resume_step_reset',BC,'step = resume.step','step = 0','restart_from_state')
case('resume_manifest_guard',BC,'if state.dataset_manifest_sha256 != dataset.manifest_sha256:','if False:','resume_rejects')
case('resume_world_guard',BC,'if state.world_size != world_size:','if False:','resume_rejects')
case('ppo_checkpoint_schema',BC,'        "env_steps": 0,','        "env_steps": 12,','best_checkpoint_loads')
case('ppo_checkpoint_optimizer_step',BC,'        "optimizer_steps": bc_steps,','        "optimizer_steps": 0,','best_checkpoint_loads')
case('bc_workload_validation_removed',BC,'        ForwardWorkload(\n            "bc_validation", config.validation_rows_per_forward * kt.PLAYERS\n        ),','', 'two_rank_bc_config')
case('script_provenance_ignored',SCRIPT,'"source_commit": args.source_commit or _git_head(),','"source_commit": "incorrect",','script_runs')
case('observation_field_removed',DATA,'if name != "action_mask"','if name not in ("action_mask", "globals_int")','contract_shapes')
# Remaining explicit input/state guards and unsupported seam claims, full new suite.
case('writer_paired_shape',DATA,'if tuple(episode.obs.still_playing.shape) != lead or (\n        tuple(episode.actions.lengths.shape) != lead\n    ):','if False:')
case('loader_rank',DATA,'if not 0 <= rank < world_size:','if False:')
case('loader_manifest_exists',DATA,'if not manifest_path.is_file():','if False:')
case('banks_finite',DATA,'if not all(math.isfinite(bank) for bank in record.terminal_banks):','if False:')
case('shard_exists',DATA,'if not path.is_file():','if False:')
case('shard_bytes',DATA,'if path.stat().st_size != record.shard_bytes:','if False:')
case('shard_arrays',DATA,'if names != SHARD_ARRAYS:','if False:')
case('shard_admitted_shape',DATA,'if array.shape[:1] != (record.admitted,):','if False:')
case('turn_layout',DATA,'if turn.dtype != np.int64 or turn.ndim != 1:','if False:')
case('turn_increasing',DATA,'if int(turn[0]) < 0 or not bool(np.all(np.diff(turn) > 0)):','if False:')
case('action_layout',DATA,'if tensors[name].dtype != dtype or tuple(tensors[name].shape[1:]) != trailing:','if False:')
case('observation_contract',DATA,'        obs.check_contract()','        pass')
case('length_bounds',DATA,'if int(lengths.min()) < 1 or bool(\n        (lengths > obs.action_mask.can_act.sum(dim=-1)).any()\n    ):','if False:')
case('tokens_nonnegative',DATA,'if int(tensors["tokens"].min()) < 0:','if False:')
case('shard_path',DATA,'    posix = PurePosixPath(path)','    return\n    posix = PurePosixPath(path)')
case('gather_nonempty',DATA,'if index.ndim != 1 or index.numel() == 0:','if False:')
case('gather_range',DATA,'if int(index.min()) < 0 or int(index.max()) >= self.num_rows:','if False:')
case('obs_field_tensor',DATA,'if not isinstance(value, torch.Tensor):','if False:')
case('config_game',BC,'if not isinstance(cfg.model, KaggricultureTransformerConfig) or not isinstance(\n        cfg.env, KaggricultureEnvConfig\n    ):','if False:')
case('sharding_step_micro',BC,'if step < 0 or not 0 <= micro < self.gradient_accumulation_steps:','if False:')
case('sharding_rank',BC,'if not 0 <= rank < self.world_size:','if False:')
case('winner_shape',BC,'if final_banks.ndim != 2 or final_banks.shape[1] != kt.PLAYERS:','if False:')
case('winner_log_probs_present',BC,'if evaluation.winner_log_probabilities is None:','if False:')
case('evaluation_nonempty',BC,'if seat_rows == 0:','if False:')
case('resume_state_keys',BC,'if not isinstance(state, dict) or set(state) != _RESUME_KEYS:','if False:')
case('resume_selection_mapping',BC,'if not isinstance(selection, dict):','if False:')
case('resume_lr_schedule',BC,'if (lr_scheduler is None) != (state.lr_scheduler is None):','if False:')
case('trainer_rank_context',BC,'if (dataset.rank, dataset.world_size) != (context.rank, context.world_size):','if False:')
case('history_exists',BC,'if not path.is_file():','if False:')
case('script_provenance_mapping',SCRIPT,'if not isinstance(record, dict):','if False:')
case('script_override_syntax',SCRIPT,'if not separator or not field_path:','if False:')
case('script_override_duplicate',SCRIPT,'if field_path in overrides:','if False:')
case('script_fresh_config_exists',SCRIPT,'if fresh and not args.target.is_file():','if False:')
case('script_resume_dir',SCRIPT,'if not args.target.is_dir():','if False:')
case('script_resume_overrides',SCRIPT,'if args.overrides is not None:','if False:')
case('script_resume_source_override',SCRIPT,'if args.source_commit is not None:','if False:')
case('script_runtime_positive',SCRIPT,'if args.max_runtime_hours is not None and args.max_runtime_hours <= 0.0:','if False:')
# Pydantic configurable safety guards: each is exercised against unchanged suite.
for key,decl in [('seed','Field(ge=0)'),('rows_per_rank','Field(ge=1)'),('gradient_accumulation_steps','Field(default=1, ge=1)'),('max_grad_norm','Field(gt=0.0)'),('value_coef','Field(default=1.0, ge=0.0)'),('eval_interval_steps','Field(ge=1)'),('patience_evals','Field(ge=1)'),('min_delta','Field(default=0.0, ge=0.0)'),('max_steps','Field(ge=1)'),('validation_rows_per_forward','Field(ge=1)')]:
    typ='float' if key in ('max_grad_norm','value_coef','min_delta') else 'int'
    newdecl='Field(default=1)' if 'default=1,' in decl else 'Field(default=1.0)' if 'default=1.0,' in decl else 'Field(default=0.0)' if 'default=0.0,' in decl else 'Field()'
    case('config_'+key,BC,f'{key}: {typ} = {decl}',f'{key}: {typ} = {newdecl}')
case('policy_length_normalization',BC,'turn_nll=program_nll / frames,','turn_nll=program_nll,')
case('autocast_seam',BC,'with autocast_context(ppo_config.rl, device):','with torch.autocast(device_type=device.type, enabled=False):')
case('compile_seam',SCRIPT,'compiled = configure_model_compile(model, ppo_config.rl)','compiled = []')
case('ddp_seam',BC,'return wrap_model_for_distributed(cast(BaseModelAPI, model), context)','return model')
case('workload_guard_seam',SCRIPT,'headroom = check_bc_workload(bc_config, ppo_config)','headroom = []')
case('compile_stack_seam',SCRIPT,'if ppo_config.rl.model_compile != "none":','if False:')
case('optimizer_step_removed',BC,'    optimizer.step()','    pass')
case('gradient_accumulation_scaling',BC,'(loss / accumulation).backward()','loss.backward()')
case('gradient_clipping_disabled',BC,'        config.max_grad_norm,','        float("inf"),')
case('nonfinite_gradient_guard',BC,'        error_if_nonfinite=True,','        error_if_nonfinite=False,')
case('scheduler_step_removed',BC,'        lr_scheduler.step()','        pass')
case('preflight_disabled',BC,'        if config.preflight_train_replay:','        if False:')
case('runtime_cap_removed',BC,'if max_runtime_seconds is not None and all_reduce_any(','if False and all_reduce_any(')
case('best_checkpoint_weights_zero',BC,'return {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}','return {k: torch.zeros_like(v).cpu() for k, v in unwrap_model(model).state_dict().items()}','best_checkpoint_loads or training_keeps')

CASES=[]
case('manifest_episode_id_nonempty',DATA,'episode_id: str = Field(min_length=1)','episode_id: str = Field()')
case('manifest_admitted_nonnegative',DATA,'admitted: int = Field(ge=0)','admitted: int = Field()')
case('manifest_shard_path_nonempty',DATA,'shard_path: str = Field(min_length=1)','shard_path: str = Field()')
case('manifest_sha256_format',DATA,'shard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")','shard_sha256: str = Field()')
case('manifest_shard_bytes_positive',DATA,'shard_bytes: int = Field(ge=1)','shard_bytes: int = Field()')
case('manifest_schema_literal',DATA,'schema_id: Literal["kaggriculture-bc-shard-v1"] = Field(alias="schema")','schema_id: str = Field(alias="schema")')
case('manifest_observation_schema_literal',DATA,'observation_schema_version: Literal[3]','observation_schema_version: int')
case('manifest_contract_document_literal',DATA,'contract_document_version: Literal[4]','contract_document_version: int')
case('manifest_split_literal',DATA,'    split: Split\n    terminal_banks:', '    split: str\n    terminal_banks:')

case('writer_exclusive_shard',DATA,'with path.open("xb") as handle:', 'with path.open("wb") as handle:')
case('writer_exclusive_manifest',DATA,'with (root / MANIFEST_NAME).open("x", encoding="utf-8") as handle:', 'with (root / MANIFEST_NAME).open("w", encoding="utf-8") as handle:')
case('rank_seed_full18_confirmation',BC,'np.random.default_rng([seed, epoch, rank])','np.random.default_rng([seed, epoch, 0])')
case('loss_sign_full18_confirmation',BC,'return terms.turn_nll.mean() + value_coef * terms.value_ce.mean()','return -terms.turn_nll.mean() - value_coef * terms.value_ce.mean()')
paths={c['file'] for c in CASES}
original={p:(SCRATCH/p).read_bytes() for p in paths}
original_repo={p:(REPO/p).read_bytes() for p in paths}
sha=lambda b: hashlib.sha256(b).hexdigest()
manifest={'source': {p:sha(b) for p,b in original.items()},'scratch':str(SCRATCH),'python':PYTHON,'cases':CASES}
(EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2))
env=os.environ.copy()
env['PYTHONPATH']=f'{SCRATCH}/python:{SCRATCH}'
env['PYTHONDONTWRITEBYTECODE']='1'
results=[]
try:
  for index,c in enumerate(CASES,1):
    path=SCRATCH/c['file']
    raw=original[c['file']].decode()
    count=raw.count(c['old'])
    if count != 1:
      result={'index':index,'name':c['name'],'status':'SETUP_ERROR','reason':f'replacement count {count}'}
      results.append(result)
      print(json.dumps(result),flush=True)
      continue
    start=time.monotonic()
    path.write_bytes(raw.replace(c['old'],c['new']).encode())
    try:
      cmd=[PYTHON,'-B','-m','pytest',ALL,'-q']
      if c['test']: cmd += ['-k',c['test']]
      proc=subprocess.run(cmd,cwd=SCRATCH,env=env,capture_output=True,text=True,timeout=90)
      output=proc.stdout+proc.stderr
      status='SURVIVED' if proc.returncode == 0 else 'KILLED' if proc.returncode == 1 else 'SETUP_ERROR'
      (EVIDENCE/f'{index:03d}-{c["name"]}.log').write_text(output)
      summary='\n'.join(output.splitlines()[-4:])
      result={'index':index,'name':c['name'],'status':status,'exit_code':proc.returncode,'seconds':round(time.monotonic()-start,3),'summary':summary}
    except subprocess.TimeoutExpired as error:
      result={'index':index,'name':c['name'],'status':'TIMEOUT','seconds':round(time.monotonic()-start,3)}
    finally:
      path.write_bytes(original[c['file']])
      assert path.read_bytes() == original[c['file']]
    result['restored_sha256']=sha(path.read_bytes())
    results.append(result)
    (EVIDENCE/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(result),flush=True)
finally:
  for p,b in original.items():
    (SCRATCH/p).write_bytes(b)
    assert (SCRATCH/p).read_bytes() == b
    assert (REPO/p).read_bytes() == original_repo[p],f'original changed: {p}'
  (EVIDENCE/'restoration.json').write_text(json.dumps({'scratch':{p:sha((SCRATCH/p).read_bytes()) for p in paths},'repo':{p:sha((REPO/p).read_bytes()) for p in paths},'all_byte_exact':True},indent=2))
print(json.dumps({'counts':{s:sum(r['status']==s for r in results) for s in ('KILLED','SURVIVED','SETUP_ERROR','TIMEOUT')},'total':len(results)}),flush=True)
