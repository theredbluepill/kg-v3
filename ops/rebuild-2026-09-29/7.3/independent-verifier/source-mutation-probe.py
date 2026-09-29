import copy, json, pathlib, sys
from owl import rs
from tests.kaggriculture.test_replay_export_oracles import _run_framework_game, _header_tape, _captured, _fixture_episode, _dump
mode=sys.argv[1]
episode=_run_framework_game()["episode"]
header,tape=_header_tape(episode)
exported=rs.export_kaggriculture_episode(_dump(header),_dump(tape))
results={"mode":mode,"extension_path":rs.__file__}
if mode == "byte":
    assert '"money":3000.0' in exported
    changed=exported.replace('"money":3000.0','"money":3.0e3',1)
    assert changed != exported
    try:
        result=json.loads(rs.verify_kaggriculture_episode(changed))
        results["same_kind_money_spelling"]={"accepted":result["ok"],"mode":result["mode"]}
    except ValueError as error:
        results["same_kind_money_spelling"]={"accepted":False,"error":str(error)}
else:
    cases=[]
    for episode_id in ("95324500", "95901360", "95921764", "95990191"):
        changed=_fixture_episode(episode_id,episode)
        changed["steps"][-1][0]["reward"]=0.8
        changed["rewards"][0]=0.8
        for seat in changed["steps"][-1]:
            seat["observation"]["farms"][0]["money"]=0.8
        cases.append(("fixture_reward_"+episode_id,changed,None))
    changed=copy.deepcopy(episode); changed["info"]["seed"]+=1
    cases.append(("framework_seed",changed,None))
    evidence=_captured(episode); evidence["banks"][1][0]+=1
    cases.append(("captured_bank",json.loads(exported),evidence))
    changed=copy.deepcopy(episode)
    changed["steps"][1][1]["observation"]["private"]=copy.deepcopy(changed["steps"][1][0]["observation"]["private"])
    cases.append(("private_leakage",changed,None))
    changed=copy.deepcopy(episode); seeds=changed["steps"][0][0]["observation"]["private"]["seeds"]
    changed["steps"][0][0]["observation"]["private"]["seeds"]=dict(reversed(list(seeds.items())))
    cases.append(("private_map_order",changed,None))
    changed=copy.deepcopy(episode); changed["steps"][1][0]["reward"]=0.0
    cases.append(("reward_number_kind",changed,None))
    for label,changed,evidence in cases:
        try:
            result=json.loads(rs.verify_kaggriculture_episode(_dump(changed),None if evidence is None else _dump(evidence)))
            results[label]={"accepted":result["ok"],"mode":result["mode"]}
        except ValueError as error:
            results[label]={"accepted":False,"error":str(error)}
print(json.dumps(results,indent=2))
