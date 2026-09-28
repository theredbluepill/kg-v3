import numpy as np
import pytest
import torch
from owl.kaggriculture.actor_codec import SLOT_NAMES, decode_action, encode_action
from owl.kaggriculture.gpu_sampling_grammar import GrammarBatch
from owl.kaggriculture.native_bridge import MyolieSampler


@pytest.mark.parametrize(
    "command", [["PICKUP", "WHEAT"], ["PICKUP", "WHEAT", 1], ["PLACE", "MILK", 1023]]
)
def test_python_codec_and_native_decoder_preserve_quantity_semantics(
    command: list[object],
) -> None:
    action = {"farmer": command, "hands": [], "market": [[], ["SELL", "WHEAT", 0]]}
    encoded = encode_action(action, hire_limit=241)
    frames = np.array(
        [[f[s] for s in SLOT_NAMES] for f in encoded["frames"]], dtype=np.int16
    )
    assert decode_action(encoded, hire_limit=241) == action
    assert MyolieSampler().decode(1, 10, 241, frames) == action


def test_native_device_grammar_replays_oracle_program_without_python_masks() -> None:
    action = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["PASS"]],
        "market": [["HIRE"], ["BUY_SEED", "WHEAT", 33]],
    }
    encoded = encode_action(action, hire_limit=241)
    table = GrammarBatch(MyolieSampler(), [(2, 10, 241)], "cpu")
    node = table.starts.clone()
    for frame in encoded["frames"]:
        for slot, name in enumerate(SLOT_NAMES):
            choice = torch.tensor([frame[name]])
            assert table.options(node, slot)[0, choice.item()]
            node = table.advance(node, choice, slot)
    assert node.tolist() == [-1]
    assert table.rebind([(2, 10, 241), (2, 10, 241)]).starts.shape == (2,)


def test_native_decoder_rejects_missing_stop() -> None:
    with pytest.raises(ValueError, match="length"):
        MyolieSampler().decode(1, 10, 241, np.zeros((1, 12), dtype=np.int16))
