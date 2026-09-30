import pytest
from owl.model.kaggriculture_teacher import slot_frames

@pytest.mark.parametrize("slot", [-1, 0, 2, 11, 12])
def test_invalid_slot_frames_rejected(slot):
    with pytest.raises(ValueError, match="not a policy slot"):
        slot_frames(slot)
