import numpy as np
import pytest

from scripts.handcap_flexiv_remote import decode_handcap_action_chunk
from scripts.handcap_flexiv_remote import encode_state_to_handcap_state
from scripts.handcap_flexiv_remote import _make_policy_observation


def test_encode_state_to_handcap_state_has_expected_shape() -> None:
    eepose = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03], dtype=np.float64)
    state = encode_state_to_handcap_state(eepose, 0.045)

    assert state.shape == (10,)
    np.testing.assert_allclose(state[:9], np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert state[-1] == pytest.approx(0.045)


def test_decode_handcap_action_chunk_reads_relative_pose() -> None:
    eepose = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03], dtype=np.float64)
    state = encode_state_to_handcap_state(eepose, 0.045)

    decoded = decode_handcap_action_chunk(state[None, :])

    assert decoded.shape == (1, 7)
    np.testing.assert_allclose(decoded[0, :6], np.zeros(6), atol=1e-6)
    assert decoded[0, 6] == pytest.approx(0.045)


def test_decode_handcap_action_chunk_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="handcap action dim 10"):
        decode_handcap_action_chunk(np.zeros((5, 7), dtype=np.float64))


def test_make_policy_observation_includes_server_keys() -> None:
    frame = {
        "wrist_img": np.zeros((480, 640, 3), dtype=np.uint8),
        "left_tactile_img": None,
        "right_tactile_img": None,
        "eepose": np.zeros(6, dtype=np.float64),
        "gripper_width": 0.045,
    }

    obs = _make_policy_observation(frame, "simple sorting task", use_tactile=False)

    assert "observation/state" in obs
    assert "observation/wrist_image" in obs
    assert "observation/left_tactile" in obs
    assert "observation/right_tactile" in obs
    assert obs["observation/state"].shape == (10,)
