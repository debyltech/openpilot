import numpy as np
import pytest

from openpilot.selfdrive.controls.controlsd import PREAP_GRADE_LOOKAHEAD_MAX, PREAP_GRADE_LOOKAHEAD_S, preap_lookahead_pitch
from openpilot.selfdrive.modeld.constants import ModelConstants


def test_pitch_is_the_models_pitch_at_the_lookahead():
  orientation_y = 0.02 * np.array(ModelConstants.T_IDXS)  # pitch rising 0.02 rad/s along the plan
  assert preap_lookahead_pitch(0.01, orientation_y) == pytest.approx(0.01 + 0.02 * PREAP_GRADE_LOOKAHEAD_S)


def test_a_bad_model_frame_moves_the_pitch_at_most_the_cap():
  assert preap_lookahead_pitch(0.0, np.full(ModelConstants.IDX_N, 1.0)) == PREAP_GRADE_LOOKAHEAD_MAX
  assert preap_lookahead_pitch(0.0, np.full(ModelConstants.IDX_N, -1.0)) == -PREAP_GRADE_LOOKAHEAD_MAX


def test_no_model_plan_keeps_the_measured_pitch():
  assert preap_lookahead_pitch(0.03, []) == 0.03
