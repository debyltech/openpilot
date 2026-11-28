import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_planner import EXP_HOLD_FRAMES, LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_tesla_preap_following import _make_planner_inputs, _make_preap_params
from openpilot.selfdrive.modeld.constants import ModelConstants

V_EGO, V_SET = 22.0, 27.0  # model cruising ~11 mph under the set speed


class _Params:
  def __init__(self, hold):
    self.hold = hold

  def get(self, key, return_default=False):
    return 4

  def get_bool(self, key):
    return self.hold if key == "NAPExpHoldSetSpeed" else False


def _planner_and_inputs(hold=True):
  planner = LongitudinalPlanner(_make_preap_params(), init_v=V_EGO, params=_Params(hold))
  sm = _make_planner_inputs(V_EGO)
  sm['selfdriveState'].experimentalMode = True
  sm['carState'].vCruise = V_SET * 3.6
  sm['modelV2'].action.desiredAcceleration = 0.0
  return planner, sm


def _run(planner, sm, frames):
  for _ in range(frames):
    planner.update(sm)
  return planner.mpc.source, planner.output_a_target


def test_open_road_with_a_flat_model_plan_cruises_to_the_set_speed():
  planner, sm = _planner_and_inputs()
  source, a_target = _run(planner, sm, EXP_HOLD_FRAMES - 1)
  assert source == LongitudinalPlanSource.e2e  # the model keeps control until open road has held 1 s
  source, a_target = _run(planner, sm, 40)
  assert source == LongitudinalPlanSource.cruise and a_target > 0.3


def test_the_model_takes_over_the_moment_its_plan_slows():
  planner, sm = _planner_and_inputs()
  _run(planner, sm, EXP_HOLD_FRAMES + 40)
  sm['modelV2'].velocity.x = np.interp(ModelConstants.T_IDXS, [0, 10], [V_EGO, V_EGO - 6]).tolist()  # slowing for a stop
  sm['modelV2'].action.desiredAcceleration = -0.3
  source, a_target = _run(planner, sm, 1)
  assert source == LongitudinalPlanSource.e2e and a_target == pytest.approx(-0.3)


def test_toggle_off_keeps_the_model_pace():
  planner, sm = _planner_and_inputs(hold=False)
  source, a_target = _run(planner, sm, EXP_HOLD_FRAMES + 40)
  assert source == LongitudinalPlanSource.e2e and abs(a_target) < 0.05
