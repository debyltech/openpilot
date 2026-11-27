#!/usr/bin/env python3
import math
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc,
  LongitudinalPlanSource,
  get_safe_obstacle_distance,
  get_stopped_equivalence_factor,
  get_T_FOLLOW,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MIN = -0.5
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
NAP_FOLLOW_DISTANCE_RANGE = range(1, 8)
# A new follow distance applies only once the menu selection has held this long, so cycling through
# the settings while driving doesn't jerk the car between gaps. The param is read once a second,
# so a change lands 4-5 s after the last tap.
NAP_FOLLOW_SETTLE_FRAMES = int(4.0 / DT_MDL)


# Pre-AP follow-mode accel cap: imported lazily to avoid circular deps
_preap_follow_cache = None
def _get_preap_follow_limit(v_ego):
  global _preap_follow_cache
  if _preap_follow_cache is None:
    try:
      from opendbc.car.tesla.preap.constants import ACCEL_PREAP_BP, ACCEL_PREAP_FOLLOW
      _preap_follow_cache = (ACCEL_PREAP_BP, ACCEL_PREAP_FOLLOW)
    except ImportError:
      _preap_follow_cache = (None, None)
  bp, v = _preap_follow_cache
  if bp is None:
    return None
  return float(np.interp(v_ego, bp, v))


def get_preap_follow_cap_strength(v_ego, lead_distance, lead_speed, t_follow):
  lead_obstacle_distance = lead_distance + get_stopped_equivalence_factor(max(lead_speed, 0.0))
  safe_obstacle_distance = get_safe_obstacle_distance(v_ego, t_follow)
  equivalent_ratio = lead_obstacle_distance / max(safe_obstacle_distance, 1.0)
  return float(np.clip(1.0 - (equivalent_ratio - 1.2) / 0.3, 0.0, 1.0))

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, angle_steers, CP, dt, accel_coast, allow_throttle):
  max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego)

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
    if not allow_throttle:
      clipped_accel_coast = max(accel_coast, ACCEL_MIN)
      coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
      max_accel = min(max_accel, coast_limit)

  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel)
  j_cruise = np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL, params=None):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True
    self.accel_boost = AccelBoost()

    self._is_preap = (CP.brand == "tesla" and CP.carFingerprint == "TESLA_MODEL_S_PREAP"
                      and CP.openpilotLongitudinalControl and not CP.pcmCruise)
    self._params = Params() if params is None else params
    self.nap_follow_dist = self._params.get("NAPFollowDistance", return_default=True) if self._is_preap else None
    self.nap_adaptive_accel = self._params.get_bool("NAPAdaptiveAccel") if self._is_preap else False
    self._nap_follow_setting, self._nap_follow_setting_frame = self.nap_follow_dist, 0
    self.active_nap_follow_dist = None
    self.t_follow = get_T_FOLLOW()
    self._frame = 0

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    self._frame += 1
    if self._is_preap and self._frame % 20 == 0:
      setting = self._params.get("NAPFollowDistance", return_default=True)
      if setting != self._nap_follow_setting:
        self._nap_follow_setting, self._nap_follow_setting_frame = setting, self._frame
      if self._frame - self._nap_follow_setting_frame >= NAP_FOLLOW_SETTLE_FRAMES:
        self.nap_follow_dist = setting
      self.nap_adaptive_accel = self._params.get_bool("NAPAdaptiveAccel")

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    # VDAS takes acceleration targets literally, so the model's negative coast
    # ceiling commands regen even when the planner is trying to hold cruise speed.
    self.allow_throttle = self._is_preap or throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    self.active_nap_follow_dist = self.nap_follow_dist if self._is_preap and self.nap_follow_dist in NAP_FOLLOW_DISTANCE_RANGE else None
    self.t_follow = get_T_FOLLOW(sm['selfdriveState'].personality, self.active_nap_follow_dist)
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality, t_follow=self.t_follow)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    self.a_cruise = get_cruise_accel(sm['selfdriveState'].experimentalMode, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.allow_throttle)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    output_a_target_e2e = self.accel_boost.update(sm, output_a_target_e2e, output_a_target_mpc, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if sm['selfdriveState'].experimentalMode:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    # Pre-AP adaptive accel: cap positive accel only when the lead's obstacle-equivalent distance is
    # close (full cap below 1.2x the safe distance, none above 1.5x) to stop overshoot -> regen ->
    # overshoot oscillation. It only ever lowers the target.
    lead = sm['radarState'].leadOne
    if self._is_preap and self.nap_adaptive_accel and lead.present and self.output_a_target > 0.0:
      follow_limit = _get_preap_follow_limit(v_ego)
      if follow_limit is not None:
        cap_strength = get_preap_follow_cap_strength(v_ego, lead.dRel, lead.vLead, self.t_follow)
        if cap_strength > 0:
          cap = self.output_a_target * (1.0 - cap_strength) + follow_limit * cap_strength
          self.output_a_target = min(self.output_a_target, cap)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.accelBoost = float(self.accel_boost.total_boost)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)
    longitudinalPlan.napFollowDistance = self.active_nap_follow_dist or 0
    longitudinalPlan.tFollow = self.t_follow

    pm.send('longitudinalPlan', plan_send)
