import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.controls.radard import KalmanParams, RADAR_TO_CAMERA, RadarD, Track, VisionRangeRate


MAX_RADAR_MEASUREMENT_AGE = 0.5


class RadarScenario:
  def __init__(self, v_ego: float = 20.0):
    services = ["modelV2", "carState", "radarTracks"]
    self.sm = messaging.SubMaster(services, ignore_alive=services, ignore_avg_freq=services)
    self.radar = RadarD()
    self.v_ego = v_ego
    self.frame = 0

  def step(self, time_s: float, vision_d_rel: float, radar_points: list[tuple[int, float, float, float]] | None = None,
           vision_v: float | None = None):
    messages = [self._model_message(time_s, vision_d_rel, vision_v)]
    if self.frame == 0:
      messages.append(self._car_state_message(time_s))
    if radar_points is not None:
      messages.append(self._radar_message(time_s, radar_points))

    self.sm.update_msgs(time_s, [message.as_reader() for message in messages])
    self.radar.update(self.sm, self.sm["radarTracks"])
    self.frame += 1
    return self.radar.radar_state.leadOne

  def _model_message(self, time_s: float, vision_d_rel: float, vision_v: float | None):
    message = messaging.new_message("modelV2")
    message.logMonoTime = int(time_s * 1e9)
    message.modelV2.velocity.x = [self.v_ego]
    leads = message.modelV2.init("leadsV3", 2)
    for lead in leads:
      lead.prob = 0.9
      lead.x = [vision_d_rel + RADAR_TO_CAMERA]
      lead.xStd = [3.0]
      lead.y = [0.0]
      lead.yStd = [1.0]
      lead.v = [self.v_ego if vision_v is None else vision_v]
      lead.vStd = [2.0]
      lead.a = [0.0]
    return message

  def _car_state_message(self, time_s: float):
    message = messaging.new_message("carState")
    message.logMonoTime = int(time_s * 1e9)
    message.carState.vEgo = self.v_ego
    return message

  @staticmethod
  def _radar_message(time_s: float, radar_points: list[tuple[int, float, float, float]]):
    message = messaging.new_message("radarTracks")
    message.logMonoTime = int(time_s * 1e9)
    points = message.radarTracks.init("points", len(radar_points))
    for point, (track_id, d_rel, y_rel, v_rel) in zip(points, radar_points, strict=True):
      point.trackId = track_id
      point.dRel = d_rel
      point.yRel = y_rel
      point.vRel = v_rel
    return message


def test_radar_silence_falls_back_to_current_vision_lead():
  scenario = RadarScenario()
  lead = scenario.step(1.0, vision_d_rel=30.0, radar_points=[(7, 30.0, 0.0, 0.0)])
  assert lead.radar

  model_dt = 0.05
  expiration_frame = int(MAX_RADAR_MEASUREMENT_AGE / model_dt) + 1
  for frame in range(1, expiration_frame):
    lead = scenario.step(1.0 + frame * model_dt, vision_d_rel=36.0)
  assert lead.radar

  lead = scenario.step(1.0 + expiration_frame * model_dt, vision_d_rel=36.0)

  assert not lead.radar
  assert lead.dRel == pytest.approx(36.0)


def test_association_rejects_distance_outlier():
  scenario = RadarScenario()
  lead = scenario.step(1.0, vision_d_rel=100.0, radar_points=[(7, 120.0, 0.0, 0.0)])

  assert not lead.radar
  assert lead.dRel == pytest.approx(100.0)


def test_association_rejects_velocity_outlier():
  scenario = RadarScenario()
  lead = scenario.step(1.0, vision_d_rel=50.0, radar_points=[(7, 50.0, 0.0, 15.0)])

  assert not lead.radar
  assert lead.vLead == pytest.approx(20.0)


def test_association_requires_minimum_score():
  scenario = RadarScenario()
  # Each residual remains inside its independent 3-sigma gate, while their
  # combined likelihood is below the score floor.
  lead = scenario.step(1.0, vision_d_rel=50.0, radar_points=[(7, 58.0, 2.9, 5.5)])

  assert not lead.radar


def test_association_retains_incumbent_until_challenger_wins():
  scenario = RadarScenario()
  two_tracks = [(81, 70.0, 0.0, 0.0), (82, 72.0, 0.0, 0.0)]
  vision_distances = [70.8, 71.2, 70.8, 71.2, 70.8, 71.2, 72.0, 70.8]

  selected_ids = [
    scenario.step(1.0 + frame * 0.1, vision_d_rel=vision_d_rel, radar_points=two_tracks).radarTrackId
    for frame, vision_d_rel in enumerate(vision_distances)
  ]
  selected_ids.append(scenario.step(1.8, vision_d_rel=70.0, radar_points=[two_tracks[0]]).radarTrackId)

  assert selected_ids == [81, 81, 81, 81, 81, 81, 82, 82, 81]


def test_unmeasured_track_does_not_update_kalman_state():
  track = Track(identifier=7, v_lead=20.0, kalman_params=KalmanParams(0.1))
  track.update(d_rel=30.0, y_rel=0.0, v_rel=0.0, v_lead=20.0, measured=True)
  track.update(d_rel=30.0, y_rel=0.0, v_rel=1.0, v_lead=21.0, measured=True)
  measured_state = (track.vLeadK, track.aLeadK)

  track.update(d_rel=30.0, y_rel=0.0, v_rel=10.0, v_lead=30.0, measured=False)

  assert (track.vLeadK, track.aLeadK) == pytest.approx(measured_state)


def test_kalman_uses_observed_radar_interval():
  scenario = RadarScenario()
  scenario.step(1.0, vision_d_rel=30.0, radar_points=[(7, 30.0, 0.0, 0.0)])

  scenario.step(1.08, vision_d_rel=30.0, radar_points=[(7, 30.0, 0.0, 0.0)])

  assert scenario.radar.tracks[7].K_A[0][1] == pytest.approx(0.08)


def test_vision_range_rate_catches_closing_the_model_understates():
  # Real closing 2 m/s; the model's vRel says 0.5 m/s, as on the 2026-11-26 pre-AP drive.
  f = VisionRangeRate()
  rate = 0.0
  for i in range(40):  # 2 s at 20 Hz
    rate = f.update(60.0 - 2.0 * i * 0.05, -0.5)
  assert rate < -1.3  # steady blend is ~5/8 distance trend, 3/8 model


def test_vision_range_rate_stays_calm_on_distance_noise():
  import numpy as np
  rng = np.random.default_rng(0)
  f = VisionRangeRate()
  rates = [f.update(35.0 + rng.normal(0, 3.0), 0.0) for _ in range(400)]
  assert np.std(rates[40:]) < 0.6
  assert abs(np.mean(rates[40:])) < 0.3


def test_vision_range_rate_resets_on_a_new_car():
  f = VisionRangeRate()
  for _ in range(40):
    f.update(50.0, 0.0)
  for _ in range(VisionRangeRate.JUMP_CONFIRM - 1):
    assert f.update(30.0, -1.0) == -1.0  # glitch so far: the model passes through
  assert abs(f.update(30.0, -1.0)) < 0.1  # confirmed new car: re-anchored, no huge range rate
  assert f.d == 30.0


def test_vision_range_rate_ignores_a_one_frame_distance_glitch():
  f = VisionRangeRate()
  for _ in range(40):
    f.update(35.0, 0.0)
  f.update(25.0, -4.7)  # one bad frame, as seen at t=65 s on the 2026-11-26 drive
  assert abs(f.update(35.0, 0.0)) < 0.3


def test_vision_lead_accel_follows_a_lead_speeding_up():
  # Lead holds 20 m/s 35 m ahead, then speeds up at 1 m/s^2; the model reports its speed but a ~0 accel.
  f = VisionRangeRate()
  v_ego, d, v_lead = 20.0, 35.0, 20.0
  for _ in range(40):
    f.update(d, v_lead - v_ego)
    assert abs(f.lead_accel(v_ego)) < 0.05
  for _ in range(60):  # 3 s
    v_lead += 1.0 * 0.05
    d += (v_lead - v_ego) * 0.05
    f.update(d, v_lead - v_ego)
    a = f.lead_accel(v_ego)
  assert 0.6 < a < 1.3


def test_vision_lead_accel_ignores_a_one_frame_glitch_and_resets():
  f = VisionRangeRate()
  for _ in range(40):
    f.update(35.0, 0.0)
    f.lead_accel(20.0)
  f.update(25.0, -4.7)  # one bad frame passes the model's value through, but must not kick the accel
  assert abs(f.lead_accel(20.0)) < 0.1
  f.reset()
  assert f.a_lead == 0.0
