"""Road speed-limit sign for the comma 4 onroad view (top-right), from mapd's OSM data.

The limit comes from mapd (github.com/pfeiferj/openpilot-mapd, MIT), which publishes mapdOut from
offline OpenStreetMap tiles; map data (c) OpenStreetMap contributors (ODbL). Display only: nothing
here feeds the planner. Sign shapes after sunnypilot's selfdrive/ui/sunnypilot/onroad/speed_limit.py (MIT).
"""
import pyray as rl
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

MAX_VALID_LIMIT = 40.0  # m/s; mapd encodes an explicit "no limit" as 145 km/h
HOLD_S = 5.0  # keep the last limit through short map/GPS dropouts
OVERSPEED_FRACTION = 0.10
OVERSPEED_MIN = {False: 5, True: 8}  # mph, km/h

SIGN_W, SIGN_H = 66, 84  # MUTCD (imperial)
RING_R = 40  # Vienna (metric)
INSET_X, INSET_Y = 14, 12

RED = rl.Color(235, 32, 32, 255)


def overspeed(speed: float, limit: float, is_metric: bool) -> bool:
  """Speed and limit in display units. Over by more than 10 % of the limit, at least 5 mph / 8 km/h."""
  return limit > 0 and speed > limit + max(OVERSPEED_FRACTION * limit, OVERSPEED_MIN[is_metric])


class SpeedLimitSign(Widget):
  def __init__(self):
    super().__init__()
    self._limit = 0  # display units, rounded
    self._last_valid_t = -HOLD_S
    self._speed = 0.0
    self._visible = False
    self._alpha = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)

  def _update_state(self) -> None:
    sm = ui_state.sm
    conv = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    now = rl.get_time()
    if sm.recv_frame['mapdOut'] >= ui_state.started_frame and sm.alive['mapdOut']:
      md = sm['mapdOut']
      if md.tileLoaded and 0 < md.speedLimit < MAX_VALID_LIMIT:
        self._limit = round(md.speedLimit * conv)
        self._last_valid_t = now
    cs = sm['carState']
    v_ego = cs.vEgoCluster if cs.vEgoCluster != 0.0 else cs.vEgo
    self._speed = max(0.0, v_ego * conv)
    self._visible = ui_state.speed_limit_sign and now - self._last_valid_t < HOLD_S

  def _render(self, rect: rl.Rectangle) -> None:
    alpha = self._alpha.update(1.0 if self._visible else 0.0)
    if alpha < 1e-2:
      return
    over = ui_state.speed_limit_overspeed and overspeed(self._speed, self._limit, ui_state.is_metric)
    right = rect.x + rect.width - INSET_X
    if ui_state.is_metric:
      self._draw_vienna(rl.Vector2(right - RING_R, rect.y + INSET_Y + RING_R), over, alpha)
    else:
      self._draw_mutcd(rl.Rectangle(right - SIGN_W, rect.y + INSET_Y, SIGN_W, SIGN_H), over, alpha)

  def _text(self, font, text: str, size: int, cx: float, cy: float, color: rl.Color) -> None:
    sz = measure_text_cached(font, text, size)
    rl.draw_text_ex(font, text, rl.Vector2(cx - sz.x / 2, cy - sz.y / 2), size, 0, color)

  def _draw_mutcd(self, r: rl.Rectangle, over: bool, alpha: float) -> None:
    white, black = rl.color_alpha(rl.WHITE, alpha), rl.color_alpha(rl.BLACK, alpha)
    accent = rl.color_alpha(RED, alpha) if over else black
    rl.draw_rectangle_rounded(r, 0.2, 10, white)
    rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(r.x + 4, r.y + 4, r.width - 8, r.height - 8), 0.2, 10, 4 if over else 2, accent)
    cx = r.x + r.width / 2
    self._text(self._font_semi, "SPEED", 15, cx, r.y + 17, black)
    self._text(self._font_semi, "LIMIT", 15, cx, r.y + 31, black)
    self._text(self._font_bold, str(self._limit), 40 if self._limit < 100 else 32, cx, r.y + 60, accent)

  def _draw_vienna(self, c: rl.Vector2, over: bool, alpha: float) -> None:
    rl.draw_circle_v(c, RING_R, rl.color_alpha(rl.WHITE, alpha))
    rl.draw_ring(c, RING_R * 0.78, RING_R, 0, 360, 36, rl.color_alpha(RED, alpha))
    color = rl.color_alpha(RED if over else rl.BLACK, alpha)
    self._text(self._font_bold, str(self._limit), 36 if self._limit < 100 else 28, c.x, c.y, color)
