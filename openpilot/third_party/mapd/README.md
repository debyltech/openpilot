# mapd (OSM speed limits)

- Source: https://github.com/pfeiferj/openpilot-mapd, release v2.4.0, asset `mapd` (aarch64, static)
- sha256: `0bc3564ec634486d11e95b76e884fd37b783c66933405cec1bd59f179fa09e94`
- License: MIT (see `LICENSE`). Map data © OpenStreetMap contributors (ODbL), served by map-data.pfeifer.dev.

Used display-only: `mapdOut.speedLimit` drives the comma 4 speed-limit sign. All of mapd's speed-control
features are off by default (its `settings/defaults.json`), and nothing in the planner reads `mapdOut`.
Runs while `NAPSpeedLimitSign` is on (default off: OSM limits are often wrong on rural roads), offroad too.

Maps are downloaded once per region over Wi-Fi, into `/data/media/0/osm`. Over SSH, with mapd running:

```
cd /data/openpilot/openpilot/third_party/mapd && ./mapd interactive
```

or non-interactively, by sending `mapdIn` (`type: download`, `str: "us_state.MI"`, comma-separated for several).
