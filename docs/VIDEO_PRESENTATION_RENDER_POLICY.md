# Video presentation render policy

For this repository, all HD and UHD presentation outputs must use the shared deterministic renderer:

`tools/render_deterministic_master.py`

The approved treatment is:

`light hqdn3d cleanup -> Lanczos resize -> CAS 0.22 -> 30 fps -> H.264 high profile`

Approved UHD encoding defaults are CRF 16, maxrate 36 Mbps, bufsize 72 Mbps, AAC 192 kbps, faststart. Approved HD output uses the same image treatment at 1280x720 with an appropriately lower bitrate ceiling.

Computer-vision analysis, tracking, player identity, court calibration and geometry must be performed on the best available native official NBA source before any presentation resize. Presentation resizing is a delivery step only and must never be described as recovering new source detail.

Do not add a second presentation-enhancement path. New workflows should call the shared renderer rather than duplicating an FFmpeg filter chain inline. Existing operational files that render HD/UHD should be migrated to this helper when touched.
