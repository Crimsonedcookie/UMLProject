"""
run_live_verify.py

Interactive verifier for multiaruco_homography_complete.py.

It reuses the same ArUco detection, internal-corner projection/refinement,
and homography validation used by the batch script.

Controls
--------
Space      pause / resume
N or Right next frame when paused 
P or Left  previous frame when paused
S          save the current verification mosaic
Q or Esc   quit
"""

from live_verify_homography import main

if __name__ == "__main__":
    main()
