"""Check proposed geometry and analytic reference derivatives, not flight dynamics.

No flight controller connection and no flyable trajectory export are provided.
The 3 kg total suspended mass is an assumption pending rigging-mass confirmation.
"""
from pathlib import Path
import json
import numpy as np


def main():
    rope_length = 1.8
    anchor_radius = 0.25
    alpha = np.deg2rad(20.0)
    mass = 3.0
    gravity = 9.81
    duration = 40.0
    u = np.linspace(0.0, 1.0, 40001)
    s = 35*u**4 - 84*u**5 + 70*u**6 - 20*u**7
    sd = (140*u**3 - 420*u**4 + 420*u**5 - 140*u**6) / duration
    sdd = (420*u**2 - 1680*u**3 + 2100*u**4 - 840*u**5) / duration**2
    sddd = (840*u - 5040*u**2 + 8400*u**3 - 4200*u**4) / duration**3
    c = 2*np.pi
    velocity = np.column_stack((6*sd, c*np.cos(c*s)*sd))
    acceleration = np.column_stack((6*sdd, -c*c*np.sin(c*s)*sd**2 + c*np.cos(c*s)*sdd))
    jerk = np.column_stack((6*sddd, -c**3*np.cos(c*s)*sd**3
                            - 3*c*c*np.sin(c*s)*sd*sdd + c*np.cos(c*s)*sddd))
    radius = anchor_radius + rope_length*np.sin(alpha)
    tension = mass*gravity/(3*np.cos(alpha))
    endpoints = np.array([s[0], s[-1], sd[0], sd[-1],
                          sdd[0], sdd[-1], sddd[0], sddd[-1]])
    expected = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    assert np.allclose(endpoints, expected, atol=1e-10)
    assert np.all(sd >= -1e-10)
    result = {
        "status": "reference_design_check_only",
        "coupled_dynamics_simulated": False,
        "px4_tested": False,
        "flight_ready": False,
        "assumptions": {
            "total_suspended_mass_kg_pending_rigging_confirmation": mass,
            "rope_length_m": rope_length,
            "anchor_radius_m": anchor_radius,
            "hover_rope_angle_from_vertical_deg": 20,
            "outbound_duration_s": duration,
            "full_suspended_sequence_duration_s": 104,
        },
        "geometry": {
            "uav_attachment_radius_m": float(radius),
            "attachment_separation_m": float(np.sqrt(3)*radius),
            "vertical_attachment_offset_m": float(rope_length*np.cos(alpha)),
        },
        "static_load_ignoring_rope_mass": {
            "tension_per_rope_N": float(tension),
            "horizontal_force_per_uav_N": float(tension*np.sin(alpha)),
            "vertical_force_per_uav_N": float(tension*np.cos(alpha)),
        },
        "analytic_derivative_sampling": {
            "sample_count": len(u),
            "speed_peak_m_s": float(np.linalg.norm(velocity, axis=1).max()),
            "acceleration_peak_m_s2": float(np.linalg.norm(acceleration, axis=1).max()),
            "jerk_peak_m_s3": float(np.linalg.norm(jerk, axis=1).max()),
            "endpoint_derivative_conditions_passed": True,
        },
    }
    out = Path(__file__).resolve().parents[1] / "analysis" / "reference_design_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
