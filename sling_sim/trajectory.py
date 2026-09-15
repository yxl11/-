from dataclasses import dataclass
import numpy as np


@dataclass
class Reference:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    phase: str


class TransportTrajectory:
    def __init__(self, config, kind="s_curve"):
        self.leg = config["leg_duration"]
        self.hold = config["hold_duration"]
        self.height = config["payload_com_height"]
        self.kind = kind
        self.duration = 2*self.leg + 3*self.hold

    def at(self, time):
        t, h, d = float(time), self.hold, self.leg
        s = sd = sdd = sddd = 0.0
        phase = "start_hold"
        if self.kind == "hover":
            pass
        elif h <= t < h+d:
            s, sd, sdd, sddd = self._time_law((t-h)/d, d)
            phase = "outbound"
        elif h+d <= t < 2*h+d:
            s, phase = 1.0, "end_hold"
        elif 2*h+d <= t < 2*h+2*d:
            v = self._time_law((t-2*h-d)/d, d)
            s, sd, sdd, sddd = 1-v[0], -v[1], -v[2], -v[3]
            phase = "return"
        elif t >= 2*h+2*d:
            phase = "final_hold"
        c = 2*np.pi
        amplitude = 0.0 if self.kind == "line" else 1.0
        p = np.array([-3+6*s, amplitude*np.sin(c*s), self.height])
        v = np.array([6*sd, amplitude*c*np.cos(c*s)*sd, 0.0])
        a = np.array([6*sdd, amplitude*(-c*c*np.sin(c*s)*sd**2+c*np.cos(c*s)*sdd), 0.0])
        j = np.array([6*sddd, amplitude*(-c**3*np.cos(c*s)*sd**3
                       -3*c*c*np.sin(c*s)*sd*sdd+c*np.cos(c*s)*sddd), 0.0])
        return Reference(p, v, a, j, phase)

    @staticmethod
    def _time_law(u, d):
        return (35*u**4-84*u**5+70*u**6-20*u**7,
                (140*u**3-420*u**4+420*u**5-140*u**6)/d,
                (420*u**2-1680*u**3+2100*u**4-840*u**5)/d**2,
                (840*u-5040*u**2+8400*u**3-4200*u**4)/d**3)

