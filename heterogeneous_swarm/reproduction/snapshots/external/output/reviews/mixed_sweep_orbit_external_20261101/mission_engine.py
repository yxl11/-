"""Allocate the numerical accumulation allowance for the registered mission duration."""
import math
import numpy as np
from engine import make_engine


def make_mission_engine(counts, duration_s, mode):
    eng=make_engine(counts,mode)
    model=eng.model
    angular_change=model.H*np.sum(model.RATE[1:])
    worst_remainder=(3*model.H*(model.HI+model.H*model.RATE[0])*np.exp(angular_change)
                     *angular_change**9/math.factorial(9))
    numerical_per_step=model.PAD+worst_remainder
    reserve=(int(np.ceil(duration_s/model.H))+1)*numerical_per_step
    ns=eng.certify.__globals__
    eng.lifetime_reserve_m=max(ns['RESERVE'],float(reserve))
    ns['RESERVE']=eng.lifetime_reserve_m
    eng.max_numerical_per_step_m=float(numerical_per_step)
    return eng
