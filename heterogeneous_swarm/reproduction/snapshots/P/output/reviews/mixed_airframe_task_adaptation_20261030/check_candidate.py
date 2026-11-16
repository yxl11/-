"""Reference-level calculation only; no individual vehicle dynamics or controller run."""
from pathlib import Path
import json, hashlib
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
CFG=json.loads((BASE/'candidate_protocol.json').read_text(encoding='utf-8'))

def reference(t, cruise=CFG['nominal_cruise_test_point_mps']):
    t=np.asarray(t,dtype=float); u=np.clip((t-15)/25,0,1)
    s=10*u**3-15*u**4+6*u**5
    d=(30*u**2-60*u**3+30*u**4)/25
    dd=(60*u-180*u**2+120*u**3)/25**2
    q0=np.stack([cruise*t,20*s,50+0*t],axis=-1)
    o=np.stack([18+6*s,12*s,0*t],axis=-1)
    v0=np.stack([cruise+0*t,20*d,0*t],axis=-1)
    od=np.stack([6*d,12*d,0*t],axis=-1)
    a0=np.stack([0*t,20*dd,0*t],axis=-1)
    odd=np.stack([6*dd,12*dd,0*t],axis=-1)
    return np.stack([q0+o,q0,q0-o],axis=-2),np.stack([v0+od,v0,v0-od],axis=-2),np.stack([a0+odd,a0,a0-odd],axis=-2)

def source_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    preserved=[ROOT/'output/reviews/ppt_execution_error_derivation_20261014/execution_error_derivation.tex',
               ROOT/'output/simulations/three_group_moving_comparison_20261023/src/mission.py',
               ROOT/'output/simulations/three_group_moving_comparison_20261023/protocol/frozen_mission.json']
    hashes={str(p.relative_to(ROOT)):source_hash(p) for p in preserved}
    t=np.linspace(0,60,60001);q,v,a=reference(t)
    speed=np.linalg.norm(v,axis=-1)
    cross=v[:,:,0]*a[:,:,1]-v[:,:,1]*a[:,:,0]
    heading=np.arctan2(v[:,:,1],v[:,:,0])
    heading_rate=cross/speed**2
    curvature=np.abs(cross)/speed**3
    lateral_accel=np.abs(cross)/speed
    bank=np.arctan2(lateral_accel,9.80665)
    cruise=CFG['nominal_cruise_test_point_mps']; pmax=1.875/25
    pstar=np.clip(6*cruise/100,0,pmax)
    exact_min=[cruise,cruise,float(np.hypot(cruise-6*pstar,8*pstar))]
    exact_max=[float(np.hypot(cruise+6*pmax,32*pmax)),float(np.hypot(cruise,20*pmax)),max(cruise,float(np.hypot(cruise-6*pmax,8*pmax)))]
    assert np.max(np.abs(speed.min(axis=0)-exact_min))<1e-7
    assert np.max(np.abs(speed.max(axis=0)-exact_max))<1e-7
    # Independent finite-difference check away from stage joins, no fitting.
    tt=np.array([3.,20.,27.,37.,51.]);eps=1e-4
    qm,vm,_=reference(tt-eps);qp,vp,_=reference(tt+eps);qc,vc,ac=reference(tt)
    qdiff=float(np.max(np.abs((qp-qm)/(2*eps)-vc)))
    vdiff=float(np.max(np.abs((vp-vm)/(2*eps)-ac)))
    assert qdiff<1e-7 and vdiff<1e-7
    assert np.all(v[:,:,0]>0)
    assert np.all(speed<=np.asarray(CFG['requested_caps_mps']))
    rows=[]
    for i in range(3):
        rows.append({'role':i+1,'reference_speed_min_mps':exact_min[i],'reference_speed_max_mps':exact_max[i],
          'speed_cap_mps':CFG['requested_caps_mps'][i],
          'upper_speed_margin_before_feedback_mps':CFG['requested_caps_mps'][i]-exact_max[i],
          'sampled_acceleration_max_mps2':float(np.linalg.norm(a[:,i],axis=-1).max()),
          'sampled_heading_max_deg':float(np.rad2deg(np.abs(heading[:,i])).max()),
          'sampled_heading_rate_max_deg_s':float(np.rad2deg(np.abs(heading_rate[:,i])).max()),
          'sampled_min_reference_turn_radius_m':float(1/curvature[:,i].max()),
          'sampled_equivalent_coordinated_bank_max_deg':float(np.rad2deg(bank[:,i]).max())})
    _,edge_v,edge_a=reference(np.array([0.,15.,40.,60.]))
    assert np.max(np.abs(edge_a))==0
    assert np.max(np.abs(edge_v[:,:,0]-cruise))==0
    assert np.max(np.abs(edge_v[:,:,1:]))==0
    result={'protocol_sha256':source_hash(BASE/'candidate_protocol.json'),
      'scope':'analytic prescribed role-center paths, not P closed-loop trajectories or aircraft certification',
      'cruise_mps':cruise,'duration_s':60,'forward_distance_m':cruise*60,
      'role_results':rows,'roles_endpoint_positions_m':q[-1].tolist(),
      'neighbor_target_center_distance_min_m':float(np.linalg.norm(q[:,0]-q[:,1],axis=-1).min()),
      'initial_and_final_reference_speed_mps':cruise,
      'reference_checks_passed':['strictly_positive_forward_speed','upper_speed_caps','C2_stage_joins','analytic_speed_extrema','independent_derivative_check'],
      'derivative_check_error':{'velocity_mps':qdiff,'acceleration_mps2':vdiff},
      'individual_vehicle_feasibility':'unverified','fixedwing_airspeed_floor':'unverified','safety_backup':'not_adapted',
      'preserved_input_sha256':hashes,'unchanged_inputs':all(source_hash(ROOT/p)==h for p,h in hashes.items())}
    assert result['unchanged_inputs']
    (BASE/'reference_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
