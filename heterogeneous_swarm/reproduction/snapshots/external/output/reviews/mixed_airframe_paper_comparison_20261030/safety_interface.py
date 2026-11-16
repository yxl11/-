"""Model-level candidate checking and persistent certified backup suffix."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
PRO=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
OLD=BASE.parent/'fixedwing_consistent_predictor_20261030'
sys.path.insert(0,str(OLD))
import predictor as model
assert hashlib.sha256((OLD/'predictor.py').read_bytes()).hexdigest()==PRO['model_sha256']
assert hashlib.sha256((OLD/'protocol.json').read_bytes()).hexdigest()==PRO['model_protocol_sha256']
RESERVE=PRO['extra_lifetime_position_allowance_m']
COMMON=np.broadcast_to(model.B,(80,3)).copy()

def advance_model(r,v,c):
    r=np.asarray(r,dtype=float);v=np.asarray(v,dtype=float);c=np.asarray(c,dtype=float)
    seg=model.make_segment(v,c)
    return r+model.displacement(seg,model.H),seg['vn'],seg

class Certificate:
    def __init__(self,states_r,states_v,target,evidence):
        self.R=np.array(states_r);self.V=np.array(states_v)
        self.target=target.copy();self.evidence=evidence
        self.cursor=0;self.generation=None;self.created_step=None
        self.expected_r=self.R[0].copy();self.expected_v=self.V[0].copy()

    def assert_state(self,r,v):
        if not np.array_equal(r,self.expected_r) or not np.array_equal(v,self.expected_v):
            dr=float(np.max(np.abs(r-self.expected_r)));dv=float(np.max(np.abs(v-self.expected_v)))
            raise RuntimeError(f'certificate_state_mismatch: position={dr}, velocity={dv}; no disturbance tube assumed')

    def command(self):return self.target.copy() if self.cursor==0 else COMMON.copy()

    def expected_next(self):
        if self.cursor+1<len(self.R):return self.R[self.cursor+1],self.V[self.cursor+1]
        rn,vn,_=advance_model(self.expected_r,self.expected_v,COMMON)
        return rn,vn

    def commit(self,r,v):
        er,ev=self.expected_next()
        if not np.array_equal(r,er) or not np.array_equal(v,ev):
            raise RuntimeError('predicted_execution_mismatch: inherited certificate cannot be used')
        self.cursor+=1;self.expected_r=r.copy();self.expected_v=v.copy()

    def summary(self):
        payload=np.concatenate([self.R.ravel(),self.V.ravel(),self.target.ravel()]).tobytes()
        return {**self.evidence,'generation':self.generation,'created_step':self.created_step,
          'consumed_segments':self.cursor,'saved_prefix_states':len(self.R),'prefix_sha256':hashlib.sha256(payload).hexdigest()}

def certify(r0,v0,c):
    """Whole-path certificate with mathematically sufficient early tail exit."""
    mapped,mapping=model.map_command(c)
    r=np.asarray(r0,dtype=float).copy();v=np.asarray(v0,dtype=float).copy()
    R=[r.copy()];V=[v.copy()];errors=np.full(len(r),RESERVE)
    lower=np.inf;samplemin=np.inf;maxsteps=1+round(model.PRO['explicit_backup_window_s']/model.H)
    minimum_pair=None
    for k in range(maxsteps):
        seg=model.make_segment(v,mapped if k==0 else COMMON)
        rn,errors,check=model.check_segment(r,seg,errors)
        if check['lower_m']<lower:lower=check['lower_m'];minimum_pair=check['lower_pair']
        samplemin=min(samplemin,check['sample_min_m'])
        R.append(rn.copy());V.append(seg['vn'].copy());r=rn;v=seg['vn']
        if check['lower_m']<model.SAFE+model.TOL:
            return None,{'status':'sampled_violation' if check['sampled_violation'] else 'unresolved_segment',
              'prediction_steps':k+1,'minimum_prefix_lower_m':lower,'minimum_sample_m':samplemin,
              'limiting_pair':minimum_pair,'mapping':mapping}
        rest=model.tail_path(v);i,j=model.PAIR
        tail=np.linalg.norm(r[i]-r[j],axis=1)-rest[i]-rest[j]-errors[i]-errors[j]
        p=int(np.argmin(tail));tailmin=float(tail[p])
        if tailmin>=model.SAFE+model.TOL:
            evidence={'status':'certified','prediction_steps':k+1,'minimum_prefix_lower_m':lower,
              'minimum_sample_m':samplemin,'tail_lower_m':tailmin,'certificate_margin_m':min(lower,tailmin)-model.SAFE,
              'tail_pair':[int(i[p]),int(j[p])],'tail_path_max_m':float(rest.max()),
              'position_allowance_max_m':float(errors.max()),'mapping':mapping}
            return Certificate(R,V,mapped,evidence),evidence
    return None,{'status':'unresolved_tail','prediction_steps':maxsteps,'minimum_prefix_lower_m':lower,
      'minimum_sample_m':samplemin,'tail_lower_m':tailmin,'mapping':mapping}

class RetainedBackupInterface:
    def __init__(self,r,v):
        initial=model.initial_check(r,v)
        if not initial['eligible']:raise RuntimeError('initial_ineligible: '+str(initial['reasons']))
        cert,proof=certify(r,v,COMMON)
        if cert is None:raise RuntimeError('initial_backup_uncertified: '+proof['status'])
        self.cert=cert;self.generation=0;self.cert.generation=0;self.cert.created_step=-1
        self.initial=initial;self.bootstrap=proof;self.ledger=[self.cert.summary()]

    def select(self,step,r,v,held_reference,force_reject=False):
        self.cert.assert_state(r,v)
        if force_reject:trial=None;proof={'status':'forced_unit_test_rejection','prediction_steps':0}
        else:trial,proof=certify(r,v,held_reference)
        replaced=trial is not None
        if replaced:
            self.generation+=1;self.cert=trial;self.cert.generation=self.generation;self.cert.created_step=step
            self.ledger.append(self.cert.summary())
        out={'status':'nominal_candidate' if replaced else 'retained_backup','trial':proof,
          'certificate_generation':self.cert.generation,'certificate_age_steps':self.cert.cursor,
          'inherited_margin_m':self.cert.evidence['certificate_margin_m']}
        return self.cert.command(),out

    def commit(self,r,v):self.cert.commit(r,v)

    def save_current(self,path):
        np.savez_compressed(path,R=self.cert.R,V=self.cert.V,target=self.cert.target,
          expected_r=self.cert.expected_r,expected_v=self.cert.expected_v,cursor=self.cert.cursor,
          generation=self.cert.generation,created_step=self.cert.created_step)
