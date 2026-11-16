"""The frozen P controller and original physical metrics for an arbitrary three-role roster."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
CAM = ROOT / 'output/simulations/three_group_moving_comparison_20261023'
OLD_RUN = BASE.parent / 'fixedwing_local_escape_20261031/runs/diagnostic/P_escape_s1008'

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_frozen():
    # harness historically imports "mission" by its short name. The working
    # mission here has a different API, so bind the old dependency only during
    # that import. Both controllers and the density metric remain read-only.
    old_mission = load_module('p_original_mission',CAM/'src/mission.py')
    src = str(CAM/'src')
    sys.path.insert(0,src)
    previous = sys.modules.get('mission')
    sys.modules['mission'] = old_mission
    try:
        h = load_module('original_p_harness',CAM/'src/harness.py')
    finally:
        if previous is None:
            del sys.modules['mission']
        else:
            sys.modules['mission'] = previous
        sys.path.remove(src)
    if Path(h.multigroup.__file__).resolve() != (h.FROZEN/'multigroup.py').resolve():
        raise RuntimeError('Wrong frozen multigroup module')
    if Path(h.point_controller.__file__).resolve() != (h.VENDOR/'projected_mobility_paired_20261014/src/point_controller.py').resolve():
        raise RuntimeError('Wrong frozen point-gradient module')
    h.verify_protocol()
    density = load_module('original_density_metrics',CAM/'src/external_density.py')
    return old_mission,h,density

def bind_reference(h, reference):
    h.multigroup.references = reference
    h.point_controller.references = reference
    # The analytic overlap kernel takes q explicitly in the online path.
    # Bind the same q to its offline diagnostics too.
    h.point_controller._graph.references = reference

class TaskFeedback:
    def __init__(self,h,density,cfg,reference):
        self.cfg, self.reference = cfg, reference
        bind_reference(h,reference)
        self.controller = h.PController(cfg)
        self.metrics = h.multigroup.Metrics(cfg)
        self.density = density.relative_density_l2
        self.groups = self.controller.groups
        self.roles = self.controller.roles
        expected=np.repeat(np.arange(3),cfg['counts'])
        if not np.array_equal(self.roles,expected) or not np.array_equal(self.metrics.model.roles,expected):
            raise RuntimeError('Controller and metric memberships must agree')
        self.sig=np.asarray(cfg['sigma_m']);self.beta=np.asarray(cfg['bandwidth_m'])

    def fast(self,r,t):
        q,qd=self.reference(t,self.cfg)
        centers=self.controller.means(r)
        errors=centers-q
        fraction=[np.mean(np.sum(((r[g]-q[i])/self.sig[i])**2,axis=1)<=4) for i,g in enumerate(self.groups)]
        return {'reference':q,'reference_velocity':qd,'centers':centers,
                'role_center_error':np.linalg.norm(errors,axis=1),
                'edge_error':np.linalg.norm(errors[:-1]-errors[1:],axis=1),
                'region_by_role':np.asarray(fraction)}

    def slow(self,r,t):
        q,_=self.reference(t,self.cfg)
        shape=self.metrics.calculate(r,t)['shape_proxy']
        rho=[self.density(r[g],q[i],self.sig[i],self.beta[i]) for i,g in enumerate(self.groups)]
        nearest=[]; horizontal_nearest=[]
        for g in self.groups:
            d=squareform(pdist(r[g]))
            np.fill_diagonal(d,np.inf)
            nearest.append(np.min(d,axis=1))
            horizontal=squareform(pdist(r[g,:2]))
            np.fill_diagonal(horizontal,np.inf)
            horizontal_nearest.append(np.min(horizontal,axis=1))
        return {'time':float(t),'relative_density_l2_by_role':np.asarray(rho),
                'shape_by_role':shape,'intrarole_nn_m':np.concatenate(nearest),
                'intrarole_horizontal_nn_m':np.concatenate(horizontal_nearest)}

def source_files():
    old_ref=BASE.parent/'mixed_airframe_task_adaptation_20261030'
    return [CAM/'src/harness.py',CAM/'src/mission.py',CAM/'src/external_density.py',
            CAM/'protocol/frozen_mission.json',CAM/'protocol/frozen_mission.sha256',
            CAM/'vendor/projected_mobility_paired_20261014/src/projected_controller.py',
            CAM/'vendor/projected_mobility_paired_20261014/src/point_controller.py',
            CAM/'vendor/unified_density_cooperation_20261014/src/task_density_controller.py',
            CAM/'vendor/multigroup_closed_loop_20261012/runs/four_group_s401/source_snapshot/multigroup.py',
            old_ref/'check_candidate.py',old_ref/'candidate_protocol.json',
            OLD_RUN/'task_history.npz',OLD_RUN/'slow_metrics.npz',OLD_RUN/'fast_metrics.npz']

def old_identity(old_mission,h,density):
    old_ref=load_module('p_demonstration_archived_ref',BASE.parent/'mixed_airframe_task_adaptation_20261030/check_candidate.py')
    def reference(t,cfg=None):
        q,qd,_=old_ref.reference(t)
        return q,qd
    cfg=old_mission.config()
    cfg['vmax_mps']=[15.,12.,10.]
    old=TaskFeedback(h,density,cfg,reference)
    archive=OLD_RUN.parents[2]
    manifest=json.loads((archive/'artifact_manifest.json').read_text(encoding='utf8'))['files_sha256']
    for name in ('task_history.npz','fast_metrics.npz','slow_metrics.npz'):
        path=OLD_RUN/name
        if sha(path)!=manifest[path.relative_to(archive).as_posix()]:
            raise RuntimeError(f'Original evidence changed: {name}')
    checks=[]
    with np.load(OLD_RUN/'task_history.npz') as h0, np.load(OLD_RUN/'fast_metrics.npz') as f0, np.load(OLD_RUN/'slow_metrics.npz') as s0:
        for index in [0,275,595]:
            r,v,t=h0['positions'][index],h0['velocities'][index],float(h0['time'][index])
            a=old.controller.command(t,r,v)
            fast=old.fast(r,t);slow=old.slow(r,t)
            fi=round(t/.02);si=round(t/.5)
            fields={'P_raw':np.array_equal(a,h0['raw_reference'][index]),
                    'reference':np.array_equal(fast['reference'],f0['reference'][fi]),
                    'centers':np.array_equal(fast['centers'],f0['centers'][fi]),
                    'edges':np.array_equal(fast['edge_error'],f0['edge_error'][fi]),
                    'centroid':np.array_equal(fast['role_center_error'],f0['role_center_error'][fi]),
                    'region':np.array_equal(fast['region_by_role'],f0['region_by_role'][fi]),
                    'shape':np.array_equal(slow['shape_by_role'],s0['shape_by_role'][si]),
                    'density':np.array_equal(slow['relative_density_l2_by_role'],s0['relative_density_l2_by_role'][si])}
            checks.append({'time_s':t,'array_equal':{k:bool(v) for k,v in fields.items()},
                           'max_raw_abs_difference_mps':float(np.max(np.abs(a-h0['raw_reference'][index])))})
            if not all(fields.values()):
                raise RuntimeError(f'Old P/metric identity failure: {checks[-1]}')
    return {'archived_identity_checks':checks,'old_sample_count':80,'frozen_sources_verified':True}

if __name__=='__main__':
    print(json.dumps(old_identity(*load_frozen()),indent=2))
