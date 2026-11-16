"""Check the common worker against saved pre-evaluation states and commands."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import numpy as np
import worker
from mission import configuration, reference


def check_case(method, run, parameters, engine, feedback):
    with np.load(run/'initial.npz') as saved:
        initial=saved['positions']
        assert np.array_equal(saved['roles'],engine.model.ROLES)
    control=worker.make_control(method,feedback,configuration(),initial,parameters)
    rows=[]
    with np.load(run/'history.npz') as z, np.load(run/'task_history.npz') as task:
        t,positions,velocities,held,capped,mapped,accepted=[z[k] for k in
            ('times','positions','velocities','held_reference','speed_capped','envelope_mapped','accepted')]
        updates=task['raw_reference']; update_times=task['time']
        assert np.array_equal(t,np.arange(8901)*.02)
        assert np.allclose(update_times,np.arange(1780)*.1,rtol=0,atol=3e-14)
        for when in (0,24,77,94,138,174):
            step=round(when/.02); u=control.command(t[step],positions[step],velocities[step])
            cap=u*np.minimum(1.,engine.model.VMAX/np.maximum(np.linalg.norm(u,axis=1),1e-300))[:,None]
            assert np.array_equal(u,updates[step//5]) and np.all(held[step:step+5]==u)
            assert np.array_equal(cap,capped[step]) and np.array_equal(engine.model.map_command(cap)[0],mapped[step])
            fields=worker.record_fields(method,control.last_fields)
            for name,value in fields.items():
                assert np.array_equal(value,task[name][step//5]),(method,when,name)
            engine.set_direction(reference(t[step])[1][1])
            r,v,_=engine.advance_model(positions[step],velocities[step],accepted[step])
            assert np.array_equal(r,positions[step+1]) and np.array_equal(v,velocities[step+1])
            rows.append({'t_s':when,'recorded_fields':list(fields),'all_checks':'bitwise match'})
    return {'source_run':str(run.relative_to(worker.ROOT)), 'method':method,'parameters':parameters,
            'frozen_history_sha256':worker.sha(run/'history.npz'),
            'frozen_task_sha256':worker.sha(run/'task_history.npz'),'samples':rows}


def main():
    spec=worker.read(worker.HERE/'evaluation_protocol.json')
    selection=worker.read(worker.CALIBRATION/'selected_parameters.json')
    assert worker.sha(worker.CALIBRATION/'selected_parameters.json')==spec['selected_parameters_sha256']
    params=spec['method_parameters']
    assert params['P']=={}
    assert all(params[m]==selection['methods'][m]['selected_parameters'] for m in ('RAS08','MMD19'))
    frozen=worker.ARCHIVE
    sparse=frozen.parent/'mixed_sweep_orbit_sparse_20261101'
    for name in ('mission.py','shape_feedback.py','shape_models.py','mission_engine.py','engine.py','p_feedback.py'):
        assert worker.sha(frozen/name)==worker.sha(sparse/name)
    _,h,density=worker.load_frozen()
    feedback=worker.ShapeFeedback(h,density,configuration(),reference)
    engine=worker.make_mission_engine([10,16,20],178,'tangent')
    assert engine.model.H==.02
    cases=[('P',sparse/'runs/P_seed31031',{})]
    oldp=worker.read(frozen/'protocol.json')['method_parameters']
    for method in ('RAS08','MMD19'):
        i=selection['methods'][method]['selected_index']
        cases.append((method,frozen/f'runs/{method}_seed31031',oldp[method]))
        cases.append((method,worker.CALIBRATION/f'runs/{method}_c{i}_s32001',params[method]))
    checks=[check_case(method,path,par,engine,feedback) for method,path,par in cases]
    worker.save(worker.HERE/'runner_regression.json',{'status':'PASS','worker_sha256':worker.sha(worker.HERE/'worker.py'),
        'script_sha256':worker.sha(worker.HERE/'runner_regression.py'),
        'selection_sha256':spec['selected_parameters_sha256'],'cases':checks})
    print('Checked raw, hold, cap, envelope map, diagnostic fields and physical step on 30 saved states.')


if __name__=='__main__': main()
