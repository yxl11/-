"""Reuse the archived controller/metric loaders without importing a second mission."""
import importlib.util
from pathlib import Path
BASE = Path(__file__).resolve().parent
source = BASE.parent / 'mixed_role_corridor_demo_20261031' / 'p_feedback.py'
spec = importlib.util.spec_from_file_location('verified_corridor_p_loader', source)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
ROOT,CAM,TaskFeedback=prior.ROOT,prior.CAM,prior.TaskFeedback
load_module,load_frozen,old_identity,source_files,sha=(prior.load_module,prior.load_frozen,
    prior.old_identity,prior.source_files,prior.sha)
