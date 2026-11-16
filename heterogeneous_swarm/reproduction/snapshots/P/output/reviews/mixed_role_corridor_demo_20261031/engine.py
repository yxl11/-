"""Bind the registered mixed-airframe model and certifiers to new role counts.

Each engine gets private module dictionaries. The frozen prediction, certificate,
cruise repair and escape function bodies are compiled from hash-pinned files.
Only role-indexed arrays and the dimensions of the common backup change.
"""
import ast
import hashlib
import json
import numbers
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REVIEWS = HERE.parent
MODEL = 'fixedwing_consistent_predictor_20261030'
CERT = 'mixed_airframe_paper_comparison_20261030'
CRUISE = 'fixedwing_local_repair_20261030'
ESCAPE = 'fixedwing_local_escape_20261031'
SOURCE_HASHES = {
    f'{MODEL}/predictor.py': '1d002b01827c8a1c39e73f83a961b47158e819f3ddf06ff7b10ea2b59e1bc8ac',
    f'{MODEL}/protocol.json': 'bb5d829aafe4b6a2b1bf41142844df365e4bd4074366cd88981ff7caeab2c7b1',
    f'{CERT}/safety_interface.py': '0eb226594fbf4fcbd547a4a7a9e49190c39fa547582b624c52e15b51c8568892',
    f'{CERT}/protocol.json': 'e82b5ee911d9c6dadb5894dd1083bd3d72d6ad44ec560de4347e947a89f05952',
    f'{CRUISE}/local_interface.py': 'f84fac6b2c3d5a738529dabf75461ebb03a445cc4a442369090fcc2a2aac1a33',
    f'{CRUISE}/protocol.json': 'ebd57bde94915d59ea9d6c7f35b926a4fb91f74dfce927235e9431964b9a6a76',
    f'{ESCAPE}/local_interface.py': '7365f2dff7dc98df4ebf7852ac3e9b2f721ca1d0aff727d342df3ddaddfcee41',
    f'{ESCAPE}/protocol.json': '0f1222fc31237c5ad3061e01c7263cc1d45fd5c1a8c3141f6a93d3dcb586d635',
}


def _private_module(path, contents, bindings, definitions_only):
    """Give each source's functions the private dependencies for this fleet."""
    namespace = ModuleType('mixed_engine_' + path.parent.name)
    namespace.__file__ = str(path)
    namespace.__dict__.update(bindings)
    syntax = ast.parse(contents, filename=str(path))
    if definitions_only:
        syntax.body = [s for s in syntax.body if isinstance(s, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(syntax, str(path), 'exec'), namespace.__dict__)
    return namespace


def make_engine(counts):
    """Create an independent 3-role shared execution/safety interface."""
    counts = tuple(counts)
    if len(counts) != 3 or any(isinstance(n, bool) or not isinstance(n, numbers.Integral)
                              or n < 1 for n in counts):
        raise ValueError('counts must contain three positive integer population sizes')
    contents = {}
    for name, expected in SOURCE_HASHES.items():
        source = (REVIEWS/name).read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {name}')
        contents[name] = source.decode('utf8')

    model_name = f'{MODEL}/predictor.py'
    model = _private_module(REVIEWS/model_name, contents[model_name], {}, False)
    roles = np.repeat(np.arange(3, dtype=int), counts)
    for key in ('VMAX', 'TAU', 'ACC'):
        old_vector = getattr(model, key)
        per_role = np.array([old_vector[np.flatnonzero(model.ROLES == role)[0]]
                             for role in range(3)])
        setattr(model, key, per_role[roles])
    model.ROLES = roles
    model.FW = np.flatnonzero(roles == 0)
    model.ROT = np.flatnonzero(roles != 0)
    model.PAIR = np.triu_indices(len(roles), 1)

    cert_protocol = json.loads(contents[f'{CERT}/protocol.json'])
    common = np.broadcast_to(model.B, (len(roles), 3)).copy()
    cert_name = f'{CERT}/safety_interface.py'
    safety = _private_module(REVIEWS/cert_name, contents[cert_name], {
        'np': np, 'hashlib': hashlib, 'model': model, 'COMMON': common,
        'PRO': cert_protocol, 'RESERVE': cert_protocol['extra_lifetime_position_allowance_m'],
    }, True)

    cruise_protocol = json.loads(contents[f'{CRUISE}/protocol.json'])
    limits = cruise_protocol['repair_rule']
    cruise_name = f'{CRUISE}/local_interface.py'
    # The validator already uses COMMON.shape. Generalize its legacy error text.
    cruise_source = contents[cruise_name].replace(
        'Expected finite (80,3) state and capped reference arrays',
        'Expected finite fleet-shaped state and capped reference arrays')
    cruise = _private_module(REVIEWS/cruise_name, cruise_source, {
        'np':np, 'COMMON':common, 'model':model, 'certify':safety.certify,
        'advance_model':safety.advance_model, 'frozen_global_safety':safety,
        'MAX_ROUNDS':int(limits['max_rounds']), 'MAX_MEMBERS':int(limits['max_members']),
        'MAX_CALLS':int(limits['max_certificate_calls']),
    }, True)

    escape_name = f'{ESCAPE}/local_interface.py'
    escape = _private_module(REVIEWS/escape_name, contents[escape_name], {
        'np':np, 'COMMON':common, 'model':model, 'certify':safety.certify,
        'advance_model':safety.advance_model, 'old':cruise,
        'PRO':json.loads(contents[f'{ESCAPE}/protocol.json']),
    }, True)
    return SimpleNamespace(model=model, COMMON=common, propose=escape.propose,
                           certify=safety.certify, advance_model=safety.advance_model,
                           RetainedBackupInterface=escape.RetainedBackupInterface)
