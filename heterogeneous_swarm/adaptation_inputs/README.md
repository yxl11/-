# Frozen task exports

These are reading/adapter inputs, not new simulation results or executable PX4 commands.

- `task_samples_10hz.npz`: 1781 samples including 0 and 178 s. q/qd/qdd axes are [time, role, world xyz]; S/dS axes are [time, role, world row, material column]. The task map is x = q_i + S_i z with dimensionless z. S has metres, dS metres/second. World coordinates are ENU by migration convention; confirm the shared physical map before use.
- Online task references must call the source analytic functions at the actual simulation time. Do not infer a continuous controller or its feedforward from linear interpolation of these samples.
- `initial_s*.npz/json`: the three selected archived states, preserving row order. member_id is the present 0--45 index; historical_member_id is provenance only. They start airborne with nonzero speed; they are not takeoff commands.
- `configuration_P.json`: selected P run configuration; use the source-map guide to distinguish active and inherited fields.
- `vehicle_map.template.json`: fill and validate actual instance/namespace/port/frame mappings before ROS use.
- `export_manifest.json`: input hashes, schemas, and output hashes. All frozen inputs were verified before export.
