# Fixed-revision data audit

Status: BLOCKED_DATA_ALIGNMENT / BLOCKED_SENSOR_CONTRACT / BLOCKED_CAMERA_ALIGNMENT.

| Dataset | Revision | All episodes | PegInsert | Inspected episodes | Inspected rows |
|---|---|---:|---:|---:|---:|
| angledusgar/forge-v1 | 0ebe4ff65ade3da8d2011d66cce0259870c95f78 | 2968 | 984 | 1 | 149 |
| angledusgar/forge-failure-v1 | 820392e1167df4d6f5d60f985ec4f3b1b4fd6237 | 53 | 26 | 1 | 149 |

Both selected videos were fully decoded; dimensions and timestamps are in `data_audit.json`.
No GearMesh/NutThread videos, full training data, reward labels or simulator snapshots were downloaded/generated.

- BLOCKED_DATA_ALIGNMENT: collection code/SHA and pre/post EMA action stage unavailable
- BLOCKED_SENSOR_CONTRACT: data-card EE frame differs from pinned stock source; sign/reference/baseline unverified
- BLOCKED_CAMERA_ALIGNMENT: collection camera extrinsics/intrinsics not supplied

No collection run IDs in metadata; episode split cannot exclude correlation across lockstep sibling environments.

Splits are provisional episode groups; no normalization has been fit and no sample admitted to long training.
