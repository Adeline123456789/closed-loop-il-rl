# Six-axis sensor contract — unresolved

Pinned `forge_env.py` reads `get_link_incoming_joint_force()` at force_sensor body, smooths all six components with alpha=.25, and calls `forge_utils.change_FT_frame` with identity rotations and a translated noisy fixed-asset origin. It adds noise only to three force components for `ft_noisy`.

HF fixed-revision card describes ft as a smoothed wrench in the wrist EE frame. The source and card descriptions cannot be equated without the collection code and physical verification. The helper's translation/sign convention must be measured rather than inferred from its name. PhysX incoming-joint-frame semantics, sign, reference point, bias and actual effective filter delay remain unverified.

Our pure transform uses an explicit convention: R maps old coordinates to new; r is new-origin to old-origin in new coordinates; f_new=R f_old; tau_new=R tau_old + r cross f_new. The test checks both rotation and translation. This does not certify the stock signal.

H=4 legal observed wrench/motion/command history is the model input. No true socket pose, future contact or future action is provided. Cost and hard protection must use N/Nm measurements before network normalization. Physical limits remain null and cannot be treated as safe defaults. Low-frequency smoothed ft cannot certify substep peaks; no bandwidth or safety claim is made.

Status: BLOCKED_SENSOR_CONTRACT. ft_noisy has only three dimensions and is never zero-padded to claim six-axis measurement.
