import json
import numpy as np
import pytest
from forcecritic.diagnostics.artifacts import atomic_json


def test_numpy_diagnostic_metadata_retains_types(tmp_path):
    path=tmp_path/'fidelity.json'
    atomic_json(path,{'checkpoints':[{'hard':np.bool_(True),'passed':np.bool_(False),
        'repeats':np.int64(3),'gain':np.float32(.25),'errors':np.array([.1,.2])}]})
    row=json.loads(path.read_text())['checkpoints'][0]
    assert row['hard'] is True and row['passed'] is False
    assert row['repeats']==3 and row['gain']==.25
    assert row['errors']==[.1,.2]


def test_bad_measurement_cannot_replace_valid_report(tmp_path):
    path=tmp_path/'fidelity.json';atomic_json(path,{'status':'measured'})
    before=path.read_bytes()
    with pytest.raises(ValueError):atomic_json(path,{'bad':np.float64(np.nan)})
    assert path.read_bytes()==before
    with pytest.raises(TypeError):atomic_json(path,{'bad':object()})
    assert path.read_bytes()==before
