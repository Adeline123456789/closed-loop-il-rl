"""Pinned, metadata-first PegInsert audit. Raw downloads stay outside Git."""
import hashlib
import json
from pathlib import Path
import av
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from .splits import group_splits


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            path.write_bytes(response.read())
    return path


def audit(config, out, cache=Path('.cache/hf')):
    out = Path(out)
    result = {'status': 'BLOCKED', 'datasets': [], 'blocking_items': [
        'BLOCKED_DATA_ALIGNMENT: collection code/SHA and pre/post EMA action stage unavailable',
        'BLOCKED_SENSOR_CONTRACT: data-card EE frame differs from pinned stock source; sign/reference/baseline unverified',
        'BLOCKED_CAMERA_ALIGNMENT: collection camera extrinsics/intrinsics not supplied']}
    all_episodes = []
    for label in ('success', 'failure'):
        repo = config['data'][label+'_repo']
        revision = config['data'][label+'_revision']
        if not revision or len(revision) != 40:
            raise ValueError('fixed HF revision required')
        root = cache / repo.replace('/', '--') / revision
        url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/'
        info = json.loads(fetch(url+'meta/info.json',root/'meta/info.json').read_text())
        tasks = [json.loads(x) for x in fetch(url+'meta/tasks.jsonl',root/'meta/tasks.jsonl').read_text().splitlines()]
        eps = [json.loads(x) for x in fetch(url+'meta/episodes.jsonl',root/'meta/episodes.jsonl').read_text().splitlines()]
        task = next(t for t in tasks if t['task'] == 'insert the peg into the socket')
        selected = [e for e in eps if task['task'] in e['tasks']]
        for e in selected:
            all_episodes.append({'parent_id':f'{repo}:{revision}:PegInsert:{e["episode_index"]}',
                                 'source':label, 'episode_index':e['episode_index']})
        samples = []
        # One complete episode from each source; never download GearMesh/NutThread videos.
        for episode in selected[:1]:
            idx = episode['episode_index']; chunk=idx//info['chunks_size']
            relative = info['data_path'].format(episode_chunk=chunk,episode_index=idx)
            path = fetch(url+relative,root/relative)
            table = pq.read_table(path)
            dims = {}
            for name, dim in [('state',8),('actions',7),('ft',6),('ft_noisy',3)]:
                a = np.stack(table[name].to_pylist())
                if a.shape != (len(table),dim) or not np.isfinite(a).all():
                    raise ValueError(f'invalid {name} shape/value')
                dims[name] = list(a.shape)
            if set(table['task_index'].to_pylist()) != {task['task_index']}:
                raise ValueError('mixed task parquet')
            ts=np.asarray(table['timestamp'].to_pylist())
            video=info['video_path'].format(episode_chunk=chunk,episode_index=idx,video_key='image')
            vp=fetch(url+video,root/video)
            with av.open(str(vp)) as container:
                stream=container.streams.video[0]
                frames=sum(1 for _ in container.decode(video=0))
                video_info={'width':stream.width,'height':stream.height,'fps':str(stream.average_rate),
                            'codec':stream.codec_context.name,'decoded_frames':frames}
            if frames != len(table):
                raise ValueError('video/parquet frame count mismatch')
            sample={'episode':idx,'rows':len(table),'dimensions':dims,
                    'timestamp_delta_median':float(np.median(np.diff(ts))),
                    'parquet_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'video_sha256':hashlib.sha256(vp.read_bytes()).hexdigest(),
                    'video_stream':video_info, 'bytes':path.stat().st_size+vp.stat().st_size,
                    'columns':table.column_names}
            samples.append(sample)
        result['datasets'].append({'repository':repo,'revision':revision,'metadata_total_episodes':len(eps),
                                   'peg_episodes':len(selected),'fps':info['fps'],'samples':samples,
                                   'collection_run_id_available':any('collection_run_id' in e for e in eps)})
    splits=group_splits(all_episodes,config['seed'])
    result['splits']={s:sum(v==s for v in splits.values()) for s in ('train','validation','test')}
    result['split_limitation']='No collection run IDs in metadata; episode split cannot exclude correlation across lockstep sibling environments.'
    (out/'episode_splits.json').write_text(json.dumps(splits,indent=2))
    (out/'data_audit.json').write_text(json.dumps(result,indent=2))
    lines=['# Fixed-revision data audit','','Status: BLOCKED_DATA_ALIGNMENT / BLOCKED_SENSOR_CONTRACT / BLOCKED_CAMERA_ALIGNMENT.',
           '', '| Dataset | Revision | All episodes | PegInsert | Inspected episodes | Inspected rows |',
           '|---|---|---:|---:|---:|---:|']
    for d in result['datasets']:
        lines.append(f"| {d['repository']} | {d['revision']} | {d['metadata_total_episodes']} | {d['peg_episodes']} | {len(d['samples'])} | {sum(s['rows'] for s in d['samples'])} |")
    lines += ['', 'Both selected videos were fully decoded; dimensions and timestamps are in `data_audit.json`.',
              'No GearMesh/NutThread videos, full training data, reward labels or simulator snapshots were downloaded/generated.',
              '', *['- '+x for x in result['blocking_items']], '', result['split_limitation'],
              '', 'Splits are provisional episode groups; no normalization has been fit and no sample admitted to long training.']
    (out/'data_audit.md').write_text('\n'.join(lines)+'\n')
    return result
