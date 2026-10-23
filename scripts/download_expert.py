"""Download only the pinned public environment-acceptance expert; verify LFS hash."""
import hashlib
import json
from pathlib import Path
import urllib.request
import time

REVISION='447a6109cec7c228763d1bfc8cf0ab73f6558a1b'
SHA256='10efdf802e8abfccaa13f6e7a18cc2887ca2f3f96b35526bc9984fde419e59f8'
SIZE=213431615
NAME='forge_peginsert_2026-06-30.pth'
root=Path('.cache/expert')/REVISION;root.mkdir(parents=True,exist_ok=True)
path=root/NAME;part=path.with_suffix('.part')
url=f'https://huggingface.co/angledusgar/forge-peginsert-v1/resolve/{REVISION}/{NAME}'
for attempt in range(5):
    if path.exists() or (part.exists() and part.stat().st_size==SIZE):break
    offset=part.stat().st_size if part.exists() else 0
    if offset>SIZE:raise RuntimeError('partial expert exceeds expected size')
    req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
    try:
        with urllib.request.urlopen(req,timeout=90) as response:
            if offset and response.status==206:
                if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise RuntimeError('server returned a mismatched range')
                mode='ab'
            elif response.status==200:mode='wb'
            else:raise RuntimeError(f'Unexpected HTTP status {response.status}')
            with part.open(mode) as f:
                while chunk:=response.read(1024*1024):f.write(chunk)
    except (OSError, TimeoutError) as error:
        # Exception strings can include signed redirect URLs; log only the class.
        print(json.dumps({'attempt':attempt+1,'error_type':type(error).__name__}),flush=True)
    print(json.dumps({'attempt':attempt+1,'downloaded_bytes':part.stat().st_size if part.exists() else 0}),flush=True)
    if attempt<4 and (not part.exists() or part.stat().st_size!=SIZE):time.sleep(2)
candidate=path if path.exists() else part
h=hashlib.sha256()
with candidate.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
if candidate.stat().st_size!=SIZE or h.hexdigest()!=SHA256:raise RuntimeError('expert size/hash mismatch')
if candidate==part:part.replace(path)
print(json.dumps({'path':str(path),'repository':'angledusgar/forge-peginsert-v1','revision':REVISION,
                  'size':SIZE,'sha256':SHA256,'download_status':'VERIFIED',
                  'compatibility':'NOT_RUN','project_expert_success':'NOT_RUN'}))
