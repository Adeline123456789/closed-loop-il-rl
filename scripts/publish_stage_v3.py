"""Publish actual staged evidence and update the existing PR through REST."""
import json
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from forcecritic.runtime import atomic_json

report=Path('reports/stage_v3_20261026')
subprocess.run([sys.executable,'scripts/report_stage_v3.py'],check=True)
result=subprocess.run([sys.executable,'scripts/publish.py'])
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
record=dict(time_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,head=head)
if result.returncode==0:
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/force-critic-act-v2'],text=True).split()[0]
    record['remote_head']=remote
    if remote!=head:raise RuntimeError('Remote SHA does not match local publication')
    body=(report/'FINAL_REPORT.md').read_text()
    body+='\n\n历史结果与失败记录保留：[历史报告](https://github.com/Adeline123456789/closed-loop-il-rl/blob/codex/force-critic-act-v2/reports/20261023T104759Z/FINAL_REPORT.md)。\n'
    body+='\n执行协议：[v3](https://github.com/Adeline123456789/closed-loop-il-rl/blob/codex/force-critic-act-v2/PROTOCOL_STAGE_V3.md)。原始数据集和模型大文件保留本地，仅提交哈希、代码、配置、日志、CSV 和自动报告。\n'
    payload=Path('.cache/stage_v3_pr.json');payload.write_text(json.dumps(dict(body=body),ensure_ascii=False))
    pr=subprocess.run(['gh','api','--method','PATCH','repos/Adeline123456789/closed-loop-il-rl/pulls/1',
        '--input',str(payload),'--jq','.html_url'])
    record['pr_returncode']=pr.returncode
atomic_json(Path('.cache/stage_v3_publication.json'),record)
raise SystemExit(result.returncode or record.get('pr_returncode',0))
