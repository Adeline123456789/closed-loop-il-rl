"""Run each simulator lifecycle in a fresh process and aggregate measured counts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from forcecritic.runtime import atomic_json


def aggregate(records):
    keys=('restore','interaction','branch','prefix','evaluation')
    return {'individual_control_step_equivalents':{
        k:sum(r['individual_control_step_equivalents'].get(k,0) for r in records) for k in keys},
        'instances':[i for r in records for i in r.get('instances',[])]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--outcome',type=Path,required=True)
    args,forward=parser.parse_known_args()
    phases=('evaluation','shared') if args.stage=='baseline' else ('single',)
    accounts=[];attempts=[];result={'status':'FAILED_IMPLEMENTATION','reason':'No phase completed'}
    try:
        for phase in phases:
            output=args.outcome.with_name(args.outcome.stem+'.'+phase+'.json')
            cmd=[sys.executable,'-u','scripts/train_forge.py','--stage',args.stage,
                 '--outcome',str(output),*forward]
            if args.stage=='baseline':cmd+=['--baseline-phase',phase]
            print('ISOLATED_PHASE '+phase+' '+json.dumps(cmd),flush=True)
            started=time.time()
            child=subprocess.run(cmd,timeout=86400)
            measured=json.loads(output.read_text()) if output.exists() else {}
            record=output.with_suffix('.accounting.json')
            if record.exists():accounts.append(json.loads(record.read_text()))
            attempts.append(dict(phase=phase,command=cmd,returncode=child.returncode,
                                 started=started,finished=time.time(),outcome=str(output)))
            if child.returncode or not measured:
                result={'status':'FAILED_IMPLEMENTATION','reason':f'{phase} child exited {child.returncode}; inspect phase log/outcome'}
                break
            result=measured
            if measured.get('status')!='PASSED':break
    except BaseException as error:
        result={'status':'FAILED_IMPLEMENTATION','reason':repr(error)}
        raise
    finally:
        atomic_json(args.outcome,{**result,'isolated_phases':attempts})
        atomic_json(args.outcome.with_suffix('.accounting.json'),aggregate(accounts))


if __name__=='__main__':
    main()
