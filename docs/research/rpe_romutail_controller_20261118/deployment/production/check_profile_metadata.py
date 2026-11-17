#!/usr/bin/env python3
"""Check deployed profile SOURCE paths and hashes; never invoke a science leaf."""
import argparse,hashlib,json,pathlib,subprocess

def inspect(profile):
    facts={};errors=[];rules=profile.get('preparation_profiles',[])
    def source(commit,path,files=None):
        key=(commit,str(path));fact=facts.setdefault(key,dict(commit=commit,path=str(path),status='PASS',checked_files=0))
        try:
            root=pathlib.Path(path)
            if not root.is_dir():raise ValueError('SOURCE directory missing')
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.PIPE).strip()
            if head!=commit:raise ValueError('SOURCE HEAD differs: '+head)
            for name,expected in (files or {}).items():
                item=root/name
                if not item.is_file()or item.is_symlink()or hashlib.sha256(item.read_bytes()).hexdigest()!=expected:
                    raise ValueError('SOURCE file differs: '+name)
            fact['checked_files']=max(fact['checked_files'],len(files or {}))
        except Exception as error:
            fact.update(status='HOLD',reason=str(error));errors.append(dict(commit=commit,path=str(path),reason=str(error)))
    for rule in rules:
        selected=rule['profile']
        for commit,path in selected.get('preparation_sources',{}).items():source(commit,path)
        manifest=selected.get('preparation_adapter_manifest')
        if manifest:source(manifest['commit'],selected['preparation_adapter_root'],manifest['files'])
        if not pathlib.Path(selected['interpreter']).is_file():errors.append(dict(reason='INTERPRETER_MISSING',path=selected['interpreter']))
    for selected in [profile]+[row['profile']for row in profile.get('route_profiles',[])]:
        manifest=selected.get('adapter_manifest')
        if manifest:source(manifest['commit'],selected['adapter_root'],manifest['files'])
        for commit,path in selected.get('route_views',{}).items():source(commit,path)
    return dict(schema='ROMUTAIL_PROFILE_SOURCE_METADATA_CHECK_V1',host_id=profile['host_id'],status='PASS'if not errors else'HOLD',preparation_rule_count=len(rules),sources=list(facts.values()),holds=errors,scientific_calls=0,compile_calls=0,queue_operations=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',required=True);args=parser.parse_args()
    result=inspect(json.loads(pathlib.Path(args.profile).read_text()));print(json.dumps(result,indent=2))
