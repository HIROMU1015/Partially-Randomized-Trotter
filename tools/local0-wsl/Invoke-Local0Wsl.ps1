[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('Join','Pause','Status','Resume')][string]$Action,
    [string]$Config,
    [int]$Slots=0,
    [int]$DpWorkers=0
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'Wsl.Common.ps1')
if(-not $Config){
    $activeConfig=Join-Path $PSScriptRoot 'local0-wsl.active.json'
    $Config=if(Test-Path -LiteralPath $activeConfig){$activeConfig}else{Join-Path $PSScriptRoot 'local0-wsl.json'}
}
$configuration=[IO.File]::ReadAllText([IO.Path]::GetFullPath($Config))|ConvertFrom-Json
if($configuration.host_id -ne 'local0-wsl'){throw 'This command controls local0-wsl only'}
if($Slots -lt 0 -or $DpWorkers -lt 0){throw 'Slots and DP workers must be positive when supplied'}
if($configuration.central_profile -and ($Slots -or $DpWorkers)){
    throw 'Resource settings are owned by the selected RomuTail worker profile.'
}
if($Slots -or $DpWorkers){
    $update=@'
import pathlib,json,os
p=pathlib.Path(CONFIG['profile']);v=json.loads(p.read_text())
for key in ['slots','dp_workers']:
 if CONFIG[key]:v[key]=CONFIG[key]
if CONFIG['dp_workers']:
 for rule in v.get('route_profiles',[]):rule['profile']['dp_workers']=CONFIG['dp_workers']
t=p.with_suffix('.update.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
print(json.dumps(dict(profile=str(p),slots=v['slots'],dp_workers=v['dp_workers'])))
'@
    Invoke-WslCode $update @{profile=$configuration.profile;slots=$Slots;dp_workers=$DpWorkers} $configuration.distribution
}
if($Action -ne 'Status'){
    $enabled=$Action -ne 'Pause'
    if($configuration.linux_participation_file){
        $setLinux=@'
import datetime,json,os,pathlib,tempfile
path=pathlib.Path(CONFIG['file'])
state=json.loads(path.read_bytes())
if state.get('host_id')!='local0-wsl':
    raise ValueError('Participation file belongs to another worker')
state.update(accepting_new_jobs=CONFIG['enabled'],changed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as f:
    temp=pathlib.Path(f.name)
    json.dump(state,f,indent=2)
    f.write('\n')
try:
    os.chmod(temp,path.stat().st_mode & 0o777)
    os.replace(temp,path)
finally:
    temp.unlink(missing_ok=True)
'@
        [void](Invoke-WslCode $setLinux @{file=$configuration.linux_participation_file;enabled=$enabled} $configuration.distribution $configuration.python)
    }
    $state=[ordered]@{host_id='local0-wsl';accepting_new_jobs=$enabled;changed_utc=[DateTime]::UtcNow.ToString('o')}
    $path=[IO.Path]::GetFullPath($configuration.participation_file)
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $path))
    $temporary=$path+'.update.tmp'
    [IO.File]::WriteAllText($temporary,($state|ConvertTo-Json)+"`n",(New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $path -Force
}
$participation=if(Test-Path -LiteralPath $configuration.participation_file){[IO.File]::ReadAllText($configuration.participation_file)|ConvertFrom-Json}else{[pscustomobject]@{host_id='local0-wsl';accepting_new_jobs=$false}}
$inspect=@'
import pathlib,json,os
p=pathlib.Path(CONFIG['profile']);profile=json.loads(p.read_text())
linux=None
if CONFIG.get('linux_participation_file'):
 flag=pathlib.Path(CONFIG['linux_participation_file'])
 linux=flag.is_file() and json.loads(flag.read_bytes()).get('accepting_new_jobs') is True
states=[]
for path in sorted(pathlib.Path(profile['work_dir']).glob('*/*/WORKER_STATE.json')):
 row=json.loads(path.read_text());states.append(dict(job_id=path.parent.parent.name,attempt_id=path.parent.name,**row))
active=sum(x['state'] in ['ASSIGNED','STARTING','RUNNING'] for x in states)
slots=profile.get('slots',profile.get('slot_policy',{}).get('initial_slots',1))
print(json.dumps(dict(host_id=profile['host_id'],slots=slots,slot_policy=profile.get('slot_policy'),dp_workers=profile['dp_workers'],free_slots=max(0,slots-active),spool=profile['work_dir'],dp_cache=profile['prefix_cache_directory'],attempts=states,accepting_new_jobs=CONFIG['enabled'] and linux is not False,windows_accepting=CONFIG['enabled'],linux_accepting=linux,inspection='READ_ONLY_NO_MAILBOX_STATUS_NO_JOB_LAUNCH'),indent=2))
'@
Invoke-WslCode $inspect @{profile=$configuration.profile;enabled=[bool]$participation.accepting_new_jobs;linux_participation_file=$configuration.linux_participation_file} $configuration.distribution
