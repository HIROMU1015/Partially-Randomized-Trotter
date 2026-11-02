"""Bind saved metadata/runtime and issue only a non-executable B2 grant draft."""
import argparse,json
from pathlib import Path
import h8_phase_b2_contract as c
import h8_phase_b2_runtime as r
import h8_strengthened_export as exported


def prepare(previous_config,input_paths,before_path,directory,*,source=None):
 old=r.read(previous_config);mapping=r.read(input_paths);before=r.read(before_path);p=c.plan()
 directory=r.private_path(directory)
 if directory.exists():raise ValueError('NEW_B2_PREPARATION_DIRECTORY_REQUIRED')
 files=[]
 for row in p['binding']['inputs']:
  path=Path(mapping[row['logical_name']])
  if path.stat().st_size!=row['bytes'] or r.sha(path)!=row['sha256']:raise ValueError('FIXED_B2_INPUT_CHANGED')
  files.append(dict(row,path=str(path)))
 font=dict(old['font_seed']);font['name']=font.get('name',font.get('filename'))
 if Path(font['name']).name!=font['name'] or r.sha(font['path'])!=font['sha256']:raise ValueError('FONT_SEED_IDENTITY')
 export=r.read(mapping['h8_preparation_export']);exported.validate_export(export,expected_hashes=old['expected_preparation_hashes'])
 for prefix in (15,21):
  expected=p['binding']['prefix_preparations'][str(prefix)]
  if c.digest(export['faithful:p28']['blocks'][:prefix+1])!=expected['ordered_block_description_sha256']:raise ValueError('PREFIX_METADATA_CHANGED')
 models=r.read(mapping['B1_frozen_models']);predictions=r.read(mapping['B1_B2_predictions']);terminal=r.read(mapping['B1_phase_terminal'])
 if (c.digest(models)!=c.MODEL_SHA or terminal['status']!='COMPLETE_MANDATORY_STOP' or terminal['model_sha256']!=c.MODEL_SHA
  or predictions['model_sha256']!=c.MODEL_SHA or [(x['candidate_id'],x['axis'],x['exact_predicted_RZ']) for x in predictions['axes']]!=
  [(x['case_id'],x['axis'],x['exact_prediction']) for x in c.AXES]):raise ValueError('CANONICAL_B1_MODEL_AND_AXES')
 cfg=dict(schema='H8_PHASE_B2_CONFIG_V1',SOURCE=source,execution_sealed=source is not None,
  plan_sha256=c.digest(p),source_hashes={name:r.sha(c.ROOT/name) for name in c.SOURCES},
  provider_root=old['provider_root'],provider_source=c.PROVIDER,provider_hashes=old['provider_hashes'],
  runtime_file_hashes=old['runtime_file_hashes'],expected_profile_digest=c.PROFILE,python=old['python'],
  expected_preparation_hashes=old['expected_preparation_hashes'],scientific_binding=p['binding']['scientific_binding'],
  input_mapping={row['logical_name']:row['path'] for row in files},files=files,font_seed=font,resources=p['resources'],
  protected_files=[dict(row,path=path) for path,row in before['protected'].items()],protected_worktrees=before['worktrees'],
  output=str(directory/'output01'),lock=str(directory/'B2_ONE_SHOT'),control=str(directory/'control01'))
 r.verify_files(cfg,git=source is not None);r.protected(cfg)
 directory.mkdir(mode=0o700);r.Writer(directory).write('config_UNAPPROVED.json',cfg)
 grant=dict(schema='H8_PHASE_B2_NEW_SCOPE_GRANT_V1',scope=c.SCOPE,approved_by_user=False,runtime_authorization=False,
  approval_text=None,SOURCE=source,config_sha256=r.sha(directory/'config_UNAPPROVED.json'),plan_sha256=c.digest(p),
  resources=p['resources'],call_caps=p['call_caps'],one_shot=True,retry=False,resume=False,refit=False,C_authorized=False,
  driver_cpu=None,monitor_cpu=None,worker_cpus=[],allowed_cpus=[],command=None,next_stage_authorized=False,valid=False)
 r.Writer(directory).write('grant_INVALID_DRAFT.json',grant)
 receipt=dict(status='PREPARED_NOT_AUTHORIZED',SOURCE=source,plan_sha256=c.digest(p),config_sha256=grant['config_sha256'],
  source_hashes=cfg['source_hashes'],provider_files_verified=len(cfg['provider_hashes']),runtime_files_verified=len(cfg['runtime_file_hashes']),
  provider_map_sha256=c.digest(cfg['provider_hashes']),runtime_map_sha256=c.digest(cfg['runtime_file_hashes']),
  input_receipts=[{k:v for k,v in row.items() if k!='path'} for row in files],
  protected_files_unchanged=len(cfg['protected_files']),protected_worktrees_unchanged=len(cfg['protected_worktrees']),
  model_sha256=c.MODEL_SHA,axes=p['axes'],compile_concurrency=1,new_H8_objects=0,new_compile=0,new_fit=0,
  new_valid_grants=0,fresh_resource_gate_issued=False,next_stage_authorized=False,mandatory_STOP=True)
 r.Writer(directory).write('preparation_receipt.json',receipt)
 return receipt,grant


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('previous-config','input-paths','before','private-directory'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--source');a=parser.parse_args()
 receipt,_=prepare(a.previous_config,a.input_paths,a.before,a.private_directory,source=a.source)
 print(json.dumps(receipt))


if __name__=='__main__':main()
