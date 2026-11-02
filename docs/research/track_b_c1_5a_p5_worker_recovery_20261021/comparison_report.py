"""Phase D: use OLD summaries/setup/terminal only; never opens old raw archives."""
import csv
import io
from common import HERE,OLD,read,save,require,digest

def comparison():
    new=read(HERE/'result/summary.json');terminal=read(HERE/'result/terminal.json')
    require(new['complete'] and new['completed']==1024 and terminal['complete'],'NEW_P5_COMPLETION_REQUIRED')
    summaries=[read(OLD/'result'/m/'summary.json') for m in ('ordinary','full_return')]+[new]
    oldterm=[read(OLD/'result'/m/'terminal.json') for m in ('ordinary','full_return')]
    oldsetup=[read(OLD/'result'/m/'setup.json') for m in ('ordinary','full_return')]
    prod=new['producer'];rows=[]
    def add(metric,values,scope):rows.append(dict(metric=metric,ordinary=values[0],full_return=values[1],closed_P5_full=values[2],scope=scope))
    for field in ('completed','accepted_paths','valid_zero_paths','accepted_fraction_observed','empirical_weight_second_moment','maximum_observed_weight'):
        add(field,[s[field] for s in summaries],'all 1024 planned classical paths; empirical')
    for key in ('signed_Q','child_rotations','controlled_gate_instructions','logical_1Q','logical_2Q','logical_multi_qubit','instructions','depth'):
        add(key,[s['logical_per_classical_trial'][key] for s in summaries],'raw logical per classical trial; TAIL_ONLY_PLUS_SCALAR')
    for key in ('angle_cache_hits','angle_cache_misses','event_cache_hits','event_cache_misses'):
        add(key,[s[key] for s in summaries[:2]]+[prod[key]],'old:21 worker caches; P5R:one producer cache')
    add('normalizer_completion_CPU_seconds',[s['stages']['sampler_normalizer_selector_setup']['CPU_seconds'] for s in oldsetup]+[prod['setup_stages']['P5_normalizer_completion_setup']['CPU_seconds']],'setup excluding input restore; P5 group prefix separate')
    add('group_prefix_setup_CPU_seconds',['included in old lookup per event','included in old lookup per event',prod['setup_stages']['group_prefix_selector_setup']['CPU_seconds']],'P5R extra exact cumulative integer index')
    add('producer_or_coordinator_input_CPU_seconds',[s['stages']['R1_input_and_registry_setup']['CPU_seconds'] for s in oldsetup]+[prod['setup_stages']['R1_input_registry_setup']['CPU_seconds']],'model/basis restore; P5R provider restores separately')
    add('online_sampling_checks_event_Evidence_CPU_seconds',[s['stages']['sampling_checks_event_Evidence_storage']['CPU_seconds'] for s in summaries[:2]]+[prod['online_stages']['sampling_checks_event_Evidence_storage']['CPU_seconds']],'same broad sampling stage; implementation/cache differ')
    add('online_angle_query_Evidence_CPU_seconds',[s['stages']['angle_query_and_Evidence']['CPU_seconds'] for s in summaries[:2]]+[prod['online_stages']['angle_query_and_Evidence']['CPU_seconds']],'global vs per-worker angle cache differs')
    add('logical_build_CPU_seconds',[s['stages']['logical_build']['CPU_seconds'] for s in summaries[:2]]+[new['logical_worker_stages']['logical_build']['CPU_seconds']],'raw logical; no native synthesis')
    add('online_stage_CPU_seconds_sum',[s['worker_batch_CPU_seconds'] for s in summaries[:2]]+[sum(v['CPU_seconds'] for v in prod['online_stages'].values())+sum(v['CPU_seconds'] for k,v in new['logical_worker_stages'].items() if k!='R1_input_registry_setup')],'old batch CPU includes overhead; new sum of timed stages; NONIDENTICAL BOUNDARIES')
    add('P5R_provider_restore_CPU_seconds',['not separately repeated per batch','not separately repeated per batch',new['logical_worker_stages']['R1_input_registry_setup']['CPU_seconds']],'new32 fresh provider restores; all hierarchy includes it')
    add('all_hierarchy_CPU_seconds',[t['all_method_hierarchy_CPU'] for t in oldterm]+[terminal['all_job_hierarchy_CPU_seconds']],'old method subtree vs new complete job; includes setup/import/validation/monitor; environments differ')
    add('online_wall_seconds',[t['online_wall_seconds'] for t in oldterm]+[prod['online_wall_seconds']+read(HERE/'result/worker_supervision/summary.json').get('supervised_wall_seconds',0)],'P5 producer online plus entire provider phase; old concurrent method')
    add('total_wall_seconds',[t['total_wall_seconds'] for t in oldterm]+[terminal['wall_seconds']],'old three coordinators concurrent; new P5 isolated, not matched load')
    add('max_single_worker_RSS_KiB',[s['max_worker_RSS_KiB'] for s in summaries[:2]]+[new['max_logical_worker_RSS_KiB']],'old inherited virtual state vs P5R fresh provider; not sum/physical job peak')
    add('sampler_owner_max_RSS_KiB',[s['setup_max_RSS_KiB'] for s in oldsetup]+[prod['max_RSS_KiB']],'old setup vs new producer lifetime; observation boundaries differ')
    add('saved_bytes_before_summary',[s['total_method_saved_bytes_before_summary'] for s in summaries[:2]]+[new['saved_bytes_before_summary']],'monitor/Evidence schemas and cutoffs differ; transport archives excluded')
    add('workspace_beyond_system',[1,1,1],'8 system +1 control; excluded deterministic halves/preparation/readout')
    for metric in ('native_T_CX','deterministic_halves','state_preparation','readout','sufficient_N'):
        add(metric,['UNKNOWN']*3,'not acquired; never substitute zero')
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=['metric','ordinary','full_return','closed_P5_full','scope']);writer.writeheader();writer.writerows(rows)
    (HERE/'comparison.csv').write_text(buffer.getvalue())
    lines=['# C1-5A-P5R：保存要約による3方式比較','',
        'Linear H4 1.00 Å / STO-3G / DF12 / R1 LD3 / 8 system qubits / 324 labels・13 bases。T=4/5, q=1, r=4, m=5。全方式1024 whole-path・4096 microsteps。`TAIL_ONLY_PLUS_SCALAR` の探索的観測である。',
        '', 'Ordinary/Fullは旧commit `b022f2ea421649015a4c21ea3c904fe619d0c198` の保存summary・setup・terminalだけを引用した。旧rawの新規再解析・旧方式再実行は行っていない。P5は今回の新規one-shot結果。',
        '', '|指標|Ordinary|General Full|Closed P5 (P5R)|','|---|---:|---:|---:|']
    chosen={'completed','accepted_paths','valid_zero_paths','empirical_weight_second_moment','maximum_observed_weight','signed_Q','logical_1Q','logical_2Q','instructions','depth','normalizer_completion_CPU_seconds','online_sampling_checks_event_Evidence_CPU_seconds','all_hierarchy_CPU_seconds','total_wall_seconds','angle_cache_misses','saved_bytes_before_summary'}
    for row in rows:
        if row['metric'] in chosen:
            values=[format(row[k],'.10g') if isinstance(row[k],float) else str(row[k]) for k in ('ordinary','full_return','closed_P5_full')]
            lines.append('|'+row['metric']+'|'+'|'.join(values)+'|')
    lines.extend(['','logical費用はclassical trial当たり。CPU/wall単位は秒。上表のCPU/wall/storageは計測境界と実装条件が異なるため、方式固有の速度優位を示す比較ではない。詳細は[CSV](comparison.csv)のscope欄。',
        '', '旧Ordinaryの4070/4096、Fullの4073/4096 microstepsがraw degree 0で、非空parentは少ない。両方式ともzero pathは観測されず、各pathに4 child rotationsが残る。Ordinaryではraw degree 2が26回、reduced length 2が25回、Fullでは各23回だった。別method keyの異なる乱数列なので、差をpaired効果や有意な改善と扱えない。保存summaryで見える小さなlogical差と大きなFull古典費用を、この範囲で説明できる。',
        '', '旧Fullは全4096 microstepsでlocal parent packetを取得し、sampling/checks/Evidence stageに4410.0071 CPU秒を費やした。これには有限bit検査と保存処理が含まれ、Green演算単独のCPU測定ではない。',
        '', 'P5Rは104,977群のnormalizer/completionを既存P5Closedで構築し、全群の構築費用を支払う。既知の累積整数selectorとproducer共通cacheを追加した。Ordinary/Fullにも同等の合法な最適化を認める必要があり、今回のCPU差を手法の新規性・公平な最速baseline比較に変換しない。',
        '', '旧実行はNumPy/Qiskit初期化後fork・各方式21 worker・3 coordinator同時実行。今回はP5 producer 1＋その終了後32 fresh exec logical providers。小規模行列向けjob-local BLAS threads=1、global angle/event cache、fsyncと監視・Evidence再読込、providerごとのbasis復元が相違点。旧forkのCOWと新provider RSSは同じ物理メモリ尺度でない。',
        '', 'P5の有限bit canonical certificateは依然UNCERTIFIED_EXPLORATORY。人工同値性確認と4096 eventの実proposal×weightチェックは、未知のcanonical一様bias/range定理やIID保証を追加しない。経験的二乗平均・最大値は母分散・厳密range・十分Nではない。native T/CX/1Q、deterministic halves、state preparation/readout、shots、coherent signal、全round RPEは未取得。',
        '', '研究継続・縮小・終了、資源優位、原著価値、投稿GOはユーザー・GPTの判断に返す。追加比較を検討する場合の最小候補は、同等selector/cacheとCPU計測境界を揃えた比較設計の要否を先に判断すること。実行・証明・native取得は別途承認が必要。',
        '', '参照：[旧summary群](../track_b_c1_5a_h4_limited_resource_pilot_20261021/result/)、[新P5 summary](result/summary.json)、[新terminal](result/terminal.json)、[技術修正](TECHNICAL_REPAIR.md)。'])
    (HERE/'comparison.md').write_text('\n'.join(lines)+'\n')
    save(HERE/'comparison_provenance.json',dict(old_data_policy='saved summary/setup/terminal only; no old raw opened',
        files=[dict(path=str(p.relative_to(HERE.parents[2])),sha256=digest(p)) for p in [OLD/'result'/m/f for m in ('ordinary','full_return') for f in ('summary.json','setup.json','terminal.json')]+[HERE/'result/summary.json',HERE/'result/terminal.json']],
        no_new_science=True,no_resource_winner=True))

if __name__=='__main__':comparison()
