"""Check completed B3 outputs and link the finished study into existing indexes."""
import json
from pathlib import Path
import h46_b3 as b


def main():
    summary=b.read(b.PUBLIC/'accounting_summary.json')
    provenance=b.read(b.PUBLIC/'scientific_provenance.json')
    for name,expected in provenance['new_source_files'].items():
        if b.signal.sha(b.ROOT/name)!=expected: raise ValueError('USED_SOURCE_CHANGED:'+name)
    for name,expected in provenance['saved_baseline_files'].items():
        if b.signal.sha(b.BASE_PUBLIC/name)!=expected: raise ValueError('SAVED_BASELINE_CHANGED:'+name)
    records=b.read(b.PUBLIC/'signal_records.json.gz');seen=set();native_error=0.;compiled_error=0.;log_error=0.;scalar_error=0.;trials=0
    seeds=set();cohorts={}
    for cohort in ('B3_EXPLORATION','B3_CONFIRMATION'):
        counts=b.read(b.PUBLIC/(cohort.lower()+'_cost_summary.json'))
        if counts['errors'] or counts['complete']!=counts['total'] or not counts['all_workers_joined']:
            raise ValueError('INCOMPLETE_COST_STAGE')
        groups=b.read(b.PUBLIC/(cohort.lower()+'_costs.json.gz'));cohorts[cohort]=counts
        values=b.read(b.PRIVATE/(cohort+'_seeds.json.gz'))
        if len(set(values))!=len(values) or seeds.intersection(values): raise ValueError('SEED_DUPLICATE')
        seeds.update(values)
        for identity,g in groups.items():
            if len(g['trial_files'])!=g['n']: raise ValueError('TRIAL_COUNT')
            native_error=max(native_error,g['observed_maximum_native_error'])
            compiled_error=max(compiled_error,g['observed_maximum_compiled_error'])
            for name in g['trial_files']:
                t=b.read(name);r=t['candidate'];key=(cohort,identity,t['index'])
                if key in seen or t['conditional'] or t['status']!='PASS': raise ValueError('DUPLICATE_OR_UNACCEPTED_SAMPLE')
                seen.add(key);trials+=1
                if sum(t['orders'].values())!=r['q']*r['R_outer']: raise ValueError('EVENT_COUNT')
                log_error=max(log_error,abs(t['log_B']-records[identity]['signal']['log_B']))
                ctx=b.read(b.PRIVATE/(r['system']+'_context.json'))
                scalar_error=max(scalar_error,abs(t['scalar']-ctx['constant']-ctx['prefixes']['0']['scalar']))
    if log_error>1e-9 or scalar_error>1e-12: raise ValueError('MEAN_NATIVE_MISMATCH')
    archive=b.read(b.PUBLIC/'event_archive_summary.json')
    if archive['files']!=trials or not archive['roundtrip_verified']: raise ValueError('PRIVATE_EVENT_ARCHIVE_INCOMPLETE')
    complete=dict(status='COMPLETE_FINITE_EMPIRICAL_BENCHMARK_MANDATORY_STOP',**summary,
        cost_cohorts=cohorts,total_accepted_new_trials=trials,duplicate_samples=0,new_seed_collisions=0,
        maximum_native_vector_error=native_error,maximum_compiled_vector_error=compiled_error,
        maximum_mean_native_log_B_error=log_error,maximum_scalar_error=scalar_error,
        baseline_and_executed_source_unchanged=True,scientific_input_regeneration=0,new_reference_calls=0,
        empirical_allowance=True,population_tail_guarantee=False,global_optimality=False,
        detailed_scientific_review_started=False,formal_paper_claim_adopted=False,
        mandatory_stop_reached=True,next_scientific_stage_authorized=False,
        publication_status=str(b.PRIVATE/'final_status.json'))
    prefetch=b.PUBLIC/'parallel_prefetch.json'
    if prefetch.exists():
        cache=b.read(prefetch)
        if b.signal.sha(b.ROOT/cache['source'])!=cache['source_sha256']:
            raise ValueError('PREFETCH_SOURCE_CHANGED')
        complete['parallel_compiler_prefetch']=cache
    interruption=b.PUBLIC/'interruption_run01.json'
    if interruption.exists(): complete['preserved_interruption_run01']=b.read(interruption)
    for name in ('interruption_run02.json','data_damage_run02.json','event_log_reconstruction.json'):
        if (b.PUBLIC/name).exists(): complete[name.removesuffix('.json')]=b.read(b.PUBLIC/name)
    replays=list((b.PRIVATE/'cost').glob('*/*/*/event_replay_check.json'))
    complete['unfinished_event_replay_checks']=len(replays)
    complete['original_event_log_reconstructions']=archive.get('reconstructed_missing_original_event_logs',0)
    b.save(b.PUBLIC/'completion_provenance.json',complete)
    rel=str(b.PUBLIC.relative_to(b.ROOT));report=rel+'/REPORT.md'
    text=f"""## 2026-10-26 H4/H6 長時間DF-native B3追加

[B3 REPORT]({{prefix}}{report}) / [結果・図表]({{prefix}}{rel}/)。linear1Å/STO-3G/tol-only DF10/19、保存H6 completion、L_D=0（one-body保持）、T=.8〜2、P=0。signal {summary['signal_candidates']}構成、価格{summary['priced_candidates']}候補、探索{cohorts['B3_EXPLORATION']['complete']}・独立確認{cohorts['B3_CONFIRMATION']['complete']}標本。取得集合でB3の探索点費用がB2より低い条件{summary['B3_below_B2']}/40、PF envelopeより低い条件{summary['B3_below_PF_envelope']}/40、四方式最良{summary['B3_expanded_best']}/40。有限標本・経験的精度会計であり、全域最適性・母平均優位・厳密bias保証ではない。旧33/40は旧集合内の結果として保持。旧SOURCE/raw/grant/STOP・他worktree不変。今回の取得・公開後mandatory STOP。

"""
    for name,prefix in [('PROJECT_MAP.md',''),('VALIDATION_STATUS.md',''),
                        ('docs/README.md','../'),('docs/research/研究概要・現状.md','../../'),
                        ('scripts/README.md','../'),('src/trotterlib/README.md','../../')]:
        path=b.ROOT/name;old=path.read_text()
        if not old.startswith('## 2026-10-26 H4/H6 長時間DF-native B3追加'):
            path.write_text(text.replace('{prefix}',prefix)+old)
    scripts=b.ROOT/'scripts/README.md'
    entry='B3実装：`h46_b3.py`（既存signal/native APIの設定と最小拡張）、`h46_b3_probe.py`（条件付き正しさ）、`h46_b3_report.py`（会計・固定確認n）、`h46_b3_finish.py`（図表/REPORT）、`h46_b3_archive.py`（完成eventの可逆保存）。`h46_b3_continue.py`は同一number-operator cacheのfork共有、`h46_b3_recover.py`は元pool異常終了後の未完了trialだけを個別processで復旧する。人工テストは`tests/test_h46_b3.py`と`tests/test_h46_b3_recover.py`。完成済み科学取得を再実行せず保存結果を参照する。\n\n'
    content=scripts.read_text()
    if entry not in content:
        pos=content.index('\n\n')+2;scripts.write_text(content[:pos]+entry+content[pos:])
    manifest=b.ROOT/'artifacts/validation_manifest.json';m=b.read(manifest)
    m['h46_long_time_b3_20261026']=dict(status=complete['status'],report=report,artifacts=rel,
        signal_candidates=summary['signal_candidates'],priced_candidates=summary['priced_candidates'],
        accepted_trials=trials,completion=rel+'/completion_provenance.json',
        interpretation='Empirical eligibility and finite sampled costs; no population tail guarantee or global optimum')
    manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    note=b.ROOT/'docs/research/研究ノート/2026-10-26.md';note.parent.mkdir(exist_ok=True)
    heading='## H4/H6長時間DF-native B3追加・取得終了'
    content=note.read_text() if note.exists() else '# 研究ノート 2026-10-26\n'
    if heading not in content:
        content+='\n'+heading+'\n\n'+text.replace('{prefix}','../../../').split('\n\n',1)[1]
        note.write_text(content)
    with (b.PUBLIC/'REPORT.md').open('a') as stream:
        if prefetch.exists():
            stream.write(f"\n空きCPUで同一の固定探索seed・native builder・compilerを用い、未着手trialのcompiled cacheを先行作成した。先行cache完成{cache['cache_ready_trials']} trial相当、新規compile {cache['new_compiled_axes']}軸、既存cache hit {cache['already_cached_axes']}軸。fingerprintごとの既存排他lockにより同じ回路の重複compileを防止。本体のnative/compiled作用検証とtrial受理は省略せず、先行処理を追加の統計標本として数えていない。本体の将来メモリ増加も予約し、比較可能な大きさのfresh full trialがPASSしてから25%の余裕を付けて補助側の予測を更新した。実行中の科学SOURCE・controllerは変更しなかった。[先行compile記録](parallel_prefetch.json)。\n")
        stream.write(f"\n## 終了時の技術確認\n\n人工・必要回帰13件と復旧cache/byte照合の人工2件PASS。atomic永続化writerの人工入出力も確認した。両cohortの全{trials}標本はPASS・worker回収済み。重複標本0、cohort内/間の新seed衝突0。最大native vector差{native_error:.5g}、compiled差{compiled_error:.5g}、mean/native log_B差{log_error:.5g}。元baselineと記録した実行SOURCEの使用fileは不変。eventは取得byteへの復元一致を確認したgzipで保存した。ただし、以下の欠損event再構成について元byteとの同一性は主張しない。これらは実装整合性の確認であり、母平均優位の証拠とは区別する。\n")
        if interruption.exists():
            stream.write('\n元poolは106 trial完成後にworkerが異常終了し、pool全体が停止した。OOM記録0、CPU/AS/DATAのOS上限は無制限で、原因と元workerのexitcodeは未確定。元のSTOP・結果・progressをprivate interruptions/run01と[公開記録](interruption_run01.json)へ保持した。完成106件は再取得せず、同じ候補・n・cohort・seedで未完了trialだけを個別fork processへ復旧した。保存済み未完了eventは再現byte列を照合。検算のrotated-number行列を同じ式で事前cacheしfork共有した。復旧cacheの代表回路でもQPYを復元してcompiled作用チェックを実施した。旧研究のgrant・one-shot・STOPは再利用していない。\n')
        if (b.PUBLIC/'interruption_run02.json').exists():
            stream.write('\n個別fork復旧では3件のSIGBUSを記録し、その後controllerとworkerが不在となった。完了記録134件のうち1件のresult JSONと2件のevent archiveが空であり、空の原本を保全した。有効な133 resultを再利用し、欠損result1件は同じ固定seedの未完了分として復旧した。resultが有効な1件のevent記録だけを固定seedから再構成し、保存済み両軸native回路fingerprint・compiled価格・log_B・event次数との一致を確認した。元resultのSHAは不変、新規価格観測・compile・signal再取得は0。この再構成で失われた元byteとの同一性は主張しない。以降はfresh Python spawn、自分専用transient user service、atomic fsync保存、fsync後の可逆archiveへ移行した。科学kernel・入力・候補・n・cohort・seedは維持し、旧STOPと欠損原本は保持した。SIGBUSとcontroller消失の根本原因は未確定。[中断](interruption_run02.json) / [欠損](data_damage_run02.json) / [event再構成](event_log_reconstruction.json)。\n')
    print(json.dumps(complete,ensure_ascii=False))


if __name__=='__main__': main()
