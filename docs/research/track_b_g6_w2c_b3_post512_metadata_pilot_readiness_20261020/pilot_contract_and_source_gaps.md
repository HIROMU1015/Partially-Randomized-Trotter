# 16896候補：SOURCEと契約の最小差分

技術準備は完了、正式Pilot releaseは未作成。現在のgate/activationは閉じ、既存512承認を新scopeへ移さない。以下は正式採択後の実装範囲であり、今回の未完技術修正とは区別する。

| 参照実装 | 512固定条件 | 採択後に必要なversioned差分 |
|---|---|---|
| contract.py/spec/tasks | S/P1のcorrectness16pairs/512calls | 6phases/48pairs、offset0..16895、breakdown512+8192+8192、独立fresh scopeのmanifest |
| integration._run | 各condition8pair、512 call_limit、1phase | 3phase barrier/condition、global planned16896、typed success/coverage/failure遷移。Global Bundle primitiveを再開発しない |
| wrapper.Endpoint | correctness/baselineのみ、各axis index<16、pair32 budgets | profile/pass/index<256、phase Nからbudget、setup/angleとnested数の境界を同じ科学意味で拡張 |
| bits.ScientificBits | correctness seed S.SEEDのみ | 正式採択された新namespaceを厳密binding。同じarm/axis prefixをA/B・pass・S/P1で再開始。実bit生成は承認後 |
| execution.worker | correctnessで常時CachedObserver | baseline off/instrumented onを旧schedulerの実装から再利用。両条件同じobserver政策・fused codec。現83.526秒をuninstrumented費用へ外挿しない |
| offline.check_pairs/inspect | 各pair32行・512completed | 48pairとcorrectness/profile-prefix/pass/global joins、採択retentionのfull/hash両row。旧独立consumer primitiveを再利用 |
| run_control.mark_completed | 成功receipt/state=512を検査 | adopted planned=16896へ厳密binding。exclusive intents、SQLite UNIQUE、spent/no-refund、retry0は再開発しない |
| host outer monitoring | _run内窓。初期admit/store・末尾export等は除外 | pre-admissionから全join/finalization/STOPまでの新外側窓、worker END ownCPU・child hierarchy/Git/helpers・postprocessを別列。observer/IPC未分離値はnull/残差 |
| retention/bytes | 全512 full、1GiB/64MiB | correctness full/profile first16+hash案又はfull全件を正式採択。旧64GiB/256MiB案は未採択、容量十分性は未保証。失敗ならSTOPし間引きなし |

run_pairはprofile nを受けるが、それだけで新実科学経路全体が16896対応になるわけではない。driver correctness event_witnessはcorrectness phaseだけ最大512。profileのB nested witnessは最大8448。boundariesの算術案はpilot_boundary_budget_candidates.jsonを参照し、未知内部callを無制限認可にしない。

契約上の未決はn256/二条件/反復、coverage不足の遷移（旧pilot STOPと新512 INCONCLUSIVEの差）、seed、保存粒度・byte上限、CPU/block/observer estimand、scope/SOURCE/runtime固定。今回これらを採択していない。
