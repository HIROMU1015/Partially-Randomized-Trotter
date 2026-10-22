# 非DF PR/PF基盤の堅牢化とH4 pilot準備（2026-10-23）

本作業は技術改修・人工検証・H2保存証拠の診断・H4の実行準備である。
H4の新SCF/FCI/DF生成・状態/信号計算・MC sampling・compileは未承認で、実行しない。
H2の追加sampling・RNG replay・compile・母平均費用評価・科学的再ランキングも行わない。
詳細科学レビューは開始せず、本書は技術報告とする。

## 基準と保全

baseはH2結果commit `c8990de7721545faff88b636108034dedb80a463`、旧科学SOURCEは
`7a6c4263d96153ecbe721dcfb6941d0d2684b3b0`。GitHub対象branchとclean local worktreeが同一HEADであることを
着手前に確認した。新branch `codex/non-df-h4-readiness-20261023` を分岐した。
旧科学結果・図・grant・marker・ledger・STOP・SOURCE対象コードは保全する。
共通運用指示2026-10-17の保存正本
`docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022/sources/common_instructions.md` を実際に読み、§8.4を適用した。
起動時の128 CPU affinity、約981 GiB available RAM、約114 GB空きdiskと低loadを確認し、共有環境を変更しない。

## A：独立processの監視・回収

旧 `isolated_qiskit_execution.py` は科学SOURCEにhash固定されているため、変更せず残す。
新 `isolated_qiskit_supervisor.py` と `run_non_df_h4_readiness.py` が今後の監視付き経路である。
旧H2の停止済み科学runnerは再実行せず、新監視経路への後付け再結合も行わない。

専用の親Python processがcaseごとにfresh interpreterと独立sessionを起動する。
Qiskit compiler/session objectをcase間共有せず、数学・compiler設定・seed・回路構造は既存APIを使用する。
全出力・marker・stdout/stderrを起動前に排他的に確保し、ready/releaseは一時JSONからhardlinkで原子的に公開する。
readyはPID・親PID・case・launch tokenを照合する。
runの排他的mkdirに加え、固定grantに属するclaim directoryでcase_idとSOURCE/input/grant bindingを消費する。
run IDだけを変えた再投入ではclaimを返却しない。claim directoryのsymlinkや別所有者は拒否する。
将来の科学runnerは、このclaim directoryとbindingを契約に固定しなければならない。

ready_timeout/run_timeout/heartbeat_timeoutは契約ごとのoptional値で、既定None。
正常計算を短時間で切る一律wall上限は設けない。有限時間の人工ハング試験では明示した短い試験値を使用する。
deadlineをNoneにした場合、無応答を経過時間だけで自動判定する保証はないが、SIGINT/SIGTERM・親消失・STOP.requestは監視する。
heartbeat timeoutは明示契約でのみ有効にする。重いQiskit処理と同じthreadのheartbeat停止を、科学的失敗と誤認する設計にはしない。

Linux pidfdで記録したidentityにだけTERM→KILLを送る。supervisor自身をprivate subreaperとし、
setsidした子孫・二重fork後のorphanも祖先関係/直接adoptionから所有を追跡する。
PID再利用で別processへsignalを送らず、Popenの直接子とadoptされた子孫をwaitで回収する。
一部caseの異常終了・親例外・中断・timeoutは残りの自分のcaseも停止し、retryせずSTOPと未完了を残す。
run/STOPとledger、case/process.json、全stdout/stderr、reason/tracebackを保存する。

保証範囲は、この専用supervisorが管理する通常のLinux process、SIGINT/SIGTERM、明示deadlineと保存可能なfilesystemである。
supervisor自体へのSIGKILL、kernelのuninterruptible sleep、machine failureでは普遍的回収保証はできない。
cleanup graceを超えて残るidentityはpendingとして失敗STOPへ残し、回収済みと偽らない。
filesystemへの書込みが不能ならSTOP保存自体も保証できず、stderrへSTOP_PERSISTENCE_FAILEDを出す。
他job・他ユーザーのprocessや共有venv/Qiskit/cacheは操作しない。

## B：H2保存済み希少eventの診断

[保存証拠診断](../artifacts/non_df_h4_readiness/2026-10-23/rare_event_diagnosis.md)と
[128 trajectoryの価格・seed・欠測join](../artifacts/non_df_h4_readiness/2026-10-23/rare_event_diagnosis.json)を参照。
条件はH2/1.00 Å/STO-3G、DF tol-only1e-8 rank3、T=.8/q4/δ=.2/S2、PR r2/K4、
Pauli L_D7/14非identity項とDF L_D1/3 fragment＋one-body、各64 paired trajectories。

Pauliは512位置でorder0/2/4が510/2/0、保存確率は0.9985477359 / 0.00145191185 / 3.52270760e-7。
DFは512/0/0、確率0.9995230410 / 0.000476921027 / 3.79418339e-8。
未観測確率・期待数は元validationから転記する。新乱数を引かず、seedの再生も行わない。

各sampleはseedとpaired RZだけを持ち、個別event列/order位置/component IDを保存していない。
Pauliのorder2がどの価格に対応するかは未同定。DFは全位置order0のため各trajectoryが0×8と集計から推定できるが、
component IDとevent位相の個別記録は欠測である。両表現のevent費用を同一視して穴埋めしない。
full wrapperではcompilerのcancellationがあるため、event単独費用を足して母平均を復元できるとも主張しない。

既存のC SEはPauli15.550576、DF1.033493 RZ/axis。名目Student-t95%は観測した64 trajectoryの分散に基づく診断で、
未観測orderの費用寄与や厳密な母平均coverageは保証しない。強制orderの代数検査は費用MC標本ではない。
[追加情報取得契約案](../artifacts/non_df_h4_readiness/2026-10-23/rare_event_acquisition_DRAFT.json)は
各表現・order2/4・8位置・8条件付きcomponent drawの計256 trajectory、両軸full wrapperという未承認案を記す。
1位置だけ指定order、他7位置order0という条件付き診断法の正式採用も未承認である。
この案は母平均推定器ではなく、複数希少eventや全component supportを網羅しない。
母平均保証には別の正当化された費用上界/supportまたは承認済み統計設計が必要で、今回採択しない。

## C：H4の実行契約案と一括承認の単位

[H4_execution_contract_DRAFT.json](../artifacts/non_df_h4_readiness/2026-10-23/H4_execution_contract_DRAFT.json)
はDRAFT_NOT_ADOPTED_NOT_AUTHORIZED。grant_valid/dispatch_enabledはfalseで、科学起動CLIも拒否する。
新しい数値パラメータ・prefix規則・標本数は実行提案であり、正式採用ではない。

|項目|具体的な提案|
|---|---|
|物理系|linear H4、z=0/1/2/3 Å、隣接1.00 Å、STO-3G、charge0/spin0、active/frozenなし|
|軌道・積分|canonical real RHF1回、conv1e-12/max100/minao、同じ未閾値MO h・chemist eriから両表現。alpha/beta交互、Qiskit endian、核/identity位相保持|
|DF|既存OpenFermion分解1回、tol-only1e-8、final_rankなし。実rankは入力生成後に記録し、H2・旧H4 rankから推測しない|
|状態|nonDFのNalpha=Nbeta=2 sector（次元36）最低状態をcanonical phaseに固定。全case同じstate。DF sector解と独立PySCF FCI各1回で照合|
|Pauli prefix|(-abs(c),label)順。非identity abs係数和の50%以上を初めて含むprefix。正の非空tailが必要|
|DF prefix|既存abs(lambda)*Frobenius(g)^2順位、index tie-break。同じく50%ランキング重みprefix、one-body常時保持。ランキングproxyをRTE lambdaと同一視しない|
|候補|各表現PF全保持/PR構造prefix各1、計4候補のみ。空tailならSTOPし、PFに置換・再探索しない|
|時間・法則|T=.8/q8/δ=.1/S2、r4/K4。既存paired finite-RTE、実際のTailごと正規化。H2条件をそのまま転写せず、単一の短いstepを提案|
|精度|ε=.05/.01/.005/.001/.0005、軸各ε/√2、α各.025、既存Hoeffding十分N、P=0|
|数値allowance|既存経験的64×roundoff proxy・safety10・floor1e-12、u/headroom≤.01。256次元のproxyが支配して厳しい精度でN/Gがnullになる可能性を保持し、閾値緩和しない|
|MC|各PR256 paired trajectories固定、PCG64 v2抽選順、SHA固定namespace。増標本・adaptiveなし。名目SE/t95%だけで、希少event/母平均精度は保証しない|
|compiler|Qiskit1.3.0/Nature0.7.2、rz/sx/x/cx、opt1、seed17、topologyなし、full ordinary-control Hadamard両軸、全位相保持|
|数値gate|旧H2閾値を維持。H operator1e-9、state residual1e-10、state norm1e-12、reference/法則平均1e-10、trajectory/compiled probability1e-9等、JSONに完全列挙|
|SOURCE/入力|将来のclean実装commitと全code/test/契約/承認/environmentをseal。入力receipt後に実rank・prefix・hashを固定し、4case release前に照合|
|one-shot/STOP|入力生成と4caseを1つの将来一括承認単位にする。新grant/固定claim directory、case別1回、失敗peer停止、retry/resumeなし、旧grant/STOP保全|
|資源・証拠|空き資源に応じcase別process並列、immutable入力共有・同一回路cacheで重複処理削減。標本抽選順・gate列は変更しない。全event/価格join・SOURCE・ログ・exit・STOPを保存|

これは比較可能な技術pilotの案で、最良候補やPR優位を狙う探索ではない。
PauliとDFのL_Dは異なる演算子単位を数える。実際のラベル/indexが未生成なので数値L_Dは未確定だが、
その決定規則は固定案として具体化した。生成後に結果を見てprefixを選び直さない。
将来の一括承認前には、H4専用molecular runnerとdimension gate、production samplerが返したeventを再抽選せず保存するhook、
その人工同値性試験、clean SOURCE/環境sealと新grantが必要である。今回のCLIには科学dispatcherを実装していない。

## 人工・保存値検証と成果物

[新supervisor](../src/trotterlib/isolated_qiskit_supervisor.py)、
[保存値診断](../src/trotterlib/saved_rte_cost_diagnostics.py)、
[H4 draft](../src/trotterlib/non_df_h4_readiness.py)、
[8 qubit人工系](../src/trotterlib/non_df_synthetic_dimension_preflight.py)、
[runner](../scripts/run_non_df_h4_readiness.py)、
[異常系test](../tests/test_isolated_qiskit_supervisor.py)、
[保存値/draft test](../tests/test_non_df_h4_readiness.py)、
[artifact](../artifacts/non_df_h4_readiness/2026-10-23/)を一組として読む。

開発初回は25 PASS/2 FAIL。人工nonzero workerの終了処理と孫worker起動より試験deadlineが先に来たため、
非ゼロ終了試験は終了を観測できる時間を確保し、子孫回収試験はfamily生成確認後のSTOP要求へ変更した。
開発27 PASS、追加テスト41 PASSを保存。最終のsource固定検証とは区別する。
人工回路の小さい4方式で逐次/4process同時の意味・seed・RZ数一致を確認し、8 system/9 wrapper qubitの人工4方式でも
dense/sparse信号・法則平均・両軸compiled probabilityを確認する。これをH4実分子検証完了とは扱わない。

最終SOURCE・回帰件数・除外理由・保全監査は結果取得後に追記する。全てローカル技術証拠で、CI/外部再現ではない。

## 最終検証・保全・終了

技術SOURCEは `8255fd0ed4e2e79b4394f8d1a91b67e9e20f6813`（科学実行SOURCEではない）。このcommitの173 source/契約/承認ファイルを固定してから
最終検証を行い、bytes一致を再照合した。最終結果は **370 PASS / skip1 / deselect5**、wall192.00秒。
新supervisor31、新draft/保存値11、従来安全回帰328（DF/PF/RTE/非DF・H2保存会計等）。
[JUnit](../artifacts/non_df_h4_readiness/2026-10-23/technical/final_tests.xml)と
[除外理由・module別結果への入口](../artifacts/non_df_h4_readiness/2026-10-23/validation_summary.json)を参照。
除外5件は内部でH2/H3のSCF/DF/JW生成等を実行する既存テスト。skip1件はPython3.11.0rc1固定fixtureと現3.12.3の差。
残りは人工/保存済み/mockのみで、opt-in分子生成禁止guardを使用した。

新監視基盤で人工4caseの逐次対4processを比較し、演算子・signal・normalization差≤1e-12、全seed/価格/probe RZ数は一致。
最終同時実行の共通重なりは小さい人工系0.293168秒、8 system/9 wrapper qubit人工系0.585509秒。
8 qubit系のdense/sparse reference差≤2.23e-16、法則平均差≤4.45e-16、全取得wrapperのcompiled probability差≤6.44e-14。
この有限人工scopeのPASSをH4実分子の一般保証としない。

最終の人工監視runは28件のSTOPを保存（正常9・意図的異常19）、記録されたowned identity39件が不在または再利用済み別identityであることを確認した。
別にSTOP書込み失敗1件は、STOP保存不能をstderrに明示し、worker回収をprocess.jsonで確認する試験をPASSした。
collision/無効scope等の消費前拒否では既存記録を変更せず、呼出し側ログへ拒否理由を残す。
正常ケース・no-ready・早期例外・非ゼロexit・hang・TERM無視・親例外・SIGINT/SIGTERM・親消失・STOP要求・heartbeat期限・
孫/setsid子孫の回収・log保存失敗・起動失敗・claim重複/参照先保護・出力競合・H4未承認拒否を検証した。

保全監査は旧tracked7,074件（artifact 4,885件）、旧manifest154 entry、旧H2のSOURCE266ファイルと
result inventory内の全保存hashを一致で確認した。元H2 worktreeはclean。旧C/N/G・図・科学raw・SOURCE・grant・ledger・marker・STOP不変。
新科学SCF/FCI/DF生成/H4 signal/MC/compile0、H2追加sampling/RNG replay/compile0、新母平均評価/再ランキング0。
[再実行可能な保存監査](../artifacts/non_df_h4_readiness/2026-10-23/verify_saved_readiness.py)は値とhashだけを読み、科学計算をしない。

H4は契約案・一括承認文案・SOURCE計画・検証gate・人工preflightまで準備完了。
[一括承認文案](../artifacts/non_df_h4_readiness/2026-10-23/H4_batch_approval_proposal.md)は未承認であり、工程別再承認を要求する設計ではない。
将来のH4専用runner/event-audit hookと人工同値性試験・clean SOURCE sealを完成した後、対象をまとめた明示承認が必要。
希少event情報取得案も定義の採用と実行承認が別途必要で、現在は実行しない。
ローカルcommitまで行い、push・詳細科学レビュー・次stageなしでmandatory STOP。
