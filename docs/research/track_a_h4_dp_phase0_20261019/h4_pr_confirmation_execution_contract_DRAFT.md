# H4 PR独立確認 実行契約 DRAFT

`DRAFT_UNAUTHORIZED`。`PENDING_D_RESULT`、`PENDING_S_STATS_APPROVAL`、`PENDING_FINAL_N`。
本契約は条件付き実行templateであり、現時点では起動できない。D終了だけでPを起動しない。

## 固定条件と依存入力

H4/1.00 Å/STO-3G/legacy固定DF12/8 system qubits/T=.8/P=0。
保存Hamiltonian/state/reference/fragment順序・scalar一回適用・元finite-RTE lawを維持する。
native symmetric-directional、元二次partial PF、元component分布・normalization・q/r/K、
Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/no backend/coupling。
原lawを条件付きに変更したり、scalar phaseでreferenceを回転させたりしない。

候補集合・endpointはP_registration.jsonの9候補（主7＋境界診断2）から拡張しない。
各εのPFと正確なGはD終了受入後、design文書の最小費用規則で決める。
候補別nは別承認のS_STATS完了後、事前設計式で最終固定する。
P承認前にD result SHA、固定PF/shot/G、探索方向、各n、科学call上限、全seedとcollision receiptをsealする。

## seedと標本来歴

新namespace：`H4_PR_INDEPENDENT_CONFIRMATION_V1|SOURCE|sealed_plan_digest|candidate_id|replica`。
sealed_plan_digestはseed一覧を付ける前の科学条件・候補・固定n・shot・PFを含むplanのdigestとし、自己参照を避ける。
masterはcanonical JSONのSHA256先頭64bit。元direct_master_to_step_seeds_v1を維持し、
q=1はmaster、q>1はRandom(master)による元step seed列を使う。
候補間は独立stream、CRNなし。cos/sinだけは同じtrajectory/event列を共有する。
seed namespace/ledgerはprivateに保存し、公開projectionへseed一覧を含めない。

旧H4 n32と新A128軌道、元seed除外台帳の参照を取得済み。Phase 0で新確認seed生成は0。
最終sealでは旧探索、旧n32、geometry全系列、条件付きpilot、T/DF政策/その他既存標本と、
master・派生step seedの衝突を照合する。現在の旧台帳だけで全geometryまで網羅したとは主張しない。
不足inventoryをmetadataとして取得し、source/scopeを記録する。未完了ならP gateは拒否する。
seed衝突0を数学的独立性の証明とは呼ばない。

探索n8、旧n32、条件付き標本、今回確認を別cohortとして保持する。
新cohortは`INDEPENDENT_H4_CONFIRMATION`。元EXPLORATION_N8出力の名称変更による流用をしない。

## 実行adapterと予算の固定手順

新runnerのP branchは元SOURCE31ef62a1…のpr_cost_nativeのsteps/control/load_contextを再利用し、
新namespace・固定n・独立cohort・paired記録を新adapterで管理する。
元preparation archiveがないH4 p2/p3は、保存入力から元builderを再構成する既存方針を維持する。
SCF/DF/state/reference solver、signal再計算、shot再計算は0。新しい構築政策を採用しない。

`N=Σ_j n_j`として科学callを以下に固定する。数値予算はn未確定のためnull。

|call|予算式|
|---|---|
|無条件trajectory/native evolution|各N|
|測定wrapper compile|2N|
|occurrence|Σ q_j n_j|
|RTE event|Σ q_j r_j n_j|
|native control|最大4×9=36|
|独立event oracle|最大9|
|保存shotによる主・診断会計|1 pass、14 endpoint|
|新PF compile/SCF/DF/state/reference/signal/shot再算出/fit|0|

各候補のreplica0でcontrolと独立event oracleを通した後、その候補集合の残りを取得する。
controlの成功/失敗だけを科学的点順位から分離して扱い、途中のGやSEを停止判定に使わない。
全固定nを1回取得し、悪い軌道・希少event・外れ値を捨てず、追加n・候補変更・refit・retryをしない。
同じevolutionを両軸で使い、軸別fingerprint/compiler/input/seed/event_digestを保存する。
各callの予約・実行結果・失敗/未確認を専用ledgerへ分離して記録し、旧ledgerへ書かない。

## 保存schemaと受入

trajectory：candidate/replica/master/step seed、元distribution、event列、event_digest、
normalization照合、preparation hash、cohort、conditional=false、希少層分類。
軸：RZ/CX/depth/size、fingerprint、compiler hash、同じmaster/event_digest。
endpoint：固定PFと保存shot、exact G平均、sample SD/paired SE、点方向、判定、
同一候補内のε間共分散、population_mean_certified=false。

SOURCE、実input、compiler/profile/runtime、preparation、law、scalar、paired、native終了、
call budget、固定n、全回収と最終SHAを受入条件とする。欠測を0補完しない。
technical failureはTECHNICAL_STOP。点順位非再現をtechnical failureへ読み替えない。
母平均保証、全PR domain最良、全決定論外への優位、energy/QPE/RPE/FT資源の主張は行わない。

候補/trajectoryをprocess並列化し、準備・hash/回路構築の既存cacheを再利用する。
worker数はfresh資源で決め、旧n48/worker数/強制wall-memory capは移送しない。
cache/tempは今回output内、共有設定・affinity・他jobは変更しない。
一時的PSI/CPU競合/無関係worktree更新だけで科学workerを中断しない。
実OOM/own異常/科学不一致/入力破損ではown-scopeのSTOPとnative回収を行い、未確認をPASSにしない。

独立したP承認receiptとone-shot grantを得た後だけ起動する。D grant・統計解析grantの流用は禁止。
終了監査と結果保存・必要な許可済み公開手続きの後mandatory STOP。結果の符号で次実験へ進まない。

次の担当：CodexがDと承認済みS_STATSの後に契約を最終固定し、ユーザーがPを別途承認する。
