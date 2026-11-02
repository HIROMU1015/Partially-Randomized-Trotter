# 第三γ0 DF：metadata修正後493-call完了・switch転移判定不成立

2026-10-29 JST。ユーザーの修正・再実行許可に基づくfresh run02。**493/493 COMPLETE、correctness FAIL 0**。32 trajectoryのraw／compiled full512、actual adjoint、complex X/Yはすべて数値gateを通過した。一方、第三DF holdoutのRZ MAEは切替モデルA **369.78698**、overlapモデルB **303.32856**であり、登録した`SWITCH_TRANSFER_NOT_SUPPORTED`となった。第三DFの条件付き総RZ点推定はM4比**2.67839倍**。この有限pilotは新cost-aware fitやη採択を支持するGOではない。

## 固定条件・修正差分

保存直鎖H4・1.00 Å・STO-3G、通常number CDF γ0 rank8、元frame順のprefix `L_D=2`、Nα=Nβ=2／sector36。Partial ordered S2、δ=h=0.1、s=1、r=128、K=0、m=0..12。CA=0.00159 Ha、DF reserve=CA/8、own-DF ground-state oracle、状態準備費用除外。元g補正κ、scalar一回、一般対称J、実直交Uを維持した。P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1、登録した初期qubit条件。登録のβ／failure配分・MPFR160・数値allowanceを変更していない。

第三候補は[Stage A保存γ0 export](../../artifacts/research_c_pr_aware_df/2026-10-27/run01/baseline/final.npz)であり、新しいfitではない。Export SHA256 `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02`、元fit SOURCE `e1085c95f5c01b68566e17d543ace5832cb94061`。元P-H4入力SHA256は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`で、Stage Aの入力snapshot hashと混同しない。

[旧run01](research_c_df_basis_third_execution_20261029.md)は141 attempts／140 COMPLETE／1 metadata KeyErrorで停止した。今回の[修正](research_c_df_basis_third_repair_preparation_20261029.md)は既存`tail.components`からB1/P4と同じcomponent recordを渡すtask-local adapterのみ。旧数値SOURCE112件は不変で、確率・符号・law・phase・compiler・モデル・判定を変更していない。旧結果・STOPを上書きせず、消費済み残枠やresumeとして扱わない。

新4＋既存42＝**46人工・回帰PASS（11.42秒）**。新adapter出力をmockしないB2 `reference_context/law_gate`へ接続し、旧欠損の再現、確率改変拒否、例外時binding復元、科学条件不変を確認した。[技術検証原本](../../artifacts/research_c_df_basis_third/2026-10-29/technical_repair_v2/verification.json)。

## 科学SOURCE・来歴

- 実行SOURCE commit：`30436b142d7fa79c841b5977d1b4c0fbfbd1bb4c`。Branch：`codex/research-c-basis-third-gamma0-repair-20261029`。
- [修正契約](research_c_df_basis_third_contract_repair_v2_20261029.json)：SHA256 `878f87d8769621a491bb99dc5f8880a9f93dcb7f58d4749f7652c6bda407846b`。
- [SOURCE115 manifest](../../artifacts/research_c_df_basis_third/2026-10-29/technical_repair_v2/SOURCE.json)：SHA256 `9790f3f1342dd9fa993b08e0985ebadf9c4103c914a1d35451b4c17a658bd006`。
- [Raw run02](../../artifacts/research_c_df_basis_third/2026-10-29/run02/)／[summary](../../artifacts/research_c_df_basis_third/2026-10-29/execution_summary_v2.json)／[postflight](../../artifacts/research_c_df_basis_third/2026-10-29/execution_postflight_v2.json)。

SOURCE115・saved input142・旧保護1725ファイルを前後でhash照合し、旧SOURCE112と旧run01の不変を確認した。契約の歴史的metadataを変更せず、承認証明書・新grantは作っていない。Seedは従来のSOURCE digest規則を維持するため新SOURCEに結合される。旧run01ではtrajectoryを生成しておらず、観測価格によるseed選択はない。

## 正しさ・PF/RTE

| 検証 | 結果・範囲 |
|---|---|
| Sector36一般K対独立CAR | base＋8 frame PASS |
| Full256対sector36 | 9因子＋sum PASS |
| Full-Fock tail対signed law＋identity | PASS、max-entry 1.44329e-15、allowance1e-10 |
| Own-ground／alias | PASS、E=-2.1663874524547952 Ha、残差2.82652e-15、alias margin1.61799 |
| PF／補正RTE登録十分条件 | 全13 round POTENTIALLY_FEASIBLE |
| Raw full512対独立reference | 32/32 PASS、最大max-entry 4.61868e-14 |
| Compiled full512対reference／raw | 32/32 PASS、最大4.11096e-12／4.10974e-12 |
| Actual adjoint | raw／compiled各32/32 PASS、復帰最大9.12604e-14／1.24808e-13 |
| Complex X/Y | raw／compiled各32/32 PASS、signal誤差最大2.06847e-14／1.21116e-13 |
| 保存head／X／Y shell | 3件PASS、登録identityとancilla条件を照合 |

上の最大値は既存検証結果の表示要約。Operator gateの登録allowance1e-10、signal2e-10を維持した。32件それぞれ全controlled512の位相込み数値比較であり、有限state probeだけの代用ではない。ただしbinary64の数値残差を厳密enclosureや全trajectory synthesis保証へ昇格させない。`full_law_gate`は生成子のsigned-tail同値性であり、全lawの回路合成保証ではない。

保存DF error＋guard **1.0297636237465247e-8 Ha**は予算内で、DF誤差を再計算していない。λ_R=**2.2097618367788057 Ha**、P_same=0.36641952146278217、期待basis切替80.46472077422668、期待runs81.46472077422666。全13 PF/RTEセルは旧run01の完了prefixと一致した。[13 round表](../../artifacts/research_c_df_basis_third/2026-10-29/presentation_v2/rounds13.csv)にはPF operator誤差・補正RTE operator誤差・attenuation・軸別shotsを保存。m=12ではそれぞれ0.07162597413905833、0.0790251656975318、0.45781668969684985。

## 第三DFへの予測移送

M4 Proposed／tail-P4の保存64行だけで、第三J/U・価格を読む前の最初の8 callsに共通interceptの6指標モデルを固定した。Mean、A_switch、B_overlapを同じ第三32行へ適用し、再訓練・第三intercept・標準化・hyperparameter探索・旧C復活は行っていない。

| RZモデル | 第三DF MAE | RMSE | 予測平均 | 負予測 |
|---|---:|---:|---:|---:|
| Global mean | 2892.23438 | 3091.08448 | 14561.95313 | 0 |
| A：basis switches | 369.78698 | 451.27540 | 17203.99033 | 0 |
| B：overlap path | 303.32856 | 374.71385 | 17425.06953 | 0 |

観測平均17454.1875。Aはmeanを改善したが、Bより主指標MAEが小さいという登録条件を満たさず、**SWITCH_TRANSFER_NOT_SUPPORTED**。主判定を別指標や事後モデルで救済しない。全6指標×3モデルのMAE/RMSE・bias・予測は[holdout18](../../artifacts/research_c_df_basis_third/2026-10-29/presentation_v2/holdout18.csv)、各32予測と特徴量・観測価格は[samples32](../../artifacts/research_c_df_basis_third/2026-10-29/presentation_v2/samples32.csv)。

これは同一H4系列の静的DF3個、追加holdout1個の記述的診断であり、32 trajectoryを32個の独立DFと数えない。Head/basis/係数差の因果寄与・新DF母集団への汎化は未確立。登録Aの不成立からbasis-aware DF全体を否定せず、旧3特徴Cの不適格も覆さない。

## Native費用と条件付き資源

登録32 trajectoryを一度だけsampleし、それぞれraw／compiled／両actual adjointの**128 QPY**を保存した。各coherent occurrenceは独立trajectoryとするlawを維持し、単一sampleの反復として費用を過小評価しない。Headの相対Gaussianとtailのexact-key共有は既存規則のままで、head/tail境界を越えた新融合はない。

| P0 kernel指標 | 標本平均 | 記述的SE | M4比 | tail-P4比 |
|---|---:|---:|---:|---:|
| RZ count | 17454.1875 | 195.9084 | 1.239958 | 1.159942 |
| RZ depth | 5355.5 | 58.2772 | 1.233338 | 1.123129 |
| CX count | 6613.75 | 61.7377 | 1.185379 | 1.132782 |
| CX depth | 3947.6875 | 31.2228 | 1.152652 | 1.092634 |
| Total depth | 11794.21875 | 127.4383 | 1.231832 | 1.124705 |
| Circuit size | 34881.6875 | 399.3560 | 1.245133 | 1.163803 |

比較は同じ保存H4/CDF8/LD2/δ=h=.1/s1/r128/K0・P0・状態準備除外のM4 Proposedとtail-P4に限定する。これは標本平均の差で、母平均差や保証付き優位ではない。RZ標本範囲15188..20567を全law supportとみなさない。[全6指標・SE・観測範囲](../../artifacts/research_c_df_basis_third/2026-10-29/presentation_v2/calibration6.csv)。

| 仮定lane | X/Y合計shots | 条件付き総RZ | M4比 | tail-P4比 |
|---|---:|---:|---:|---:|
| ideal_exact_native | 21754 | 627057959562.5 | 2.678387 | 2.276714 |
| assumed_raw_signal_budget | 21846 | 629935457235.75 | 2.678803 | 2.276701 |

登録したserial accounting `W×kernel平均＋S`を適用した。Ideal laneのW=35925928、S_RZ=76139。記述的校正SEの加重は約7.03819e9 RZであり、同時保証区間ではない。全指標・両laneの[24比較](../../artifacts/research_c_df_basis_third/2026-10-29/presentation_v2/conditional_comparisons24.csv)と[24損益分岐](../../artifacts/research_c_df_basis_third/2026-10-29/presentation_v2/breakeven24.csv)も保存した。

両仮定laneは登録上`CONDITIONAL_CA_PASS`、条件付きenergy bound=0.0012783276208742716 Ha、統計failure上限0.01。一方、取得証拠だけのlaneは**UNQUALIFIED_ROUNDS、shots=null**。未知synthesis/law誤差を実際に0と認定していない。実装law差・全law合成誤差・cost support・actual CA・母平均優位はいずれもnull／未資格であり、この総RZを保証付きCA達成費用と呼ばない。実quantum shotsをdrawしたわけでもない。

## Call・資源・保全とSTOP

| 段階 | COMPLETE |
|---|---:|
| T：旧2DFモデル固定 | 8 |
| A：第三DF reference | 64 |
| RPE | 48 |
| Price／GO | 8 |
| B：native32＋特徴量 | 357 |
| C：6指標holdout＋資格 | 7 |
| 最終summary | 1 |
| **合計** | **493／上限493** |

[Journal](../../artifacts/research_c_df_basis_third/2026-10-29/run02/science_calls.jsonl)と[STOP](../../artifacts/research_c_df_basis_third/2026-10-29/run02/STOP.json)。Run02内retry/resume=0、新fit／optimizer／rank探索=0。技術relaunch=1。旧run01の141 attemptsを保全し、2 run累計634 attempts（633 COMPLETE＋旧technical FAIL1）を隠さない。

実行前32 affinity CPU・空きRAM約52.5 GiB・pressure0を確認。既存sector13/native32並列、worker BLAS調整、Qiskit build/compile/adjoint排他を再利用した。CPU/RAM/worker/wall上限を追加せず、task-local cache以外の共有環境・他jobを変更していない。[GNU time実測](../../artifacts/research_c_df_basis_third/2026-10-29/process_time_v2.txt)：wall **20分28.97秒**、user14273.11秒、system215.03秒、max RSS3376308 KiB、swap0、exit0。異なる旧runとの速度倍率は主張しない。

Raw395ファイル（197551105 bytes）、SOURCE・入力・旧保護原本をpostflightで照合。表示CSVは登録計算済み値の転記で、追加モデルfit・予測・PF・価格計算は行っていない。SOURCE実装と結果のcommitを区別し、commit/push後にremote SHA/treeと成果物を照合する。

**ユーザー／研究C**の次の判断は、このA/Bの負結果とλ・減衰・費用の結果を踏まえ、追加の最小実験が必要か、まず設計を見直すかを決めること。今回からswitch重みη、新目的、新fit、別DFや別分子の実行を採択しない。**Codex**は別の具体的依頼と必要な科学実行許可の後だけ次作業を行う。詳細科学レビューも別許可とする。

公開照合後は**mandatory STOP**。消費済みrun02を再起動・resumeせず、追加科学計算へ自動進行しない。
