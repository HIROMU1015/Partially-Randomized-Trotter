# O1 M1：固定C0逐次runのpartial保全と並列切替

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**ユーザー許可による並列切替のため旧逐次runを計画停止した。価格は16/21 point完了、全horizon比較は未成立。科学FAILによる停止ではない。** 元の対象範囲確認に続く明示実行許可を受け、別worktree・branchで実施した。以下はこの固定条件の限定pilotのpartial記録であり、UWCに対する新規性・優位性の判定ではない。

## SOURCE・入力・実装

- 固定library commit：`8a13539e51bb113e82cb86ddb608e2e6e27786ca`。既存matched adapter、Gaussian/native回路、finite-RTE信号validator、resource bridge、実半径shot accounting、full no-prep価格provider、強DF-S2を再利用した。固定libraryはbyte照合し変更していない。
- 新しい[薄いdriver](../../../scripts/run_o1_m1_c0_pilot.py)を追加し、q昇順の信号検証→baseline検証→価格という順序へ接続した。不適格が分かった段階で条件変更・追加実験へ進まず停止する。
- input：commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。旧pathを変更せず専用`/tmp`へ取得し、hash確認後にdecodeした。
- [provenance](../../../artifacts/track_a_o1_m1_c0_science_20261106/run01/provenance.json)と[集約結果](../../../artifacts/track_a_o1_m1_c0_science_20261106/run01/result.json)に、入力・関係SOURCE hash・runtime・取得済み結果を保存する。

## 固定C0条件

H4-A、linear H4、1 Å、STO-3G、保存rank10、元Hの`state`、共通`T_eff/V_square`（LD0）、faithful tail identity、全signed factorと核scalarを維持した。元Hの厳密基底状態を仮定し、状態準備コストを除外した。`DF_state`への交換や新しいstate solveはない。

`epsilon_E=0.1 Ha`、成功条件95%、`delta=.05`、`m=0..6`（q=1,2,4,8,16,32,64）、r=2、K=2、`beta_RPE=.30`、energy区間`[-10,0] Ha`。PF/RTE/stat/model/実装の位相配分は`.06/.03/.15/.03/.01`、reserve `.02`、各軸の失敗配分は`.05/(2*7)`を維持した。

compilerはQiskit 1.3.0、rz/sx/x/cx、optimization 1、seed 7、backend/coupling mapなし。主指標CX、副指標RZ・depth等。PR価格MCは各arm/round32 trajectory、seed `2026110600+100*arm_index+m`。主baselineは強決定論DF-S2の4 substeps、relative fusionと固定spin policyである。旧F2 prep込み価格・旧proxy・旧短round結果を補完に使わない。

数値allowanceは[固定C0報告](../track_a_o1_m1_cost_validation_preparation_20261106/REPORT.md)の式を維持した。元stateに対するPF信号半径を実際に評価し、半径1を仮定しない。価格MCのsample SEと量子測定失敗予算を分け、SEをCIや数学的証明へ格上げしない。

## 結果

PRは全7roundで固定operator/controlled-signal検査とC0の位相・半径条件を通過した。必要shot数は既存式による条件付き算出であり、量子測定shotを実行した数ではない。

| q | P attenuation | DF attenuation | P shot/axis | DF shot/axis | PF phase bound | RTE phase bound |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.679025234 | 0.685460448 | 2459 | 2413 | 3.50256345e-06 | 1.05774529e-05 |
| 2 | 0.461075268 | 0.469856025 | 5332 | 5135 | 6.77541595e-06 | 2.09251251e-05 |
| 4 | 0.212590403 | 0.220764685 | 25082 | 23259 | 1.33211188e-05 | 4.16202562e-05 |
| 8 | 0.0451946795 | 0.048737046 | 555014 | 477265 | 2.64124701e-05 | 8.30096128e-05 |
| 16 | 0.00204255906 | 0.00237529965 | 271773370 | 200964493 | 5.25946591e-05 | 0.00016578439 |
| 32 | 4.1720475e-06 | 5.64204844e-06 | 65164976569615 | 35631910593242 | 0.000104956369 | 0.000331317866 |
| 64 | 1.74059803e-11 | 3.18327106e-11 | 3746527790271109800132608 | 1120156469303095630757888 | 0.000209680227 | 0.000662336937 |

cosine/sineのshot数は各arm/roundで同じである。JSONには整数値、全normalization、PF半径下界、観測半径下界、各誤差と判定を保存した。PF/finite-RTE誤差が条件内でも、減衰係数Aはqの増大とともに小さくなり必要shot数が増える。元stateをDF/PF固有状態とは仮定していない。

強DF-S2も全7roundで元stateのC0条件を通過した。q=1のfull controlled operator（phase alignmentなし）の最大差は`1.8962726262663195e-14`。q>1は各qの実回路に対する独立S2参照とのcontrolled state-actionを検証し、full operatorの全成分を検査したとは主張しない。

| q | PF phase bound | controlled state-action L2差 | shot/axis |
|---|---:|---:|---:|
| 1 | 2.17184266e-06 | 1.86565028e-14 | 1134 |
| 2 | 4.11397613e-06 | 3.52045914e-14 | 1134 |
| 4 | 7.99825315e-06 | 6.8190993e-14 | 1134 |
| 8 | 1.57668473e-05 | 1.38324411e-13 | 1134 |
| 16 | 3.1304195e-05 | 2.71196297e-13 | 1134 |
| 32 | 6.2379525e-05 | 5.21556074e-13 | 1134 |
| 64 | 0.000124532771 | 9.34672592e-13 | 1134 |

baselineはA=B=1で、各qの実際のPF半径下界を使った。元H stateのRayleigh energyは`-2.1663874486347643 Ha`で固定区間内、元H eigen residual L2は`1.916853751084478e-15`。state solve・DF_stateへの置換はない。

価格は**16/21 point**を完了した。q<=16の3armとq32 Pauliである。未完了はDF q32、強DF-PF q32、P/DF/強DF-PF q64の5点。完了済みPR pointはいずれも固定CX relative sample SE 10%以下を通過した。未取得価格はnull/未取得のままで、全round値や優位性は主張しない。

| q | arm | CX mean cosine | sample SE cosine | CX mean sine | sample SE sine |
|---|---|---:|---:|---:|---:|
| 1 | P | 197.59375 | 9.29043683 | 197.59375 | 9.29043683 |
| 1 | DF | 256.4375 | 10.5601138 | 256.4375 | 10.5601138 |
| 1 | strong_DF_PF | 34208 | N/A（決定論） | 34208 | N/A（決定論） |
| 2 | P | 309.125 | 11.6753592 | 309.125 | 11.6753592 |
| 2 | DF | 471.28125 | 16.3456388 | 471.28125 | 16.3456388 |
| 2 | strong_DF_PF | 68320 | N/A（決定論） | 68320 | N/A（決定論） |
| 4 | P | 520.28125 | 16.2644622 | 520.28125 | 16.2644622 |
| 4 | DF | 847.34375 | 22.1660526 | 847.34375 | 22.1660526 |
| 4 | strong_DF_PF | 136544 | N/A（決定論） | 136544 | N/A（決定論） |
| 8 | P | 1053.71875 | 19.5067831 | 1053.71875 | 19.5067831 |
| 8 | DF | 1649.625 | 35.8675248 | 1649.625 | 35.8675248 |
| 8 | strong_DF_PF | 272992 | N/A（決定論） | 272992 | N/A（決定論） |
| 16 | P | 1998.59375 | 29.8774668 | 1998.59375 | 29.8774668 |
| 16 | DF | 3228.09375 | 49.8159386 | 3228.09375 | 49.8159386 |
| 16 | strong_DF_PF | 545888 | N/A（決定論） | 545888 | N/A（決定論） |
| 32 | P | 3858.6875 | 42.9376996 | 3858.6875 | 42.9376996 |

RZ/count/depth/sizeの平均・軸別sample SEも[元resultのpriced_rounds](../../../artifacts/track_a_o1_m1_c0_science_20261106/run01/result.json)へ全文保存した。depthは1 shot回路の指標でありhardware wall-clock timeではない。価格CIは算出していない。報告用[6指標summary](../../../artifacts/track_a_o1_m1_c0_science_20261106/run01/mean_cost_summary.json)は未完了を`UNASSESSED`とし、全round総費用を`null`に保つ。

## ユーザー許可による計画切替と旧run STOP

ユーザーが並列版への切替を明示許可したため、owner/args/startを照合した自分のPython PID1848166だけへSIGINTを送った。Pythonとshell1848162の消失、exec session51382のexit code130を確認した。**C0の科学FAILによる停止ではない。**

元resultの`status=RUNNING`は最後のcheckpointとしてbyte不変更で保存し、停止事情と完了16点・未完了5点を別[transition.json](../../../artifacts/track_a_o1_m1_c0_science_20261106/run01/transition.json)に記録した。result SHA256は`1242e030c7a28d6476aed9c699b2abe3acd41c51d2281329b9ccc8c888786cbe`、実行driver SHA256は`0b6da463853bc3b6ba881cae636e6169ef751e3311e5dcc8f92d1cb42c24f8e3`。科学raw、driver、provenance、signal_m*.json、入力/tmpコピーは変更していない。

停止時の[ログ](../../../artifacts/track_a_o1_m1_c0_science_20261106/run01/execution.log)では、`canonical_qiskit_circuit_fingerprint`がQiskitの`XXPlusYYGate.definition`を展開している箇所にKeyboardInterruptが記録された。これは停止時の1 stackで、全経過時間の定量profileではない。

旧runは**mandatory STOP**。旧runでは以後追加SCI・再実行を行っていない。未完了pointを同じ固定seed・32 trajectoryで測り直す新parallel実行は、このbranchと別のworktree/runで継続する。継続SOURCEは`93d6b49b77c619e0e12e669b5a5884ffea343ff0`。完了済み16点は再計算せず引き継ぐ。旧runのこの報告と未完了の全horizon比較を混同しない。

## 人工テスト・限界・次の担当

新driverのallowance/domain/固定条件の人工テストは**5 passed in 0.12s**、報告用6指標集計の人工テストは**6 passed in 0.11s**。発明したスカラーのみで、許容値の差へのfitやdomain外clampがないことを確認した。実H4の許容値には人工atolを転用していない。

旧SOURCE・入力・raw・結果・grant・marker・STOPは保持する。科学終了後は結果の公開・照合だけを行い、追加科学計算・条件変更・新fit・rank探索・UWC判定へ進まない。


次の担当：切替記録作成後に担当分担が更新され、**Codex（m1_c0_science agent）**が別SOURCE・別worktreeで許可された未完了5点の並列継続・全結果公開を担当する。起動時に16行carry、prior SHA一致、SOURCE/cohort一致を確認した（新master PID2099882、exec session10518）。`transition.json`のroot担当表記は記録作成時の計画を保存したものである。本branchの旧runはpartial保全・公開照合でSTOPし、新条件・追加実験・新しい詳細科学レビューへ進まない。
