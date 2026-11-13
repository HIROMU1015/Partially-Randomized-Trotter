# A → B/C：採用したH_eff手法と初回H4適用の境界

[REPORT](REPORT.md)／[未実行SCIENCE_PLAN](SCIENCE_PLAN.md)／[PLAN](PLAN.json)。実H4 step/eigensolve/保存signal再解析は0、科学実行は承認待ち。

## 既存APIと追加出力

`pf_effective_hamiltonian.evaluate(U,delta,state=...,operator=...,reference=...,branch=...,tolerances=...,reference_hamiltonian=H_full)`を再利用する。署名・既存fieldは維持し、元Hを渡す場合に既存の1回の`eigh`結果から次の情報を追加した。

| field | 意味 |
|---|---|
| `reference_comparison.eigenvectors/state_weights/energy_groups` | 元full DF-Hの固有basisと保存stateの支持。declared reference energyとは区別 |
| `reference_lowest_group_to_effective_groups_trace_overlap` | 元最低energy groupの射影と各effective groupの射影のtrace。縮退時に総和が1とは限らず、state確率ではない |
| `matched_effective_group` | 常にnull。対応先やground資格を自動選択しない |
| `diagnostics.reference_eigen_residual_fro/reference_eigenvector_gram_defect_fro/reference_weight_sum_error` | 元固有系の数値gate。不成立時UNKNOWN、残差は保存しスペクトル資格を付けない |
| `spectral_shift_diagnostics.minimum_branched_energy_minus_reference_matrix_minimum_ha` | 指定分枝のminimum同士の差。対応したRPE biasとは限らない |
| `declared_reference_minus_matrix_minimum_ha` | 保存proxyと今回の元行列minimumとの差。proxyは上書きしない |

`scientific_method_status=ADOPTED_MAIN_DETERMINISTIC_PF_ERROR_ROUTE`はユーザーによる評価方針の採用を示す。`status=EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE`は数値診断取得のみ。`physical_alias_qualified/ground_identification_qualified/numerical_error_certified=false`、`rpe_bias_ha/sufficient_N/global_RMSE_bound/G=null`を保持する。PR head/exact-tailのmethod statusは`NOT_ADOPTED_FOR_TOTAL_PR_ERROR`。

旧2f0 packetの`PROPOSED_NOT_ADOPTED`は当時の状態として保存する。旧packetのIDやstatusを新しい結果へ再ラベルしない。Bのcanonical identity生成・risk/shot・Cのbuilderは変更していない。

## 初回の接続先

Cの固定`c5466946...:BASELINE_FINITE_MANIFEST.json:candidates[0].configuration`、configuration ID=`f2d58f320ed0ebbeff8e6d4c0fbd5de14d6338b493c92015c966935201fd93ef`を参照する。全13round metadataはPLANへ転記したが、実検証の提案はm0/m12だけである。

Full PF S2/prefix10/Δ.1：元full Hとretained Hは同じtargetで、scalarは元constant一回。PRのL_D=1/2やtail λ、B0/B0Iには置換しない。legacy bindingの準備時刻T=.8をround Tへ書き換えず、round固有のcanonical T/q/identityを使う。B/Cの未確定科学statusや既存RMSE policyの資格を、この手法採用によって変更しない。

承認後の出力には、configuration/二時点candidate ID、H/state/snapshot/sector、元/H_eff全spectrum、state weights/近接group、分枝整数・cut余裕、δE診断、数値残差、二時点の直接PF/全spectrum signalと差、元/現実行SOURCE、全未資格fieldを含める。

## Bのriskとの会計

- 実PF raw signalをそのままBのdecoder/有限shot riskへ入力する経路は、state支持・PF位相等の効果を既に含む。同じPF効果をδEとして再加算しない。
- δEを独立した時間発展errorへ使う経路では、対応する固有state/branchと他spectral成分、比較する理想モデル、全round共通step、decoder/alias/loss、Minkowskiの各誤差確率変数を特定する必要がある。単一shiftを採用したことだけでこの契約は完成しない。
- Truncated PFではretained Hのenergy差に加えて元full-H targetとの差を区別する。PR headのshiftはPR全体のerrorではない。PR exact-tail unitary controlの共通step条件を満たす場合も別検証が必要。finite-K corrected/rawの非unitary operatorにはHermitian H_effを適用しない。

新4＋既存15の人工test PASSはAPI/数値実装の検証で、実H4の物理的資格ではない。ユーザーの有限scope承認後にAが取得し、Bが既存risk評価へ接続する。科学pilot・幅探索・N/C/Gには自動進行せずmandatory STOP。
