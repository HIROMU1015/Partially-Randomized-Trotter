# R可変PRの詳細N P1結果（2026-11-11 JST）

**承認されたP1の12候補を一回評価し、全12件でideal条件付きround PASSを得た。** 各軸の十分Nは、H4 m6でR4の9,638からR128の766・R256の735へ、H6 m8でR16の1,015からR128の735・R256の720へ変わった。**Cは未評価なので総費用や最適Rを判定しない。** P2/P3・m12の新評価には進んでいない。

科学実行SOURCEは **`a1a79213a9b5331145e20e8778a3be9d9fc4e497`**。固定計画は `d33380a8c2fe68d0be1f2c5faaf7fc3b3a956b58` の[P1](../pr_r_detailed_n_preparation_20261111/REPORT.md)。準備原本の未承認/未実行statusと過去STOPはそのまま保持した。今回のユーザーチャットによるP1実行許可の範囲だけを実行し、追加grantや承認証明書を作っていない。

[科学原結果](run01/result.json)／[B向けN_HANDOFF](run01/N_HANDOFF.json)／[全signal診断CSV](run01/SIGNAL_DIAGNOSTICS.csv)／[N・位相・半径CSV](run01/ROUND_N.csv)／[HANDOFF](HANDOFF.md)／[保全・照合結果](CHECKS.json)／[SOURCE索引](SOURCE_INDEX.json)。

## 科学条件と方法

Linear H4/H6・隣接1.00 Å・STO-3G、保存tol-only 1e−8 DF10/DF19、weighted-Hermitian completion・fragment/signed piece順・scalar/identity一回会計、H/state/snapshot bindingを維持した。sectorはH4 Nα=Nβ=2・dimension36、H6 Nα=Nβ=3・dimension400。保存DF-Hの同sector ground energyを対象とし、保存数値stateを使う条件付き評価、P=0。元分子Hamiltonianへの誤差保証へ拡張しない。

H4 PRはS2/δ=.1/prefix1/K2、H6 PRは4th(m5_best)/δ=1.6/prefix7/K2。H4 PF B0/p7、H6 PF Full19、H6 m0..3の保存別構成は不変・新計算0。R_outerはwhole-outer値で、H6の11 signed pieceへ既存min-one/largest-remainder・tie順で配分する。pieceごとにRを重複適用せず、qPFとqRPEも区別する。

h=.1、13round m0..12、εE=.00159360010199040 Ha、各方式αtotal=.05、各軸α=.05/26、βtotal/stat=.4/.2、保存prior/decoder・無energy shiftを保持。今回の複数Rは代替候補の評価で、全候補を一つのRPE scheduleとして合算していない。有限時間評価のalpha_axis=.025を持ち込まない。E_stateはreference整合性の診断にだけ使い、prior/branchの決定に使わない。

数値kernelは既存 `pr_r_dependent_n.RBatch/analyze`、`h6_df19_rpe_d2` finite action/reference、D1 geometry/Hoeffding、整数ceil検査を変更せず再利用した。B `ac3e1ce87` のRMenuBatch/_law/bound_context_factoryもAST本体をそのまま使い、C catalog/optimizerはloadしない。新しいRPE solver、power backend、Taylor近似の変更、cacheアルゴリズム変更はない。

## 実行範囲とSOURCE再利用

| 処理 | 実施数 | 対象 |
|---|---:|---|
| 保存baseline signalの同policy scalar再評価 | 3 | H4 m0/R4、m6/R4、H6 m8/R16 |
| 新finite mean・独立corrected/raw伝播 | 9 | 新R8件＋H6 m4/R16 baseline1件 |
| PR exact-tail内部対照 | 9 | 各新meanの独立column |
| 新full-H reference action | 2 | H6 T1.6のexpm_multiply/expm各1 |
| 保存2経路reference再利用 | 3時点 | H4 T.1/6.4、H6 T25.6 |
| scalar phase/radius/十分N | 12候補・24軸 | 全P1候補 |
| trajectory/sampling/compile/C/G・科学retry | 0 | 対象外 |

H6 m4の保存D1には現行RBatchのpair/norm証拠が不足したため、計画どおりreferenceとbaselineを新評価した。他の時点で新referenceを暗黙生成していない。保存baseline3件は原signalを再利用し、旧Nを単に再ラベルしたものではない。4 baselineを同policyで通した後、新R8件を独立system並列・B group内並列で評価した。

保存referenceのSOURCEはH4 `89c91c692`（科学SOURCE90dcc3ed2）、H6 `f4c0bdabd`（科学SOURCEfca50d36f）。元二経路のcomplex値・差・binding/Tを照合して再利用した。古いreference record内のreference_state_actions=2は元実行の証拠で、今回の新actionへ再加算しない。H6 T1.6の新pairは今回SOURCEであり、差は保存値 **7.021666937153402e−16**。

## N・normalization・位相・半径

下表のAはnormalization attenuation、ρlowerはraw信号半径の下界、βdetは誤差disk由来の位相allowance。表示は保存値を丸めたもので、新しい式への再fitやBだけからのN外挿ではない。すべてX/Y同数だが、出力には両軸を独立fieldで保持した。

| 系 | m / T | qRPE / qPF | R | event数 | log B | A | ρlower | βdet [rad] | N_X / N_Y |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| H4 | 0 / .1 | 1 / 1 | 2 | 2 | .04048939179 | .9603193517 | .9603064774 | 1.34063222e−5 | **764 / 764** |
| H4 | 0 / .1 | 1 / 1 | 4 | 4 | .02042887934 | .9797783765 | .9797652414 | 1.34062156e−5 | **734 / 734** |
| H4 | 0 / .1 | 1 / 1 | 8 | 8 | .01023782586 | .9898144023 | .9898011326 | 1.34062091e−5 | **719 / 719** |
| H4 | 6 / 6.4 | 64 / 64 | 4 | 256 | 1.30744827765 | .2705094414 | .2702775418 | .000857270052 | **9638 / 9638** |
| H4 | 6 / 6.4 | 64 / 64 | 128 | 8192 | .04098249418 | .9598459327 | .9590230858 | .000857269849 | **766 / 766** |
| H4 | 6 / 6.4 | 64 / 64 | 256 | 16384 | .02049129300 | .9797172268 | .9788773447 | .000857270082 | **735 / 735** |
| H6 | 4 / 1.6 | 16 / 1 | 11 | 11 | .01676289819 | .9833768174 | .9833517467 | 2.54945225e−5 | **729 / 729** |
| H6 | 4 / 1.6 | 16 / 1 | 16 | 16 | .01138443249 | .9886801249 | .9886549191 | 2.54944325e−5 | **721 / 721** |
| H6 | 4 / 1.6 | 16 / 1 | 32 | 32 | .00534535907 | .9946689019 | .9946435434 | 2.54944125e−5 | **712 / 712** |
| H6 | 8 / 25.6 | 256 / 16 | 16 | 256 | .18215091991 | .8334755429 | .8331415714 | .000400697436 | **1015 / 1015** |
| H6 | 8 / 25.6 | 256 / 16 | 128 | 2048 | .02106165021 | .9791585974 | .9787662510 | .000400697451 | **735 / 735** |
| H6 | 8 / 25.6 | 256 / 16 | 256 | 4096 | .01051878564 | .9895363433 | .9891398386 | .000400697509 | **720 / 720** |

全12件で正半径、βdet+βstat≤.4、既存norm/closure/reference・整数gateが成立した。各signalのreference/corrected/raw/exact-tail complex値、signed bias、B/logB、corrected/raw disk、観測raw半径、u/logB margin、中間最大norm・closure/exact normを[全signal CSV](run01/SIGNAL_DIAGNOSTICS.csv)と原結果に保存した。raw=corrected/Bを事後定義せず、独立raw伝播を維持した。

allowanceは全候補で `R_AWARE_EMPIRICAL_ACTION_COUNT_V1`。既存stage式と観測中間norm・closure・pair差を使う経験的policyで、rigorousな誤差上界ではない。旧baseline Nと異なるpolicyの新R Nをそのまま比較していない。H4旧巨大整数の正本 `89c91c692:N_HANDOFF.json.shots` は変更・float変換しておらず、m12を再評価していない。

今回の範囲では、R増加によりnormalizationと必要Nが改善する有限例が得られた。corrected bias/位相diskも毎回評価しており、normalizationだけからPASS/Nを決めていない。この観測を全R・長時間m12・H4/H6全ラウンドへ外挿しない。回路event数も増えているため、N改善だけではGの削減や最適Rを意味しない。

**actual laneは全12件で未資格、N_X/N_Y=null。** 保存stateの厳密ground同定、actual implementation/law距離、経験的allowanceの厳密包含、全RPE成功・選択後保証は得ていない。ideal finite-lawの仮定とactualの未知誤差を区別し、未知を0にしてactual PASSへ昇格していない。

## 検証・資源・保全

[6人工/mockテスト](TECHNICAL_TESTS.txt)は実行前にPASS。実B groupingとmock leafで9 mean/3 saved scalar/1 new pair、context/reference共有、baseline FAIL・後続未実行、欠測pairのfallback禁止、fresh出力・SOURCE不一致を確認した。今回の作者自身による技術確認で、独立科学レビューではない。

科学実行はexit0、stderr空。[STDOUT](EXECUTION_STDOUT.txt)。保存wall **63.892730秒**はexecute内（preflightと後段保全照合・最終出力を除く）。新benchmarkや大R所要時間のfitには使っていない。既存CPU/RAM/仕事数による適応worker選択・プロセス内BLAS1、独立H4/H6とcandidateの並列化を利用し、独自CPU/RAM上限・共有設定変更・他process操作は0。

23 SOURCEと7入力hashを保全。A0の24件・旧N/保存signal/結果・grant/ledger/marker/KEEP/過去STOP、B/C worktree/sourceは変更していない。run01/progress.jsonはbaseline完了時点のcheckpointであり、完了状態はresult.jsonとexit0を正本とする。読取専用[VERIFY_RESULTS.py](VERIFY_RESULTS.py)で全12 ID/完全identity/law/allocation、同policy、N整数・軸field、JSON/CSV/HANDOFF、SOURCEと旧結果保全を照合してPASSした。再伝播・N再計算は行っていない。

## B/C接続と次段階

`run01/N_HANDOFF.json.shots` は既存common N record＋Bのlaw_binding/N_comparison_policy。Bは今回12件を公開SOURCE付きで取り込み、旧baselineの比較policy不足をこの範囲で解消できる。その他round/PFの保存在庫を置換せず、未評価R/価格不足・不確かさを保持する。

今回のC照合は固定計画のmetadataのまま。H6 m4/R16の保存有限cohortとR32のC Phase1予定セルを区別し、価格取得/予測/fitはしていない。H4 m6やH6 m8のR128/256は確認した小q価格domain外で、未知価格・全law期待値・cross-R/round covarianceを補完しない。Nへ価格標本数やBを再乗算しない。

**Codex B**：この公開12 N recordを既存pipelineへ取り込み、**Codex C**：自身の許可範囲で対応R/qの価格検証を行う。**ユーザー**：P2（H6 m12のR16/32/2048）やP3（H4 m12のR4/8/8192）、追加C/実Gの範囲を別途判断する。P2/P3の数百万〜数千万eventには既存[計算量・精度計画](../pr_r_detailed_n_preparation_20261111/REPORT.md)が残る。今回はcommit/push、remote SHA/tree・主要成果物照合後mandatory STOPし、次の科学計算・詳細科学レビューへ自動進行しない。
