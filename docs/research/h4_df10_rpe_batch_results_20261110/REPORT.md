# H4 DF10 固定13-round N batch結果（2026-11-10 JST）

承認された固定範囲を一回実行した。保存m=3を再計算せず、m=4保存signalのscalar2候補と新規11時点22候補を評価し、全13ラウンドのPF/PR各軸Nが揃った。既存round gateは全て`CONDITIONAL_ROUND_PASS`。価格取得・予測、実N×C/G、最適化、sampling/回路build/compileは0。actual N・厳密誤差・全RPE成功・総費用順位は未資格である。公開照合後mandatory STOP。

## SOURCE・範囲と保存

- 科学SOURCE：`90dcc3ed20f30dfab6b22a68cc4adb9461485a98`。[固定準備REPORT](../h4_df10_rpe_batch_preparation_20261110/REPORT.md)・[PLAN](../h4_df10_rpe_batch_preparation_20261110/PLAN.json)の16 SOURCEを変更せず、`run_h4_df10_rpe_batch.py --execute`を一回実行した。
- 整数検証SOURCE：`0dc17b0329e1a8aa2737c86f259e049985317dc5`。後述の巨大Nの浮動小数点丸めだけを同一式・同一保存ε/αで検証した。signal/phase/radiusの再計算や候補変更はない。
- 実行時間：11.203793秒。新reference11時点×2経路=22 action、finite mean22候補、PR exact-tail内部対照11、保存m4 scalar2。m3再評価0。half-step8と候補22をそれぞれ8/22 workerで処理した（process内BLAS1、利用可能CPU/RAMの観測に基づく）。
- 原結果：[run01/result.json](run01/result.json)、[raw coverage](run01/COVERAGE.json)、個別candidate JSON26件。stdout/exit0、stderr空。原結果は整数検証後も不変。
- C担当への受渡し正本：[N_HANDOFF.json](N_HANDOFF.json)の`shots`。全signal・位相・半径・raw/受渡しNの転記：[ROUND_SUMMARY.csv](ROUND_SUMMARY.csv)。m3 CSVの空signal欄は旧N結果にsignalを含まないためであり、欠測を再計算で埋めていない。

Linear H4、1.00 Å、STO-3G、保存tol-only 1e−8 DF10・weighted completion・fragment順、8 mode、Nα=Nβ=2・sector dimension36、保存数値stateを固定した。対象は保存DF-Hの同sector基底エネルギー、P=0。Hamiltonian/state/snapshot bindingは準備PLANと一致する。

H4のS2、δ=.1、PF B0/prefix7、PR prefix1/R_outer4/K2、h=.1、m=0..12を維持した。q_RPEとq_PFは別fieldだがこのmenuでは双方2^m。PF B0はtailを捨て、保存constantだけを総Tで一回適用する。PRは保存tail scalar・identity・normalization規則と独立corrected/raw伝播を維持し、S2 allocation `[4]`、event数4q_PFである。

エネルギー精度0.00159360010199040 Ha、方式ごとαtotal=.05、各軸α=.05/26、βtotal/stat=.4/.2、prior `[−4.994249624417489, −2.0985459369977186]` Ha、shift0、既存decoderを変更していない。state energyは診断に限り、prior/枝選択へ使っていない。

## 全13ラウンドのN

X/Yは各roundで同じ。表のPR m10..12は表示用近似で、全桁はJSON/CSVの十進整数を使う。NはHoeffding十分条件の数値評価であり、actual minimum shotではない。

| m | T (a.u.) | q_PF | PF N/軸 | PR N/軸 |
|---:|---:|---:|---:|---:|
| 0 | 0.1 | 1 | 705 | 734 |
| 1 | 0.2 | 2 | 705 | 765 |
| 2 | 0.4 | 4 | 705 | 830 |
| 3 | 0.8 | 8 | 705 | 977 |
| 4 | 1.6 | 16 | 705 | 1355 |
| 5 | 3.2 | 32 | 705 | 2605 |
| 6 | 6.4 | 64 | 705 | 9638 |
| 7 | 12.8 | 128 | 706 | 131934 |
| 8 | 25.6 | 256 | 708 | 24723982 |
| 9 | 51.2 | 512 | 712 | 868266239829 |
| 10 | 102.4 | 1024 | 719 | 約1.07091027e+21 |
| 11 | 204.8 | 2048 | 733 | 約1.62959440e+39 |
| 12 | 409.6 | 4096 | 764 | 約3.77790860e+75 |

m3のPF705/PR977は旧科学SOURCE `311255aaac39dc1c3321108c296bde7dc93dd543`のideal/actual laneをそのままコピーした。m4はPF705/PR1355で、保存signal/raw/allowanceからの新scalar評価である。旧有限時間endpointのNやαaxis=.025は使用していない。m4の保存referenceは旧spectral経路で、独立経路の保存証拠は旧T=2に限られる。新たなT=1.6 reference照合が済んだとは扱わない。

## 巨大Nの整数丸め検証

旧共通helperはbinary64の式を`ceil`後にPython intへ変換する。PRのm10・11では、その整数が保存ε/αを使った高精度の同一Hoeffding式より僅かに小さかった。raw記録のPASSと整数を保全し、受渡しrecordだけを上方向へ補正した。

| PR round（各X/Y軸） | raw整数N | 受渡し整数N | 増分 |
|---|---|---|---|
| m10 | 1070910265913026543616 | 1070910265913026585642 | 42026 |
| m11 | 1629594403126324358597983899992778080256 | 1629594403126324540902444237287822208824 | 182304460337295044128568 |

相対不足はそれぞれ約3.92e−17、1.12e−16。新24候補の48軸をDecimal140/180桁で照合し、両精度でceilが一致した。受渡しNは`max(raw N, 高精度ceil)`とし、元Nを下げていない。m12は元Nが高精度ceilを僅かに上回るため不変。m3は整数算術も再評価していない。

これは**保存binary64 ε/αに対する整数丸め検査**であり、signal diskや数値allowanceのrigorousな証明ではない。元SOURCE/閾値/条件を変更した再実行はない。整数補助処理の人工4テストは[INTEGER_TESTS](INTEGER_TESTS.txt)でPASS、元adapterの人工13件はSOURCE不変の[保存記録](../h4_df10_rpe_batch_preparation_20261110/TEST_LOG.txt)を再利用した。

## 位相・半径・normalizationと資格

[CHECKS.json](CHECKS.json)、[検査・CSV転記SOURCE](VERIFY_RESULTS.py)に保存した。

- 新referenceの二経路差max 9.279812219411241e−12（gate1e−10）。新候補のexact-tail norm error max 2.332578574737454e−13、raw/corrected vector closure max 1.817376746641006e−15（gate各1e−9）。
- 最大βdetはPR m12の0.05488792699282713 rad。βstat=.2との和0.25488792699282714はβtotal=.4以下。PF m12はβdet=0.03963371644704053、raw半径下界0.9603766590529247、N764/軸。
- PR m12はlogB=83.67668976947293、B=2.1893975653862874e36、attenuation=4.567466483975762e−37、raw半径下界4.31689357748351e−37。位相gateが成立しても、固定R4のattenuationでNは約3.77790860e75/軸へ増える。PASSを実用的費用の証拠としない。
- corrected disk・u・logB margin・reference offset・保存residualの扱いは旧H4算術を維持した。経験的数値allowanceとrigorous上界を区別する。actual implementation/law誤差、厳密ground同定は不明のまま、actual Nは全てnull。
- 16 SOURCE、5保存artifact hash、3入力/contextファイル、旧m3 N lane、H6全13-round結果を保全した。科学retry0、他worktreeへの変更0、C/G0。詳細科学レビューや別AIによる独立レビューは実施していない。

## 価格担当との接続と次の作業

`N_HANDOFF.json.shots`は既存`pf_pr_rpe_pipeline`のN record形式で、candidate ID・m・q_RPE/q_PF・signed recipe・科学binding・configuration・compiler/control/full-wrapper profileを変更していない。共通plannerで26件を受入れ、missing_N/unqualified_Nは空、totalsはnull。Codex Cはこの`shots`と固定PLANのtaskを使い、既存C recordを接続できる。**raw resultの整数を巨大Nの受渡し正本として使わない**。

巨大Nをbinary64へ丸めるJSON処理系では全桁を失う。Python整数を保つか、`diagnostics.N_X_decimal/N_Y_decimal`を使う。B/attenuationをCへ再乗算しない。Cのcohort/expectation/domain・共有不確かさの資格は別担当の結果に従い、Nだけで総費用順位を決めない。

この固定menuのN側追加実行は不要。次はCodex Cが本handoffを価格側へ接続し、ユーザーがA/Cの価格資格と合わせて実G集計等の次段階を判断する。今回は価格処理・全G・候補最適化へ進まず、GitHub公開・remote SHA/主要成果物照合後mandatory STOPする。
