# H4 B0I・Full-H pure RTE：全13round Raw signal取得

2026-11-14。承認済み26 candidate-roundを一回実行し、**B0I 13/13、Full-H 13/13が元数値gateを通過**した。Raw complex mean 26件（52座標）とcorrected/reference/normalization/数値診断を保存した。全meanは `SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED` / `UNQUALIFIED`。RPEの成功、必要N、方式の最適性は認定していない。

## SOURCEと実行範囲

- 固定scope：`6306fb3a74e7864a44a3e6cd022e9a4b062e7d44` の [SCOPE.json](../truncated_fullh_signal_scope_20261114/SCOPE.json)。原39round manifest `c5466946ab0ef9b50511210ca1e58fb7147bb0f0` は変更せず、候補1・2だけを使用した。Full PF m6受入への依存はない。
- 実行SOURCE：`5a172629db4d32a4230f8bb10a871b42807b1cd5`、[caller](../../../scripts/run_truncated_fullh_signals.py)。既存 `I.plan/produce_signals/signal_input` → `C.execute` → `baseline_signal_numerics.evaluate` を使用した。
- 数値moduleは `5cc7c07e112ce97205948465c94441cba0042416` とbyte一致のまま。直接expm/powerは `dd24dd329949c255d5e37b90ff8993fa8df7159b`、Full-H spectrum/polynomialは `e417ffc9f52b13c18f5493193ce510d858664259`、finite-Kは `71ba49bb28b295aa28e51101bc3b1bf07d500767` の元関数を再利用した。
- fresh出力 [run01](run01/) にB0I→Full-Hの順で実行。新評価26、旧signal受入/copy 0、retry 0。保存reference/scalarの使用時照合は承認範囲内。旧Full PF値との比較・受入再資格はしていない。

## 固定物理条件と入力

Linear H4、1.00Å、STO-3G、保存tol-only DF10、sector(2,2)/36次元、同H/state/snapshot、hは元binary64の0.1、M12、P0。targetは元のfull DF-Hの同sector ground-energy taskであり、保持Hamiltonianの基底エネルギーへ変更していない。保存stateの厳密ground同定は未証明のまま。prior・decoder・target/error contract・AのH_eff評価を変更していない。

H hash `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`、state hash `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`、snapshot hash `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`。元INDEXのsnapshot/receipt/contextのbytes/hash、元array/sector/layoutとshared bindingを既存loaderで照合した。保存係数から11行列のcontextを一回復元した。SCF/DF fit/state生成は0。

prefix7 scalarは保存contextから `4.5302064712666724e-05` をcopyした。欠測時extractorは起動していない。Full-Hもmanifestの保存preparation/scalarを使用。13時刻のreference pairは `bba150834ca30bdb186fbfd58c55a8d51177b855:docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json` から同binding/m/Tでcopyし、新reference actionは0。[INPUT_PROVENANCE.json](run01/INPUT_PROVENANCE.json) に元locatorとhashを保存した。

|系列|固定構成|新評価|保存copy|数値gate|B資格|
|---|---|---:|---:|---|---|
|B0I_P7_S2_D01|one-body＋fragment0..6、S2、ΔPF=.1、qPF=2^m|13|0|13 PASS|UNQUALIFIED|
|FULL_H_K2_QDYADIC_R448|one-body込みfull non-scalar、head0、finite-K2、grouping q=2^m/R448|13|0|13 PASS|UNQUALIFIED|

B0Iは元fragment/signed順の直接expm＋matrix_power、元H定数phaseの後にdiscarded tail identity phaseを総Tで一回適用した。Raw=corrected、B=1。Full-HのqはgroupingでありqPFではない。`dt=(T/q)/448`、finite polynomialのR乗→q乗、raw/correctedの独立伝播、元constant＋抽出identity scalar一回を維持した。PR prefix0やB3へ置換していない。

## 結果と数値診断

全26件は [SIGNALS.csv](run01/SIGNALS.csv) に既存recordの値を転記した。正本は系列別 `round_m.json` と `RESULT.json`。次は取得値の抜粋で、新しいfit・比較指標は算出していない。

|系列|m|Raw real|Raw imag|log B|
|---|---:|---:|---:|---:|
|B0I|0|0.9766285152027931|0.21493427317987748|0|
|B0I|6|0.2698105803624372|0.962913355522076|0|
|B0I|12|0.20435749917378324|0.9788961919549124|0|
|Full-H|0|0.9670337403954656|0.2128370883657245|0.009869835202359914|
|Full-H|6|0.14299373051957578|0.5121145426362452|0.6316694529510345|
|Full-H|12|4.0778157937848435e-19|2.742171710777319e-18|40.42684498886621|

元complex128、state norm1e−12、Hermiticity/PF power norm/unitarity1e−9、reference pair1e−10、raw/corrected closure1e−9、strict Raw座標[-1,1]、finite normalizerとB/logB/attenuationのgateを維持した。Full-H有限Kへunitarityを要求していない。clamp・投影・再正規化・閾値緩和は0。

m12の保存診断では、reference pair差 `9.279812219411241e-12`、B0I power unitarity誤差 `1.3370099748451616e-11`、Full-H独立raw/corrected vector closure `1.2539462437982403e-30`。Full-H m12 raw半径は元診断値 `2.7723260103869133e-18`。小さなRawについても厳密な絶対/相対誤差boundや必要Nを新しく認定していない。normalizerのone-ulp感度は診断であり、全誤差保証ではない。

数値gate PASSとrigorousなsignal誤差保証は別である。全recordの `numerical.certified=false`、law distance/state error/empirical allowanceはnullを保持した。Full-Hの平均はideal finite-K lawに対する数値伝播で、PCG64/CDF samplerとの距離は未資格。unknownを0にしていない。

## 技術確認・資源・保全

新callerの人工3test（正常2系列、第一系列失敗時のglobal STOP、保存referenceのcopy/異SOURCE拒否）は [CALLER_TESTS.txt](CALLER_TESTS.txt) にPASSを保存。数値providerの既存24人工/回帰記録を再利用し、同じ試験やexpm詳細検証を繰り返していない。実26件のcanonical/physical identity、law、SOURCE、packetと正本Raw/corrected/referenceのcopy一致、未資格/nullの保持を [CHECKS.json](run01/CHECKS.json) で確認した。作者自身の技術確認である。

Python3.12.3/NumPy1.26.4/SciPy1.14.1、既存専用venvを使用。shared context一回と既存half-step/step/spectrum/polynomial/normalizer cacheを共用し、batch経過時間は `4.14884963305667 s`。メモリピーク/速度向上率は計測していない。CPU/RAM/workerへ新しい上限を設定せず、共有環境・他担当worktree/processを変更していない。実行logには既存moduleのSyntaxWarningが残るが、数値gate失敗やretryはない。

新sampling、回路build/compile、SCF/DF/state再生成、Full PF/H_eff処理、MC/risk/N/RMSE、価格/残差fit/G/候補最適化は0。c546/6306の原計画と旧STOPは当時の記録として保全した。本結果は承認された2系列subsetの後続成果であり、原39round計画を実行済みと書き換えない。

## 受渡しとSTOP

B向けは系列別 `B_SIGNAL_INPUT.json` と `PLAN.json.configuration`。詳細は [HANDOFF.md](HANDOFF.md)。`I.signal_input` の既存形式を使用し、N/C/Gはnull。今回の取得完了はrisk資格や価格資格の認定ではない。

[STOP.json](run01/STOP.json)：指定26件取得完了、mandatory STOP。**次はユーザーがBによるbaseline risk評価の具体的scopeを判断し、許可した範囲をBが評価する。** Cは自動的にrisk/N/価格/Gへ進まない。AのFull PF選択とDのcompiled価格資格はそれぞれ別作業のまま。
