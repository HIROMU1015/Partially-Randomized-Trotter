# DF全Hamiltonian純RTE：固定代表2条件の軽量追加検証

linear H4/H6、1.00 Å、STO-3G、保存tol-only DF10/19（H6の保存weighted-Hermitian completionを維持）、T=0.8、P=0。
保存Hamiltonian・state・reference・DF fragment順を再利用し、入力生成と新reference計算は0。
**両系とも全Hamiltonian純RTEの構築・一体項sampling・native/compiled full-wrapper整合を確認した。以下は固定候補各n=9の点費用と経験的shot会計である。全域最適性、母平均費用保証、統計的優位を示さない。**

## 実装と再利用

既存B3のL_D=0は一体項を決定論側に残すため、そのままpure RTEとは呼べない。新adapterは既存B3 preparationの一体項固有値ε_jとGaussian基底を再利用し、
`H_one = (sum ε_j)/2 I - sum ε_j/2 U Z_j U†` の非scalar Z成分を既存DF tailの末尾へ追加した。
一体項scalarを既存抽出identityへ合算し、元Hamiltonian constantとともに一回適用する。決定論blockは空。二体DF fragment順・既存component順・係数閾値0を維持した。
この一体項symbolic追加と独立occupation oracleの対応だけが必要な科学実装拡張である。既存APIは一体項を必ず決定論blockに置くため、新adapterが必要だった。

`h46_b3.py`の構造、`h46_new2_native.prepare/OccupationOracle`、`h46_new2_core`のoccupation/spectral/polynomial、
`h46_long_time_signal.evolve_batch/qualify`、既存DF RTE sampler、native compose/cost_trial、compiler cache、
`h46_long_time_report.cost_for`を再利用した。高速fingerprintはB3でbyte一致検証済みのwriterを再利用。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing追加指定なし。cosine/sineとも測定付きfull Hadamard wrapper。P=0なのでstate preparationは価格外。

## 事前固定候補と正しさ

[selection.json](selection.json)はpure RTEの出力を見る前に保存B3のT=0.8・S2・ε=0.0005探索nomineeから固定した。H4 q16/R16/K4（s256）、H6 q32/R32/K6（s1024）。
Kは最大偶数event次数であり、paired Taylor numeratorはK+1次。各精度で同じsignalと価格標本を再利用し、回路再作成・再最適化は0。

|系|λ_R|一体項λ|一体項sampling質量|一体項成分数|全H再構成最大差|決定論block|実標本の一体項application|
|---|---:|---:|---:|---:|---:|---:|---:|
|H4|21.0279441|7.30424549|34.735899%|8|2.86e-15|0|816|
|H6|41.2279152|13.1792764|31.966876%|12|5.77e-15|0|2953|

再構成は保存DF Hamiltonianの独立linear operatorとも照合し、全粒子数sectorでは保存stateと人工複素vector2本で全component作用を照合した。
一体項自身の再構成、成分確率p=|a|/λ、一体項を含む正負時間・全採用偶数次数event、low/high control、actual adjointを検証した。
人工2成分の全event列挙でも有限K mean×normalizationがTaylor polynomialに一致した。人工・必要回帰14件PASS。
実候補ではraw/corrected vector closure、signal/native log B、scalarを照合し、各系trial0で独立occupation oracle、cosine/sine Hadamard期待値、compiled作用vectorを検証した。
詳細は[H4検証](H4_validation.json)、[H6検証](H6_validation.json)、[価格trialとwrapper検査](price_trials.json)、[人工テスト](technical_tests.json)。条件付き正しさfixtureは価格標本へ混入していない。

## 固定候補のsignal・費用

B=exp(log B)。biasは保存referenceとの差の観測値。共通u floor=1e-9、alpha_axis=.025、log B marginとheadroom/u政策を維持し、厳密bias保証とは区別する。
shot数は必要量子shotの計算値であり、実際の量子測定shotを実行した結果ではない。Cはn=9のcompiled RZ数平均、G=Ncos Ccos+Nsin Csin。
biasはcompile前の有限meanに対する数値観測であり、compiled回路の厳密bias保証ではない。代表compiled vector差の許容値は既存条件の1e-7。

|系|log B|B|max軸bias|Ccos / Csin|CXcos / CXsin|
|---|---:|---:|---:|---:|---:|
|H4|1.10279347|3.01256981|1.79e-14|40676.6 / 40676.6|12554.2 / 12554.2|
|H6|1.06172635|2.89135819|2.66e-14|479795 / 479795|135386 / 135386|

|系|ε|適格性|Ncos / Nsin|G ± MCSE|
|---|---:|---|---:|---:|
|H4|0.05|ELIGIBLE|63632 / 63632|5.176661e+09 ± 9.21e+07|
|H4|0.01|ELIGIBLE|1590778 / 1590778|1.294147e+11 ± 2.3e+09|
|H4|0.005|ELIGIBLE|6363111 / 6363111|5.176589e+11 ± 9.21e+09|
|H4|0.001|ELIGIBLE|159078128 / 159078128|1.29415e+13 ± 2.3e+11|
|H4|0.0005|ELIGIBLE|636314311 / 636314311|5.176615e+13 ± 9.21e+11|
|H6|0.05|ELIGIBLE|58614 / 58614|5.624536e+10 ± 5.34e+08|
|H6|0.01|ELIGIBLE|1465342 / 1465342|1.406126e+12 ± 1.34e+10|
|H6|0.005|ELIGIBLE|5861369 / 5861369|5.624506e+12 ± 5.34e+10|
|H6|0.001|ELIGIBLE|146534547 / 146534547|1.40613e+14 ± 1.34e+12|
|H6|0.0005|ELIGIBLE|586139845 / 586139845|5.624534e+14 ± 5.34e+12|

## 保存結果との比較

同じq/R/KのB3は制御比較。他の列は固定pure RTE候補と各方式の保存済み取得域最良（公式・q/R/K・prefixは方式ごとに異なる）の比較である。
B2は一体項＋一部DFを保持、B3は一体項だけ保持。PF envelopeはFull/B0/B0+Iを含む価格取得済み適格候補から選ぶ。
比較baselineは保存探索cohort、pure RTEは今回n=9。独立確認cohortは新たに取得せず、保存確認値とのpoolingも行っていない。

|系|ε|pure/B3同qRK|pure/B3保存最良|pure/B2保存最良|pure/Full PF保存最良|pure/PF envelope|
|---|---:|---:|---:|---:|---:|---:|
|H4|0.05|3.7833|4.4741|67.993|34.379|69.002|
|H4|0.01|3.772|4.2553|35.483|16.456|23.239|
|H4|0.005|3.7578|4.1204|22.936|10.075|14.331|
|H4|0.001|3.6442|3.6442|12.979|5.8391|8.357|
|H4|0.0005|3.5008|3.5008|10.053|4.2881|6.1519|
|H6|0.05|3.4191|3.6512|128.1|60.286|150|
|H6|0.01|3.4146|3.574|69.932|27.03|47.316|
|H6|0.005|3.4089|3.5163|53.155|19.38|34.202|
|H6|0.001|3.3639|3.3639|26.152|9.9441|17.868|
|H6|0.0005|3.308|3.308|25.953|9.3388|15.29|

pure RTEでは決定論側との分割が消えるため、観測biasは両系とも丸め誤差水準まで低下した。一方、一体項をsamplingへ含めることでλ_RとBが増え、shot数が増加した。
ここでの正味傾向は上表の固定点比較に限る。高費用でも「最適化pure RTEが不利」とは結論せず、低費用の比較があっても一般的優位とは扱わない。

## 希少event・未確認域・次の科学的価値

[rare_event_status.json](rare_event_status.json)は保存lawの確率と実効float CDFの確率、n=9で未観測となる確率、観測数を分離している。
条件付き価格、未観測event費用、母平均tail費用上界はnull。未観測費用を0として補完していない。条件付きnative正しさPASSは条件付きcompiled価格の取得を意味しない。
n=9のMCSEは有限標本の診断であり、希少eventを含む母平均保証や統計的順位保証ではない。すべての個別component列の条件付き価格は未取得。

実装が成立し、分割biasの除去とnormalization増大のトレードオフが見えたため、追加探索を検討する科学的価値はある。
ただし、この2点から広域探索の費用対効果や優位性は判断できず、広域探索を優先する根拠は弱い。特に緩い精度でも最厳精度由来の同一候補を使ったため、費用比は精度別最適化の比較ではない。今回取得していないsやλ低減を扱う研究は新しい範囲となる。広域探索、R/K再最適化、別PF次数、40条件追加、詳細科学レビューは開始していない。
S1論文Main Claimは変更していない。必要なテストとGitHub公開照合の完了後mandatory STOP。

## 成果物と再現

[RESULTS.csv](RESULTS.csv)はpure10行＋比較baseline50行、[comparisons.csv](comparisons.csv)は50比較のID・q/R/K・公式・cohortと点比率。
[SOURCE_INDEX.json](SOURCE_INDEX.json)は実行SOURCE、再利用SOURCE、保存入力hash・reference、保存結果参照と出力hash。
新科学実装は`scripts/h46_full_h_pure_rte.py`、人工テストは`tests/test_h46_full_h_pure_rte.py`、保存値会計は`scripts/report_h46_full_h_pure_rte.py`。
private event・seed・QPY/cacheは専用directoryのPRIVATE/へ保持し、公開trial記録には価格・law・検査結果を含めた。旧SOURCE・結果・grant・ledger・marker・STOPを変更していない。
branch: `codex/df-full-h-pure-rte-light-20261109`。実行SOURCE: `ac33da5f0cb038a780a447746a38e478bb0cec61`。
次の作業なし。追加探索はユーザーが別途採否を判断する場合のみ。
