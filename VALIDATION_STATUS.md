# Validation status

## 2026-08-26 H4 follow-up・H5 system-size circuit-cost completion note

2026-08-25のH4 follow-upは全jobがreturn code 0で完走し、専用validatorを通過した。同一H4 chain、
距離1.0 Å、STO-3G、DF rank 12、固定snapshot、Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、
seed 17、coupling mapなしの範囲で、$L_D=0$の全Taylor patternを同一trajectoryで比較した
K1--K3 paired residualは全metric最大1.679%、主RZ 95%上側診断3.201%だった。$L_D=6$、
short-step 0.025、$K=2$の固定K1--K4を未使用$L=8$へ適用した点誤差は最大3.750%だが、
主RZ 95%上側診断は7.045%だった。同一trajectoryのpaired K1--K4 $L=8$構造検証は最大4.008%、
主RZ 95%上側診断4.902%だった。controlled $q=1,2$ affine modelの未使用$q=8$ holdoutは
全metric最大0.0529%だった。

H5 chain、距離1.0 Å、STO-3G、10 qubit、project設定DF rank 9、$L_D=4$、short-step 0.025、
$K=2$、同じcompiler条件では、$L=4,6,8$のpaired K1--K3構造残差が全metric最大1.665%、
K1--K4が0.551%だった。5%を満たす最小cluster長としてK1--K3を選んだ。独立calibrationは
最大RZ相対standard error 0.745%で停止し、all-order-0各長さ500、single-order-2各位置125の
独立full holdoutは全metric最大点誤差3.776%、主RZ最大z 2.009、予測側95%半幅1.459%だった。
事前の5%点誤差と2%予測精度を通過した。点wise正規近似95%上側診断7.461%は硬い受理条件ではなく、
rigorousな5%保証とは扱わない。

したがって、exact Hamiltonian snapshot・compiler条件ごとにK1--K3を較正して代表holdoutを行い、
不通過条件だけK4へ進む運用規則を採用する。回路costの広いpilot検証は一旦区切るが、$L>8$、
compiler/coupling/backend変更、新snapshot、controlled scope拡張または5%未満の候補差では再検証する。
これらはlocal dirty-worktree evidenceであり、full RPE、quantum shot、noise、実backend、最終総cost、
immutable CIまたは外部再現結果ではない。

整理後のlocal全test suiteは`472 passed, 4 warnings`だった。4 warningは既存grouped-UWC testの
complex-to-real castであり、今回の回路cost検証由来ではない。H4 paired K4、H5 paired、H5独立
calibration、H5独立holdoutの保存payloadは各専用validatorを再通過した。

## 2026-08-25 circuit-cost follow-up実装note

最初の追加batchはA--Cすべてreturn code 0で完走し、専用validatorを通過した。Aの独立K1--K3は
主誤差12.376%で不通過だが最大z 1.787で原因を分離できず、Bの独立paired K4は最大点誤差
1.281%だがRZ 95%上側診断5.714%、Cのcontrolled $q=8$は全metric最大0.0529%で通過した。

follow-up用に、複数order-2のpaired full/local-window残差、K4 500標本へのincremental resume、
固定K1--K3のL8 holdout、固定K1--K4のL8評価を実装した。変更後のlocal全suiteは
`470 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。follow-upのlive stateは
`artifacts/rte_cost_followup_batch/2026-08-25/status.json`へ保存し、成果物validator通過前は
追加科学結果として扱わない。

## 2026-08-25 circuit-cost追加batch実装note

同一H4 fixed snapshot上で、$K=2$の複数order-2 pattern、$L_D=6$の対応あり独立K4、
controlled $q=8$ holdoutを採取する実装とdetached batch runnerを追加した。
変更後のlocal全suiteは`469 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
数値jobのlive stateは`artifacts/rte_cost_data_batch/2026-08-25/status.json`に保存する。
完走済み成果物を専用validatorで検証するまでは、追加の科学結果や最終cost評価として数えない。

## 2026-08-24 connected-cluster short-step 0.030 transfer completion note

中断していたH4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、short-step 0.030、
finite Taylor cutoff `K=2`、同一Hamiltonian snapshot・compiler条件のtransfer holdoutを再開した。
固定calibrationに対し、未使用`L=4,6` full回路の全order-0を各1500標本、order-2が1回の条件を
各位置500標本とした。

全6 metricの最大絶対相対点誤差は3.541%、主RZ countは3.488%、RZ最大absolute z-scoreは
3.037、予測側95%相対半幅は1.784%だった。事前の点誤差5%と予測精度2%は通過した。
500/150標本の予備runでは全metric 7.138%、RZ 5.754%、予測半幅2.265%で不通過だったため、
当該超過は再開runで維持されなかった。再開時はcalibration相対標準誤差目標も1.0%から0.8%へ
締めており、改善をholdout標本数だけには帰属しない。一方、RZの点wise正規近似95%上側診断は5.738%であり、
残差ゼロまたはrigorousな5%保証とは扱わない。

再開には旧pattern checkpoint、v3固定sample chunk、既存SQLite metric cacheを併用した。8000回路中
1250件がpersistent cache hit、6750件がmissで、3 workerの実経過時間は393.2秒だった。生成artifactは
内部fingerprint・chunk意味検査を通過し、一時checkpointは残っていない。この結果はdirty local
worktree evidenceであり、別compiler、coupling map、系サイズ、full RPEまたは最終総コストへは
一般化しない。

## 2026-08-24 connected-cluster lightweight operation implementation note

既存の主1500/375 holdout結果を変更せず、compiled-cost処理をoffline calibration、transpileを
呼ばないprediction、固定calibrationのtransfer holdoutへ分離した。v2 calibration/transfer schema、
pattern単位task/checkpoint、same/different基底の直接条件付きsampling、cache状態に依存しない
deterministic-work Neyman配分、完全な数値回路・compiler・backendをkeyとするSQLite metric cacheを
実装した。さらにv3では各patternの標本列を固定indexのsample chunkへ分割し、chunkごとの十分統計量を
atomic checkpointへ保存して合成する。標本数を増やす際は完了済みfull chunkを再利用し、末尾partial
chunkの置換と新規chunkだけを計算する。same/different pairはclass別統計量を合成後に解析確率で
再重み付けする。旧v2 pattern checkpointは標本数まで完全一致する場合だけ読み取り再利用する。
production後の実現相対標準誤差が目標未達なら、分散に基づき不足層へ再配分するadaptive roundも追加した。

同じevent identityを複数short-step時間でcompileする角度不変性validatorも追加した。既存H4固定
snapshot、`L_D=3`、short-step 0.020/0.025/0.030、cluster長1--3、各pattern 2標本のsmokeでは
全6 metric差0だった。ただし低標本のimplementation smokeであり、manifestへ科学的artifactとして
登録せず、角度を除外したstructural cache reuseも有効化しない。別`L_D`、境界coverage、
compiler/coupling条件で検証するまで、short-stepごとに数値回路keyを分離する。

H4のpilot 2・production cap 4 smokeはcold約39.8秒、別checkpointから同じSQLite cacheを使う
warm run約37.3秒だった。warm transpile missは0だが、回路構築とcanonical fingerprint計算が残る。
同一checkpoint再開は完了taskを読み飛ばす。checkpoint fingerprintにはtask実装versionも含め、
実装変更後に旧summaryを黙って再利用しない。固定chunk範囲、十分統計量の合成、標本数増加時の
full chunk再利用、旧checkpoint互換、worker数による科学的出力の不変性、chunk seed改変の拒否を
回帰testで確認した。変更後のlocal全suiteは`468 passed, 4 warnings`で、warningは既存grouped-UWC
test由来である。本節は実装能力の記録であり、
新しいcost精度、full RPE、
最終総コストまたは大規模性能の検証ではない。

## 2026-08-24 operational connected-cluster compiled-cost note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`r=4`、`K=2`の
1 compiler条件で、Taylor次数条件付きのK1--K3 connected-cluster運用推定器を検証した。
order-0単eventは厳密列挙し、pairはsame/different基底で層別、pilotからRZ countのNeyman配分を
決めた。productionとholdoutのseedは分離した。

正確なDF Hamiltonian配列をNPZ snapshotへ固定し、未使用full回路を`L=4,6,8`で評価した。
全order-0は各長さ1500標本、order-2がちょうど1回は各位置375標本である。RZ countと全6 metricの
最大点誤差は2.936%、RZ最大absolute z-scoreは2.074、予測側95%相対半幅は1.537%だった。
事前の点誤差5%と予測精度2%は通過した。一方、点wise正規近似95%上側診断は5.724%で未達、
order-2 K1は要求1535に対して標本cap 1500へ到達した。したがって代表1条件の
「実用点誤差5%内の暫定候補」であり、rigorousな5%保証とは扱わない。

別processで同じ分子条件から再構築したholdoutは、元holdoutと最大11.68%、z 5.215ずれた。
これは主結果へ結合していない。compiled-cost検証の再現単位は分子条件だけでなく正確なDF snapshotとする。
generatorはpilot K1--K3、production K1--K3、holdout L4/L6/L8の9 taskをfingerprinted checkpointへ
保存し、中断後は未完了taskだけを再開する。

全order-0 RZの差は`L=4,6,8`で+2.59%、-0.35%、+2.94%と単調増加せず、order-2が1回の
RZ点誤差は1.02%以下だった。この条件ではK4を追加せず、別`L_D`、short-step、compiler/coupling
条件への移送holdoutで悪化した場合に再検討する。full RPE、量子shot、noise、実backend、
resource accounting接続、最終総コストは未評価である。

主artifact、snapshot、9 checkpoint、versioned source、2 generator、専用test 4件および
[`docs/rte_connected_cluster_cost_validation.md`](docs/rte_connected_cluster_cost_validation.md)を
追加した。fingerprintとsnapshot SHA-256は再検査済みである。artifactはdirty local worktree evidenceで
ありimmutable CI evidenceではない。repository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

当該主artifact生成後に軽量運用APIを追加した最新local全test suiteは`468 passed, 4 warnings`だった。warningは既存grouped-UWC testの
complex-to-real castであり、今回のconnected-cluster検証由来ではない。

## 2026-08-24 hierarchical compiled-cost holdout note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`r=4`の
同一Hamiltonian表現を3 workerへ渡し、`K=0`の未使用`L=8`、`K=2`の`L=4,6`、
controlled partial-S2の未使用`q=4`を独立seedで評価した。三artifactの`preparation_hash`は一致した。

`K=0`では`C2,C3,C8`を各2000標本とし、pair-onlyはcount/sizeで最大8.851%残った一方、
triple補正は全metric最大1.744%、最大absolute z-score 0.521だった。したがって、この条件では
count/sizeにtriple、depthにpairを候補とし、4-event以上の係数は追加しない。

`K=2` runを監査すると、1 eventのorder-2確率は0.0001063で、旧`C1,C2,C3,C4,C6`
全8000 event位置中order-2は1回だけだった。したがって旧4.113%値は`K=2`内部の根拠から外す。
Taylor次数を強制した独立較正/holdoutでは最大点誤差9.119%、最大z 1.651となり、500標本の
独立係数差引きは精度不足と確認した。一方、同一trajectory上で1--3 event局所窓と全回路の
差を直接取る対応あり検証では、`L=4,6`のall-order-0/order-2が1回の全条件・全metricで
最大点誤差1.373%、RZ countの点ごとの正規近似95%診断1.796%だった。最大z 7.535なので
小さい4-event以上の残差は非ゼロだが、代表条件の暫定5%を通過した。運用時は解析的order重みと
対応ありconnected-cluster係数を使い、独立500標本係数推定は使わない。

controlled `q=1,2`各300標本から別seedの`q=4`を予測すると、全metric最大点誤差0.307%、
最大z 0.925、点ごとの正規近似95%上側診断0.958%だった。この代表条件ではaffine `q` model候補を支持するが、
`q>4`、別`L_D`、compiler/coupling条件または最終resource accountingへ一般化しない。

versioned source、並列generator、専用test、五つのfingerprinted local artifactおよび
[`docs/hierarchical_cost_validation.md`](docs/hierarchical_cost_validation.md)を追加した。
fingerprint、source hash、seed分離は再検査済みである。artifactはdirty local worktree evidenceで
ありimmutable CI evidenceではない。repository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

最新local全test suiteは`457 passed, 4 warnings`だった。warningは既存grouped-UWC testの
complex-to-real castであり、今回のcost検証由来ではない。

## 2026-08-23 high-statistics and stratified RTE boundary-cost note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`K=0`、short-step時間0.025の
同一Hamiltonian表現を3 workerへ渡した。独立2 seedについて`C2,C3,C4,C6`を各1000標本、
別のcalibration/holdout seedでfragment-pair補正を1500/1500標本評価した。三artifactの
`preparation_hash`は一致した。

1000標本runでpair補正の最大絶対相対誤差は8.07%、8.18%、最大absolute z-scoreは
2.96、3.02だった。系統差はcount/sizeで確認し、depthではpair-onlyの最大zは1未満だった。
triple補正は最大2.33%、3.73%、最大z 0.79、0.97だったが、`mu3`自体のabsolute z-scoreは
最大1.40なので、非ゼロを確定したとは扱わない。

same-fragment確率は0.7310604だった。different境界をゼロとするsame-only modelは別seedの
pair holdoutに対して最大誤差3.65%、最大z 2.59で外れた。same/different双方の条件付き補正を
解析確率で重み付けすると最大0.849%、最大z 0.587だった。したがってpair係数には少なくとも
二分類が必要であり、長いevent列ではdepthをpair候補、count/sizeをtriple候補とする。

`rte_boundary_pair_validation_v1` source、並列generator、専用test、三つのfingerprinted local
artifactおよび[`docs/rte_boundary_pair_validation.md`](docs/rte_boundary_pair_validation.md)
を追加した。受理閾値、他の`L_D,K`、compiler、controlled回路、resource accounting接続、
最終総コストは未評価である。artifactはdirty local worktree evidenceでありimmutable CI
evidenceではない。repository全体の`overall_status = not_reproducible_from_repository`は
変更しない。

最新local全test suiteは`451 passed, 4 warnings`だった。

## 2026-08-23 RTE boundary-corrected compiled-cost pilot note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`K=0`でshort-step時間を0.025に
固定し、RTE event列のcompiled-cost cluster modelをcalibration/holdout分離して検証した。
`C1`は218 eventを厳密列挙し、`C2,C3`は各300標本で較正した。別seedの未使用`C4,C6`を
各300標本で評価した結果、六指標を通じた最大絶対相対誤差はevent単純和157.51%、pair補正
8.73%、triple補正4.06%だった。最大absolute z-scoreは48.21、1.50、0.52だった。
同一DF fragmentが隣接する確率は0.73106だった。

triple残差は全metricで自身の標準誤差より小さく、depthではpair補正を一様に改善しなかった。
したがってpair補正を次の最小model候補、triple項を高次境界効果の診断量として記録する。
受理閾値、他parameter・compiler条件への一般化、resource accounting接続および最終総コストは
未評価である。

source、generator、専用test 2件、versioned/fingerprinted JSON artifact 1件および
[`docs/rte_boundary_cost_validation.md`](docs/rte_boundary_cost_validation.md)を追加した。
最新local全test suiteは`449 passed, 4 warnings`だった。artifactはdirty local worktree
evidenceでありimmutable CI evidenceではない。したがってrepository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

## 2026-08-23 random-circuit compiled-cost pilot note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`K=0`で、
complete circuitのcompiled costと、部分回路を別々にtranspileしたコスト和をpaired比較した。
`r=1`のpartial-S2は218 trajectoryを完全列挙し、3部分加法モデルは最大0.987%過大評価した。
同じDF表現を共有した`r=2`の100標本では、event別加算がRTE occurrence一体compileを
48.30--57.58%過大評価し、paired differenceのabsolute z-scoreは15.86--17.79だった。
別DF表現の300標本replicateでも52.99--61.15%の過大評価を確認した。
したがって、個々のeventを独立加算するモデルは採用せず、RTE occurrence以上をcost proxyの
最小較正単位とするpilot判断を記録した。

source、generator、専用test 3件、versioned/fingerprinted JSON artifact 3件および
[`docs/random_circuit_cost_validation.md`](docs/random_circuit_cost_validation.md)を追加した。
関連する既存cost testを含む最新local実行は`53 passed`だった。artifactはdirty local worktree
evidenceであり、1条件だけのpilotである。controlled回路、実backend、量子shot、ノイズ、
全round RPE、最終compiled総コスト、または他の`L_D,delta,r,K,q`への一般化を検証したものではない。
また、同じ分子条件でもprocess間でDF `preparation_hash`が変わるため、主$r=1,2$比較は
Hamiltonian共有batchへ置き換えた。異なるhash間の絶対compiled costは直接比較しない。

したがって、repository全体の`overall_status = not_reproducible_from_repository`は変更しない。

## 2026-08-19 current local approximation-validation note

研究内容と現在地の短い統合要約は
[`docs/research/研究概要・現状.md`](docs/research/研究概要・現状.md)を参照する。

2026-08-19のdirty local worktreeでは、最終コスト評価の入力を検証するため、次の三つの
result setを追加した。

- finite-RTE signal、attenuation、radius、phase-boundのH4 grid検証
- PF誤差surrogate、論文Appendix D Eq. (D6)のCPU Qiskit摂動係数、理想QPE分枝の
  H4全`L_D`検証
- H2--H6の実行可能delta窓と、PF演算子を構築しないEq. (D6) state-action係数検証

対応する文書、source、test、dirty-worktree artifactはmanifestへ登録され、構造検査は
成功している。H4全12分割では、well-conditionedな4点でfitしたEq. (D6)係数と
支配固有位相係数の差が最大0.288%だった。H2--H5の実行可能窓では両者の上包絡差が
最大1.144%で、事前の2%条件を全系が通過した。H6（DF rank 11、$L_D=5$）ではEq. (D6)による
`C_use=0.02086663`を得た。local testは`444 passed, 4 warnings`だった。これらは近似手法と実装経路の
local evidenceであり、最終compiled cost、H12の係数、量子shot、ノイズ、または外部から
再現された科学的結論ではない。artifactはimmutable CI evidenceでもない。

したがって、下記auditの`overall_status = not_reproducible_from_repository`は変更しない。
旧DF screeningとprose-only UWCを使用禁止とする判断も引き続き有効である。

## 2026-08-02 implementation hardening note

The current worktree replaces the DF legacy overlap proxy with a
shift-invariant, state-specific survival-phase-bias estimator (cache schema 8,
definition v3). It records explicit estimator status and is marked
`is_rigorous_bound=false`. Legacy/unmarked Cgs tables are now rejected by the
analytic PR-bound screening entry point, so the stale rankings described below
remain invalid and cannot be silently regenerated from the new surrogate.

Finite RTE distribution validation/serialization, actual-circuit/backend cache
identity, exact-zero circuit pruning, pre/post build workload guards, bounded
metric-only LRU caching, online compiled-cost statistics, and rolling
provenance digests were also hardened. A non-scientific Level-5-R regression
fixture now freezes all 32 combinations of `q=1..4`, raw/boundary-optimized,
controlled/uncontrolled, and exact/Monte Carlo compiled-cost evaluation.

The follow-up memory hardening makes event, partial-S2 request, exact
trajectory, and Monte Carlo trajectory generation single-pass. Level-5-R
provenance retains an explicit bounded prefix (1024 records by default) while
rolling digests cover the full stream. Cache-independent total build,
transpile-request, and instruction-application plans are rejected before any
Qiskit builder is called and checked against actual post-build work. The
Level-5-R fixture was rerun without changing gate/depth/mean/standard-error
values; schema 2 adds runtime/PRNG metadata and a digest over all 32 result
streams. The local regression suite reports `224 passed` in the documented
Python 3.11 environment. This remains implementation evidence only; no H3--H14
scientific baseline was generated by this change, and no long RPE circuit,
quantum shot, GPU statevector, noise simulation, or backend job was run.

## 結論

**監査基準 commit [`cf285c0`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cf285c0ac1e3d587df4a8eb6bee2279a12ced462) の内容だけでは、現在の DF screening / UWC 検証結果を外部から再現・追跡できません。**

ここで「外部から再現可能」とは、clean checkout から、commit 済みの入力と手順を使って結果を再生成し、その結果が公開済みの数値と一致することを確認できる状態を指します。本書は既存 artifact の棚卸しであり、新たな科学計算を実行した結果ではありません。

> **DO NOT USE:** 現在 commit されている DF screening JSON を、修正済みの結果または最終結果として引用しないでください。ファイル内の算術は整合していますが、その Cgs 入力は後の commit で基底状態の不整合を理由に削除され、screening は再生成されていません。

## ステータス一覧

| 対象 | commit `cf285c0` にある証拠 | 判定 | 読み方 |
|---|---|---|---|
| 旧来の高次 Trotter 評価 | 出力付き `abe_trotter_project.ipynb` と、`artifacts/trotter_expo_coeff_gr{,_original}/` 内の係数 pickle 計 540 個 | **historical** | README が説明する旧来の高次 Trotter 解析の成果。現在の DF screening / UWC の検証証拠ではない |
| DF reduced screening | `epsilon_total=1e-4` の JSON 1件。635候補、12分子の best を収録 | **DO NOT USE / stale** | 保存値の加算と best 選択は内部整合するが、元の Cgs 表が削除済みで再生成不能。protocol 上も shortlist 前の近似 screening |
| DF 最終評価 | protocol と実装 | **incomplete** | shortlist の explicit-`L_D` Cgs 再 fit、H14 `8th(Morales)`、`4th(new_2)` が未完了 |
| UWC | 実装説明と H2--H6 等の数値表を含む Markdown | **reported only** | 表が参照する machine-readable JSON は commit されておらず、表から元 run を追跡できない |
| テスト | 4ファイルに `test_*` 関数定義が28件。UWC note に過去の `26 passed` の記録 | **current result unknown** | `cf285c0` に対するテスト実行結果ではない。この変更で追加する manifest 構造検査も科学計算・全 test suite は実行しない |

## 証拠と監査結果

### 1. DF screening

対象 artifact:

- [`artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json`](artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json)
- [`Partial Randomized Study Protocol.md`](Partial%20Randomized%20Study%20Protocol.md)
- [`artifacts/partial_randomized_pf/README.md`](artifacts/partial_randomized_pf/README.md)

JSON 自体について確認できる範囲は次のとおりです。

- `candidates` は635件で、1件は `(molecule, PF, L_D)` の組です。
- `best_by_molecule` は H3--H14 の12件です。
- 全635候補で、保存値の `g_total` は `g_det + g_rand` と一致します（最大絶対差 0）。
- 12件の `best_by_molecule` は、それぞれ同じ molecule の候補中で最小の `g_total` と一致します。

これは **JSON 内部の算術と選択処理だけ** の確認です。入力データ、Cgs fit、物理モデル、または結果の科学的妥当性を検証したことにはなりません。

再現性を失っている直接の理由は次のとおりです。

1. JSON の `cgs_table` は `/home/AbeHiromu/Project/.../df_cgs_cost_table.json` という生成環境の絶対パスを指します。
2. commit [`98f960c` (`基底状態ずれてたので削除`)](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/98f960c2dd09fc1ae6b8b5c802dc5ce84fc61604) は、集約 Cgs 表、split 表、index の計37ファイルを削除しています。
3. その後も上記 screening JSON は残っていますが、削除理由を反映した正しい Cgs 入力から再生成された artifact はありません。

また protocol は、この計算を候補を絞るための近似と定義しています。screening では anchor の `C_gs,D(p,L_anchor)` を各 `L_D` に使い回し、**最終評価では shortlist の各 `(p, L_D)` で Cgs を再 fit して `G_total` を再計算する必要があります**。同じ protocol には、次も未完了と記録されています。

- H14 `8th(Morales)` の anchor Cgs
- H3--H14 `4th(new_2)` の anchor Cgs 計算、cost table への merge、再 screening
- shortlist に対する explicit-`L_D` Cgs の再 fit

したがって、入力問題がなかったとしても現在の JSON は最終結果ではありません。

### 2. UWC

[`notes/uwc_current_implementation_and_results.md`](notes/uwc_current_implementation_and_results.md) には、H2--H6 grouped UWC、H3 time-grid 診断、theta sweep、simple shift の条件と数値表があります。一方、同文書が参照する次の出力を含む `artifacts/grouped_uwc_pf_qpe/` は commit `cf285c0` に存在せず、`.gitignore` でディレクトリ全体が除外されています。

- `H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json`
- `H3_bliss_sector_scaling_diagnostics.json`
- `H2_H6_2nd_grouped_uwc_alpha_simple_shift_gpu.json`
- theta sweep 表の元になった run 出力

したがって Markdown の表は「報告された数値」として読めますが、repository 内の canonical raw/summary artifact と照合することはできません。なお文書自身の結論も、現在の simple BLISS quadratic shift では grouped PF+QPE cost がほぼ低下していない、という限定的なものです。

### 3. テストと CI

commit `cf285c0` の `tests/` には、静的に数えた `test_*` 関数定義が28件あります。

- `tests/test_df_hamiltonian.py`: 5件
- `tests/test_df_partial_randomized_pf.py`: 9件
- `tests/test_grouped_uwc_comparison.py`: 7件
- `tests/test_uwc_preprocessor.py`: 7件

UWC note が保存している実行記録は `.venv/bin/python -m pytest -q` の `26 passed` です。これは後から追加されたテストを含む現在の suite に対する結果ではなく、実行 commit、依存環境、完全なログも記録されていません。監査基準 commit `cf285c0` には `.github/workflows/` もありませんでした。この変更では manifest と記載パスの構造検査だけを追加しており、科学計算または全 test suite の CI ではありません。このため、`cf285c0` の28定義が pass するとは本監査から主張できません。

## 再現を妨げているもの

- 修正済みの DF Cgs 集約表・split 表・index がない。
- stale screening JSON に入力 hash、生成元 commit、実行環境、実行 command がない。
- DF screening の修正後再実行と shortlist の explicit-`L_D` 再 fit がない。
- protocol に記載された H14 `8th(Morales)` と `4th(new_2)` が未完了。
- UWC の Markdown 表に対応する machine-readable run artifact がない。
- UWC artifact の保存先が `.gitignore` され、レビュー可能な canonical summary の例外設定がない。
- 現在の HEAD を対象とする自動テスト結果がない。

## 「検証完了」とするための条件

以下をすべて満たした時点で、DF / UWC の結果を repository から外部検証可能と扱います。

1. **DF 入力を修正して固定する。** ground-state のずれを修正した Cgs を再計算し、集約表、全 split 表、index を同時生成する。各表に molecule、PF、`L_D`、入力 Hamiltonian hash、生成元 commit、生成 command を記録し、相互の件数と hash を検査する。
2. **未完了の DF ケースを埋める。** H3--H14 `4th(new_2)` と H14 `8th(Morales)` の必要な anchor Cgs を生成し、同じ canonical table に merge する。失敗または除外する場合は、対象、理由、結果への影響を明記する。
3. **screening を再生成する。** 修正後の canonical table だけを入力として `epsilon_total=1e-4` screening を実行する。結果には相対的な入力 path、全入力の content hash、生成元 commit、command、依存環境、candidate 件数を保存する。`g_total = g_det + g_rand` と molecule ごとの best 選択を自動検査し、旧 JSON を stale として置換または明確に隔離する。
4. **最終 DF 評価を実行する。** screening の shortlist と選定規則を保存し、各 `(PF, L_D)` で anchor ではない explicit-`L_D` Cgs を再 fit して `G_total` を再計算する。最終表から各 fit の machine-readable artifact と入力 hash へ追跡できるようにする。
5. **UWC の根拠データを公開する。** Markdown に載せる全表について canonical JSON/CSV を commit し、条件、baseline、seed（使用時）、backend、入力 hash、生成元 commit、command を保存する。Markdown の値が artifact から自動生成または自動照合されるようにし、必要な summary だけを `.gitignore` の例外にする。
6. **clean checkout で検証する。** 固定した依存環境と文書化した command で、小規模な end-to-end 再生成および全 test suite を CI から実行する。結果 artifact を作った commit に対する成功 check を GitHub 上に残し、比較 tolerance と期待値を test または検証 script に固定する。

上記が完了するまでは、旧来の高次 Trotter artifact、DF screening、UWC 表を互いに独立した進捗資料として扱い、現在の partial-randomized DF/UWC の完成済み検証結果として一括して引用しないでください。
