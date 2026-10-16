# M1-S Gate 0 最終実験契約 v3 候補（2026-10-16）

**CANDIDATE_NOT_ADOPTED / IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED。新科学計算0、数値correctness test0、formula評価0。mandatory STOP。**

## 1. 承認境界と来歴

基点 repository `HIROMU1015/Partially-Randomized-Trotter`、branch
`track-b-m1s-gate0-precision-contract-20261016`、commit `4019d045cbfad94ba9333b544f7950a6d10d467d`。
新専用branch `track-b-m1s-gate0-formula-contract-20261016`。
固定v3 SOURCE commit `dcfb97b8275b77e0e959c17715006f787e2993bf`。
[machine candidate](../../../artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1/execution_contract_v3_candidate.json)、
[SOURCE hash](../../../artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1/source_identity.json)、
[未決表](../../../artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1/unresolved_items.json)を参照。

[共通運用2026-10-16](../../research/research_project_instructions_unified_2026-10-16.md)を適用。
旧2026-10-12版を現行指示として併用しない。共通指示の更新は旧契約・結果・grant・marker・STOPを解除しない。
ユーザーが採択済みなのは[限定科学scope記録](../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/gate0_scientific_scope_adoption_record.json)のみ。
本v3、17項目の値/式、runtime/資源案の正式採択をCodexは代行しない。

## 2. モデル・入力・比較を固定

linear H4、1.00 Å、STO-3G、8 spin orbitals、DF rank12、saved prefix G3/G6/G9。
delta windowなし（finite-step runなし）。入力は保存development NPZ一つのみ：
`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`、8652 bytes、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
H hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
mode0 MSB、alpha-even/beta-odd、保存DF順を使用。holdout、新rank、resort、DF再生成はしない。
H=c+one-body+Σλ_l A_l²、A_l=Σg_lpq a†p aq。Pauli cutoff0、signed aggregation後にabs、lexicographic IXYZ。
identityをHRからHDへ移し、+i T_rel Hでのscalar phase ledgerを残す。
入力gはdimensionless、λ/one_body/constantはHartree。λのqDRIFT rateとDF eigenvalueを混同しない。

保存stateを上書きnormalizeせず、norm²でquotientを取る。exact eigenstate、full-Fock ground、非縮退性、SP条件は未認証。
W=1−〈h²〉、v=Var〈signed P〉、ρ=(W+v)/(2W)はnumerical-oracle headroom診断。
same ideal taskでR個のconceptual event positionsを比較し、shared r=R、ordinary r=R、independentは各branch R/2（R even）。
λ²/Rで正規化したleading coefficientはshared (W+v)/4、ordinary/independent W/2。
native shots/回路/総費用の優位やfinite-R/finite-delta signalを意味しない。

## 3. 17項目の精度契約

[全算定式・導出・必要入力・coverage条件](m1s_gate0_preregistered_error_formula_design_20261016.md)が本候補の一部。
既存8 scalar候補はそのまま：factor Hermiticity/basis各0、state norm/imag expectation各2^-40、
probability normalization2^-46、eps_W=eps_v=2^-24、eps_rho=2^-12。
残る9欄はallowlisted formula IDへ対応づけた。うち8欄は条件付き候補として具体化し、
eps_state_residual=η_oracle S+B_res の **η_oracleはnull**。
全項目でfixed scalar指定も維持するが、結果を見た自動切替・式への任意expression追加は禁止。

formula pathの対象はexact saved binary64 H/canonical law。IEEE/BLAS/fsum・underflowなし等の条件を明示しており、
platform証明・数値検証はしていない。`coverage_model_adopted=false`を維持。
formula pathは入力h/gのexact Hermiticityを追加quality gateとし、repairしない。
scalar quality gate自体に必ず収まるとの事前主張はしない。

global尺度はH構築・energy/residual評価より前、law尺度はhpsi/Pauli expectations/W/v/ρより前に解決する。
resolved ledgerをmetric前にhash固定し、selectorをsplitごとに独立に適用する。
λlo≤0、eW>eps_W、ev>eps_vは`PRECISION_UNRESOLVED_BEFORE_METRICS`、そのsplitのW/v/ρはnull。
予算は保守的で全splitがunresolvedとなる可能性があり、解決値を予想・試算していない。
W_floorは(eW+ev)とeps_rhoから導き、任意のfixed multiplierを追加しない。

rhoは外向きW/v区間からrectangle intervalを作り、zero/small denominator・raw rhoの範囲・eps_rhoで報告可否を分類する。
raw pointはclampしない。ρやdressingからのautomatic scientific GO/STOPはfalse。G3/G6/G9を選別せず全てstatus付きで記録。
global failureで未到達splitがあってもnull/statusを残す。

## 4. runtime、performance、出力

[既存runtime/caps候補](m1s_gate0_runtime_and_resource_contract_candidate_20261016.md)を維持。
Python3.11.0rc1、NumPy1.26.4、6 build-file hashes、CPU1 worker、
OPENBLAS/OMP/MKL/NUMBA/RAYON各1thread。4 GiB AS、output/log各256 MiB、context JSON4 MiB、
canonical terms65536、global product contributions1000000、scientific wall-time capなし。
cached bilinears/DF block maps/reference、mask cacheを再利用し、過剰な全Pauli matrix cacheは作らない。
resolverはraw scaleとlawの線形走査＋小数個数のFraction式。性能/数値benchmarkは未実施。
将来認可された実行では固定runtime/resource guardを適用し、逸脱はtechnical/precision unresolvedとして記録。
sampling・simulation・eig/gap・σ・coupling・circuit/compile・総resources・T1c等は含まない。

## 5. 実行防止、one-shot、保存

[v3 runner](../../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v3.py)はdefault/status/schema/validate-configでNPZを開かず、numerical libraryをimportしない。
全formula helperもvalid permit取得後だけ呼べる。現在candidateは品質係数・モデル/全契約採択が欠けるためexecution_ready=false。
実行には、source commit < 採択契約commit < 別実行承認commit の順序、clean dedicated checkout、
全raw bytes/hash・scope・Python/NumPy build/threadの一致が必要。
separate authorization schemaは`m1s_gate0_separate_execution_authorization_v3`。
scope採択記録やv2 authorityを流用しない。new result namespaceのexclusive作成で再試行/上書きを防ぐ。
今回authority、grant、marker、result directoryは作成しない。
旧v1/v2 SOURCE、全旧結果、authorization/marker、scope記録、入力、T1a/T1b/GPT reviewは保全。

## 6. 採択時に残る判断と次の順序

ユーザーが決める事項は以下。

1. oracle residual quality η_oracle、又は同等の品質基準（現時点で未決）。
2. 条件付き誤差coverageモデルを受理するか、従来8scalar候補と今回8formulaを含む17項目の採否。
3. runtime/caps、quality失敗・precision unresolvedの扱いを含む最終契約全体の採択。

採択だけでは実行できない。さらに別途、対象source/契約に結びつくGate 0科学計算の明示承認が必要。
数値correctness testを含む追加数値処理も今回の作業には含まれない。
現在の状態は**最終契約の採択用資料準備完了・実行不可**。mandatory STOP。
