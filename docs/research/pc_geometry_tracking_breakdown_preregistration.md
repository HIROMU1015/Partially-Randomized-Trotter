# P-C geometry tracking・breakdown validation事前登録

最終更新：2026-09-25 JST

## 目的

既存P-C pilotはH4 linear chainの0.80--1.20 Åで、少数geometryから未使用geometryと未使用deltaの
符号付きPF biasを予測できた。ただし、その結果だけでは

- 各geometryの独立DF分解が既に連続だったのか
- orbital／DF fragmentを明示的に追跡すると固定prefixを保護できるのか
- smoothな局所補間がどこで破れ、その破れをPF biasを見る前に診断できるのか

を区別できない。

本検証では結果を見る前に、orbital overlapでgeometry間の共通Hilbert-space対応を作り、DF fragmentを
連続追跡した `L_D=3` とgeometryごとの独立順位 `L_D=3` を比較する。未使用geometry領域のsigned
PF errorと差分biasを予測できるか、追跡の利益または予測可能なbreakdownのどちらかを識別できるかを判定する。

## 固定物理条件

- molecule：H4 linear chain、charge 0、multiplicity 1
- basis：STO-3G
- qubit／electron：8 qubit、4 electron number sector
- DF：OpenFermion low-rank two-body decomposition、rank 12固定
- deterministic prefix：`L_D=3`
- PF：二次partial-`S_2`、randomized tailはexact dense reference
- identity policy：`extract_identity_phase`
- matrix backend：CPU dense、小系参照
- QPE/RPE sampling、RTE sampling、回路compile、noise：対象外

## geometryとblind分離

固定gridは0.70、0.80、0.90、1.00、1.10、1.20、1.40、1.60 Åとする。

- coefficient training：0.80、1.00、1.20 Å
- blind holdout：0.70、0.90、1.10、1.40、1.60 Å
- tracking anchor：1.00 Å
- left path：1.00 → 0.90 → 0.80 → 0.70 Å
- right path：1.00 → 1.10 → 1.20 → 1.40 → 1.60 Å

0.90、1.10 Åは未使用のinterpolation holdout、0.70 Åはcompression側、1.40、1.60 Åは既存範囲外の
stretch側holdoutである。PF bias結果を見てgeometryを追加・削除しない。

## 比較する表現

### independent

各geometryでOpenFermionが返すrank-12 fragment順をそのまま使い、先頭3 fragmentを
deterministic prefixとする。

### continuously tracked

隣接geometryのcanonical spatial orbital間にAO cross-overlapを作り、SVDのpolar factorで
位相・並び・近接縮退空間を含む最小二乗unitary対応を得る。これをspin orbitalへ持ち上げる。

anchor 1.00 Åの独立先頭3 fragmentから開始し、各pathの次geometryで、軌道対応後のnormalized
Frobenius overlapの絶対値をscoreとしてHungarian matchingを行う。前geometryの選択3 fragmentを
異なる現geometry fragmentへ一対一対応させ、その3 fragmentをtracked prefixとする。残りfragmentは
独立順を保つ。

fragmentの符号反転は同じ二体項を表すため、matching scoreは絶対値を使う。PF biasやground energyを
matchingへ使わない。

## signed biasと予測

各geometry・policyでfit delta 0.025、0.05、0.10とdelta holdout 0.20を評価する。
dominant target phaseの共通規約
`b(R,delta) = E_PF(R,delta) - E_exact(R)`を使い、原点を通る
`b(R,delta) ≈ C(R) delta^2`をfitする。

training 3点の `C(R)` だけからpiecewise linear predictorを作る。0.80--1.20 Å内は線形補間、
外側は最も近い2 training点の傾きで線形外挿する。blind PF biasをpredictor作成へ使わない。

固定pairは次の4組とする。

1. interpolation：0.90 → 1.10 Å
2. compression boundary：0.70 → 0.90 Å
3. stretched region：1.40 → 1.60 Å
4. cross-region：0.90 → 1.40 Å

pair予測はdelta 0.20で評価する。cancellation量は
`kappa_cancel = |b(R_b)-b(R_a)| / (|b(R_a)|+|b(R_b)|)`とし、差分が小さすぎるだけの例を避けるため
DF-rank-12 exact energy differenceも併記する。

## PF結果を使わないcontinuity診断

各tracking edgeで次を記録する。

- spatial-MO cross-overlapの最小特異値
- full rank-12 ground-stateの隣接geometry overlap
- ground-state spectral gap
- tracked prefix 3 fragmentの最小matching similarity
- assigned similarityと次善similarityのmargin
- independent prefixとtracked prefixの一致／不一致

次のいずれかを満たすedgeは、事前に `predicted_breakdown=true` とする。

- minimum orbital singular value < 0.80
- ground-state overlap < 0.95
- ground-state gap < 0.02 Ha
- minimum tracked-fragment similarity < 0.90

blind coefficient予測誤差10%以下をactual smooth、20%以上をactual breakdown、間をinconclusiveとする。

## 固定gate

次の7 gateを結果後に変更しない。

1. representation integrity：
   - tracked並べ替え前後のfull Hamiltonian sector operator差が最大 `1e-10` 以下
   - 0.80、1.00、1.20 Åの再生成ground energyが訂正済みP-C artifactと `1e-8 Ha` 以内
   - 同3点のindependent `C(R)` が訂正済みP-C artifact値と相対1%以内
2. delta holdout：全geometry・両policyで `C delta^2` のdelta 0.20 bias相対誤差が最大10%以下。
3. blind coefficient prediction：tracked policyの5 blind点中4点以上が相対15%以内。
4. pair prediction：tracked policyの固定4 pair中3 pair以上で、予測誤差が大きいendpoint biasの15%以内。
5. nontrivial cancellation：固定pairの少なくとも1つでexact energy differenceが0.02 Ha以上かつ
   `kappa_cancel <= 0.50`。
6. diagnostic transfer：actual smooth／breakdownに分類できるblind点が4点以上あり、
   preregistered continuity診断の正解率が80%以上。
7. mechanism discrimination：次のどちらかを満たす。
   - tracked prefixがblind 2点以上でindependentと異なり、trackedのblind coefficient予測誤差中央値が
     independentより10%以上小さく、pair normalized errorを5 percentage point超悪化させない。
   - actual breakdownがblind 1点以上あり、その全てをcontinuity診断が事前にflagする。

## 固定decision rule

- gate 1--7を全て通過：
  `advance_pc_tracking_and_breakdown_design`
- gate 1--6を通過しgate 7だけ不通過：
  `pc_signed_error_remains_smooth_only_require_new_condition`
- gate 1--5は通過するがgate 6不通過：
  `pc_geometry_response_unexplained_do_not_advance`
- gate 1--5のいずれか不通過：
  `stop_pc_current_h4_family_as_primary`

gate 7不通過後に同じH4 pathへgeometryやthresholdを足して主張を復活させない。別条件を試す場合は
新しい事前登録とblind領域を用意する。

## scope

通過しても、potential-energy surface全体、force、反応障壁、別分子、別basis／rank／PF、
full RPE/RTE、量子shot、回路resource、noise、H12、最終総costまたは科学的優位性は確立しない。
orbital overlapは共通表現を作る診断であり、独立量子測定間の統計共分散を意味しない。

## 実行規約

runnerの `--dry-run` でgeometry、policy、delta、tracking path、pair、gateをexpected-task artifactへ
固定する。その後、implementation、runner、expected-task artifactを変更せず本計算を行う。

- implementation：
  `src/trotterlib/research_direction_geometry_tracking_breakdown.py`
- runner：
  `scripts/run_research_direction_geometry_tracking_breakdown.py`
- expected task：
  `artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/pc_tracking_breakdown_expected_tasks_v1.json`
- final artifact：
  `artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/pc_tracking_breakdown_validation_v1.json`

## compile-before固定記録

- expected-task content fingerprint：`cbe260750d081316070d3684a2a24194d91d029ad700a54cf4f61152f2ed4a4e`
- expected-task file SHA-256：`a3dd1621dc8777e9f6429fd2d35c6e1270e18d27e4996c7168ec9bd216b2d5b2`
- implementation SHA-256：`a77c427f569b36218baa0a6343ed77db96951767a12a0f9a373bef8f2da3195b`
- runner SHA-256：`53410e399e4a4b8bc31991d0b6dcef1e9d746522b9bb4179906727fff4968912`
- 訂正済み先行P-C artifact content fingerprint：`3c199dbc20d0892cf1bcfad4b27646ea8c90d1c1d32fc5e2c75be610bb1fe611`
- 訂正済み先行P-C artifact file SHA-256：`499427a801d3e35bac9151077b314d0848588189f8440c023cfd34adfccaae9b`
- expected task数：16
- 固定時点ではfinal artifact未生成。以後、implementation、runner、geometry、gateを変更せず本計算する。
