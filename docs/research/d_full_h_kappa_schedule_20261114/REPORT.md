# Full-H pure RTE：固定R448からκ型event scheduleへの移行調査と有限検証計画

2026-11-14 JST、Codex D。**既存SOURCEの調査、κ型scheduleの接続、metadata・人工検証、未承認の有限計画までを行った。** 新しい実H4/H6計算、保存signal・risk・価格の数値再解析、回路build/compile/sampling、N・MSE・期待価格・Gの評価、κ・K・decoder・誤差予算の採択は行っていない。

[R448確認](R448_CONFIGURATION.md)／[κ接続・signal・価格](KAPPA_CONNECTION.md)／[有限科学計画](FINITE_SCIENCE_PLAN.md)／[HANDOFF](HANDOFF.md)／[候補metadata](KAPPA_CANDIDATES.json)／[生成script](BUILD_PLAN.py)／[人工test](TESTS.txt)

## 固定SOURCE

|用途|commit|
|---|---|
|Full-H R448 signal結果|`367dc98d832ecb4d769f2a3da3d174bc697331a5`（scope `6306fb3a`、計画 `c5466946`、caller `5a172629`）|
|C数値・pipeline・比較guard|数値 `5cc7c07e`（`3a8b37f1` と同byte）、guard `3a8b37f1a282c0c06ac1a363b070f4376e9ff41a`|
|D S1/V1|`3d39f8c0ca68fb1acf86fe65d8ee5e3cc1a15673`（新ZZ `01e068af`、F-H案）|
|Full-H preparation／T2.0|`9b83b96b`（`prepare_full_h`、λ）／`7a8bc7f1`（R448の由来）|
|B controller／D V2 adapter|`437ba4ac`（D branchの `547b1ada` blobと同一）／`e7dca5f7`（Cが固定するpublished API）|
|κ定義|A `510fbacd`（論文v2 E21/E22/E.3/B.2の転記）、B `82bc774a`/`d562bc38` interface|
|未実行screen|`256caad3`（R2-ADAPT、κ_eff≈2.03〜2.06、未評価）|

既存D branch `codex/d-compiled-residual-c-minus-s-20261113`（基点 `120b618a`）を継続した。D S1/V1とFull-Hの価格・lowering調査に連続する作業のため。

## 1. R448固定構成の問題点と有効範囲

R448は、T=2.0単一時刻の最適化で得たH4 q8/R448/K2（κ_eff=2.03）から、**R=448/groupだけを流用し、q=2^m（group時間0.1）**として全13roundへ置いた「単純な接続seed」である（c546で非最適と明記）。s=448·2^m∝Tなので、κ_eff=s/(λT)²はm0の101からm12の0.025へ下がり、logBはroundごとに倍になる（保存値：m6 0.63、m9 5.05、m12 40.43）。m0〜m5は必要以上のeventで回路が長く、m7以降は減衰が指数的に拡大する。

13 meanはこの構成のideal finite-K law平均として数値gateを通過しており、R448 baseline・数値経路の確認・固定R scheduleが不適であることの観察には有効である。一方、m9以降はRawの減衰が6.4e−3以下で推定に実用的でなく、Full-H pure RTEの最適性能や劣位を示すものではない。κ候補とphysical IDを共有しないので、値を流用しない。詳細は [R448_CONFIGURATION.md](R448_CONFIGURATION.md)。

## 2. κ型scheduleへの必要最小限の実装差分

- **実装済み（metadata専用、約150行）**：[`src/trottertracks/full_h_kappa_layout.py`](../../../src/trottertracks/full_h_kappa_layout.py)。
  - `kappa_descriptor`：B/A共通定義のκを名前・SOURCE付きで作る。
  - `layouts`：総数はB `paper_event_mapper` の `ceil(κ(λ_R T)²)` を使い、grouping q（既定2^m）とR=ceil(r/q)へ分ける。
  - `check_configuration`：κ供給routeでBが省略する明示layout/preparation検査を生成後に行う。
- **production無変更で通るもの**：D `full_h_rte_spec(kappa=…)`/`layout_mapper`/`physical_schedule`、B `generate/_validate`、C `configure/prepare`、`baseline_signal_numerics.evaluate`、C `price_implementation_guard`。
- **未実装（範囲の承認後）**：
  - S-1用のsignal caller（既存callerのscope差替え＋追加gate）。
  - P-1用の新ZZ Full-H compile caller（composeの範囲だけ新loweringを有効化、cost_trial index0、sidecar）。
  - Bのrisk/N受入。

## 3. 再利用できる既存コード・SOURCE

- 入力：保存H/state/snapshot、DF10/sector、h、target、13 reference pair、Full-H preparation・λ・scalar、prior/decoder/RMSE policy。
- signal：C数値SOURCEとpacket形式（`I.plan/produce_signals/signal_input`）、gate一式。
- identity：B/D/Cの生成・照合関数。builderは保存R448のconfiguration IDと13 physical IDを完全に再現した。
- 価格：Full-H `setup/context/full_oracle/trial_seed`（9b83）、native `compose/cost_trial`、新ZZ subclass、full measured wrapper、C guard、C `quote_model`（FULL_H ADDITIVE）と `MeanKernel.repeat`。

## 4. λ・R・B・raw signal・N・価格の依存関係

```
λ_R（one-body込みL1、9b83）, κ, T_m=h·2^m
  → r_m=ceil(κ(λ_R T_m)²) → (q_m, R_m=ceil(r_m/q_m)), s_m=q_m R_m, τ_m=λ_R T_m/s_m
  → b(τ_m; K2), logB_m=s_m log b ≤ (λT)²/s_m ≤ 1/κ（＋1 ULP余裕）
  → corrected_m = P(x·T/s)^s（固有値x；finite-K bias）, raw_m = corrected_m·e^{−logB_m}
  → N_m（Bのdecoder/risk。経験形ではB²[11+4(M−m)]）
  → E[C_m]（新ZZ、full wrapper；≈a+b·s_m+c(q_m−1)、未検証model）
  → G = Σ_m (N_X C_X + N_Y C_Y)
```

R448ではsが固定率でしか増えずBが爆発する。κ型ではBが一定で、価格がs∝4^mで増える。差は価格だけではなく、raw減衰とshot数も同時に変わる。

## 5. lowering・Full-H価格比較の整合条件

旧Full-H価格（light/T2.0）は旧 `RZZ.control(1)`、別T/q/R/K、別cohortで取得されており、κ候補の価格にも新ZZ価格の上界にもならない。比較policyは、PR/Full-Hとも `CONTROLLED_ZZ_CX_CRZ_CX_V1`（SOURCE `01e068af`）、同じcompiler/control/full measured X/Y wrapper/P0、native compose `af284d85`、scalar=c_H＋Full-H抽出identityを総Tで一回、で揃える。C guardは変更なしでκ layoutを照合し、旧新のRTE混在を拒否する（人工test）。長時間roundは直接compileせず、group分解model（単一group R≤36,223）をUNQUALIFIED scenarioとして扱う。詳細は [KAPPA_CONNECTION.md §6](KAPPA_CONNECTION.md)。

## 6. 推奨する有限科学検証とSTOP

| 段階 | 内容 | 規模 |
|---|---|---|
| **S-1（推奨・最初）** | κ2・K2・DYADICの13round signal/normalization（数値のみ） | context 1、normalizer 13、mean 13、新reference 0、数秒想定 |
| S-2（任意） | κ1・κ4の括り | normalizer 26、mean 26 |
| **P-1（推奨・S-1と独立）** | κ2 m3（q8/R71、s568）新ZZ、normal 1 path、X/Y | **compile 2** |
| P-2（任意、Cのmodel scope） | 単一group R∈{566, 4,528, 36,223}＋m5/m6 holdout | 例：compile 18（n・精度はB/C決定後） |

m12の直接compile、全13roundの価格取得、N/G評価は含めない。STOPは、SOURCE/input/ID/q/R/reference/scalar不一致、数値gate・logB包絡（1 ULP余裕付き）・Raw範囲の失敗、compile側index0 gate失敗、新ZZ非有効、OOMなど。部分結果を保存し、retryせずmandatory STOPする（[FINITE_SCIENCE_PLAN.md](FINITE_SCIENCE_PLAN.md)）。

## 7. 未取得の証拠・残るUNKNOWN

- κ候補のfinite logB、raw/corrected、finite-K bias・位相（特にevent数がR448より少ないm0〜m4）。
- BのN/risk、decoder alias、十分N。κ=2が実riskとcompiled価格の下で最適かどうか。K=2が最適かどうか。
- 新ZZ Full-Hの実回路と価格（どのsでも未取得）、per-event定常性、group境界項、law/rare・sampler距離、m7〜m12価格modelの誤差包絡。
- grouping（2^m、1、その他）の価格への影響。H6のκ型。
- 原論文PDFの再読（本環境で描画不可のため、Aの転記に依拠）。

## 8. ユーザーの承認が必要な事項

1. S-1（κ2、13 mean）の科学実行。callerを固定したcommit SHAを示してから許可を受ける。S-2（κ1/κ4）を含めるかどうか。
2. P-1（κ2 m3、新ZZ、compile 2）の実行。D S1のF-H案（R448 m0）を代わりに又は併せて行うかどうか。
3. 正式比較の実装scenario（PR・Full-Hとも新ZZ）。R448を正式比較familyから外し、対照に限定するかどうか。
4. BによるκFull-Hのrisk/N評価scope。CのP-2（価格model）の精度目標と件数。
5. 上記の結果を見た後のκ・K・groupingの採択（今回は採択していない）。

## 技術確認・保全

[TESTS.txt](TESTS.txt)：人工・metadata test **10/10 PASS**。

- B/A総数式との一致、layoutの被覆と包絡、EXPLICIT groupingとidentity、拒否条件、生成routeとκのecho、改ざん拒否。
- κの有無でphysical IDは共有し、configurationは別になること。C guardの解決とmixed拒否。
- 人工τ gridでの包絡と1 ULP余裕。
- 保存R448 ID再現（numpy/scipy/qiskit/pyscf非import）。

`BUILD_PLAN.py` は約1 sでR448を再現し、κ1/2/4を生成した。作者自身の技術確認であり、独立科学レビューではない。

既存SOURCE・input・raw・結果・grant・ledger・marker・KEEP・STOPと、A/B/Cのbranch・worktree・実行中のB DP processは変更していない。merge、履歴変更、ref削除、prune、gcは0。commit/push・remote照合後 **mandatory STOP**。
