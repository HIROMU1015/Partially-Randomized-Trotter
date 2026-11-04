# 研究C：Head bridge／Stage Pの技術準備

2026-11-04 JST。**Head bridgeと1-word Stage Pのadapter・人工検証・独立した未承認契約を準備した。新規43件、関連回帰込み355件PASS（理由付き7件除外）。両passive preflightはPASS、実H4の新規科学call・配列decode・build/compileは0。** 実H4のHead bridge上界／per-word上界はnullを維持する。

起点はStage R結果 `1c0e1d1b4967efecb08dd3d8e7a747eba6c83039`。[256-call結果](research_c_population_range_schema_execution_20261104.md)のexact a/Mと保存full headsを再利用し、Stage Rを再実行しない。独立branch `codex/research-c-head-bridge-stage-p-readiness-20261104`。正式共通指示原本2026-10-25版の§8.1・§8.4・§10を確認し、承認証明・ID・承認原文保存・新grantを作成しない。

固定対象はH4・直鎖1.00 Å/STO-3G、number CDF8、元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1、m0..12、sector36/full256/controlled512。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、own-DF ground oracle・状態準備除外。元g補正κ、scalar、frame順、既存P0 Qiskit1.3/opt1/seed17/rz,sx,x,cx・MPFR160 declared phase、独立53-bit IID-grid・既存CA/RPE政策は変更しない。

## Head不一致の静的原因と確認範囲

| 経路 | コード・定義・算術 |
|---|---|
| Stage R full head | `c1_stage1.car_bilinears`、`raw_base`のdense einsum、`raw_frame`のcoefficient@E、J@density、density@weighted、np.sum。Full Fockのbinary64出力G0/G1/G2をexact dyadic targetとして保持 |
| B2へ渡る保存sector head | `c_near_practical`の物理worker→`c_pr_integrated_evaluation.Evaluation.assess`→`rank_assessment`→`SectorCAR.generators/one/raw_frame`。Sparse excitation-table bilinear、loop/weighted sums、sparse productsで得たsector generatorを保存 |
| B2での指数化 | 固定SOURCE `01baf2b5e400773e00a5ae548941265f85b9882b` の`c_reference_bound_contract.arrays/_head_task/head_products`。保存sector bitsをArb384へ入力し、exp(−iG/20)の元forward/reverse順を使用 |

B2の公開SOURCE本文を取得し、保存SOURCEのSHA256 `243cfed90352de2e5771cf2e8f31e0b7e10041916d30ee5f3ecb4df9d9608ddd`と一致させた。B2全branchのmergeやrunner再実行は行っていない。

同じκ/J/UのCAR多項式は、spin-preserving sectorへの制限と積が可換なら厳密算術では一致する。κは両側とも同じ`h−.5*einsum("prrq->pq",g)`、核scalarはG0に一度、tail scalarは別の位相一度。相違するのはdense/sparse contractionと積和のbinary64評価経路であり、捕捉後のbit値をexact inputとするtargetは別になる。

人工4-spatial入力で、basisを正しく揃えたfull/sector生成経路のbit差を再現した。**静的な経路差と、bit差を生む算術機構を特定した。実H4のどの演算・entryが差を支配したかという定量的な原因帰属は未実施**である。保存NPZを新しくdecodeせず、0/3不一致を単なる無害な丸めとして無視しない。実H4でのbasis/scalar/phase適合性と差の上界は、新H契約で検査する。

## Head bridgeの方法とB2接続

[c_head_reference_bridge.py](../../src/trottertracks/c_head_reference_bridge.py)は既存`Arithmetic/Matrix/pack/unpack`、`_ordered_exponentials`、Arb `_matrix/_norm`を使用する。共有B2 moduleはこのcheckoutへ丸ごと持ち込まず、適合する既存CAR/Arb primitivesと同じhead順序を再利用した。

1. Stage Rの保存full G0/G1/G2と、B2が使った保存sector G0/G1/G2を読む。再生成・対称化・polar化しない。
2. Basisは`bit_reverse(saved_interleaved_sector_indices)`。Full headのsector↔complement両cross blockが**exact zero**であることを要求し、漏れがあればSTOPする。個々のGaussian gateをsectorへ射影する方式ではない。
3. Arb384で両側のexp(−iG/20)を包囲し、forward Lとreverse R、差のphase-sensitive Frobenius上界を求める。Global phase fitなし。
4. 非Hermitian headのnormにはanti-Hermitian部分のlogarithmic-norm上界を使う。Unitaryと仮定しない。

同じtail作用Eと共通位相に対し、A=R_F E L_F、B=R_S E L_Sとすると、

`||A−B|| ≤ ||E|| (||R_F−R_S|| ||L_F|| + ||R_S|| ||L_F−L_S||)`。

Tail normはStage Rの保存event exact上界と正supportから得る。現固定データでは正q-support216が全216 atomを覆うという保存記録を照合するので、同じhead差をp/qのいずれの共通tailにも使用できる。確率閾値でatomを落とさない。Full側aはStage Rのexact rationalをそのまま使用し、sector側もnorm上界bを求める。

n=2^mではtelescopingにより`H_m = n e max(a,b)^(n−1)`を上向き包囲する。13roundをArb384で評価し、巨大値は既存compact dyadic endpointを保持する。未知値を0にせず、非unitary targetの誤差にcap2を適用しない。

B2の二項は固定結果 `b7a3db09e395d36965e5b9dc28ae4a299db3b2f0`のまま。接続は**新Head差H_m＋既存B_m（atom→CAR）＋既存U_m（CAR→binary64 reference）**の三項を一回ずつとする。既存totalを第四項にせず、Arb rounding、normalization、PF/RTE、categorical、shellを重ねて足さない。全項はraw complex signalの単位で、余分なattenuationの乗除をしない。

このbridgeの資格は**physical sector36**。Full512 head bridge、物理積分・真のgenerator生成誤差、ground精度やactual CAは対象外。実H4のH_mはnullのままで、bit不一致を解消済みとは扱わない。

## Stage P：2-leg、inverse、shell

[c_stage_p_literal.py](../../src/trottertracks/c_stage_p_literal.py)と共通runner [c_head_stage_p_execution.py](../../src/trottertracks/c_head_stage_p_execution.py)は、次を使用する。

- 既存StrongBuilder、typed event emitter、P0 `compile_checked`。Head内相対融合・tail exact basis-key共有は元条件のまま。新たなhead/tail融合は導入しない。
- Stage Aの保存angle/normalization/pをtyped K0 emission recordへ接続する。Law生成、sampling、PF/RTE誤差やattenuationの再計算は行わない。
- `fixed_certificate.record/compare`による**compiled C→raw R**の全512入力列包囲。既存のunitary pairに対するcap2は、このlegにのみ適用できる。
- `build_literal_reference/compare_literal`による**raw R→literal D**。全256 Fock基底を25 spin-count blockで漏れなく覆い、raw側は全512入力列・全出力・controlled相対位相・leakageを含める。
- 同じraw record digest、word、target、angle bitsを照合し、二項の和でC→Dを得る。R→Dと最終triangleにcap2はない。

旧fixed CAR APIのpolar targetは使わない。Full literal headはStage Rで保存したbitsであり、新しいJ/Uからheadを再生成した値へ置換しない。B2の平均reference上界や旧4-word numerical pilotを、新per-word certificateとして流用しない。

**Raw→literalとHead bridgeは別leg。** PのDはfull-CAR head targetで、R→Dにはそのnative/head相違を含める。HはそのDからB2側の保存sector-head atom targetへ移す追加legである。HをPのR→Dに再加算しない。将来B2へ接続する段階でHを一度だけ使う。

Actual inverseは同じ実現raw/compiled回路から各一つ生成し、全512列でexact record adjointと比較する。その誤差とforward boundの和はD†への上界になる。**Nonunitary Dの逆行列を再現したとは主張しない。** 独立wordのinverseを引き直さない。

保存head/X/Y shellは6 QPYを読込む。厳密にancilla-onlyと確認できる場合だけ2×2へreduceし、登録末尾measurementだけを除く。各shellでcompiled→ideal、raw→ideal、compiled→rawを包囲し、信号会計にはcompiled→idealを一度だけ使う。他二項は診断である。

1 word・1 occurrenceのraw信号は、kernel誤差e、D norm≤a、shell誤差e_H/e_Qに対して、composite作用差≤e_Q+e+a e_H、観測量差≤(1+a)倍で包囲する。X/Yを別々に保存し、complex差も上向き合成する。Nonunitary targetの両normが入り、未知shellはnull。既存の期待値校正／反復用算術は用途・前提が異なるため、この有限wordの信号算定へ無理に転用しない。13roundの期待bias、shots、CAを計算する経路ではない。

## Stage Rのexact rangeとの接続

正本はStage R `population.json`のexact rational a/Mである。表示近似を契約値へ戻さず、`M=1+a`をexact arithmeticで検査する。保存target_id・head/J/U bits・wordの各angle hexを照合するので、per-wordとpopulationで数学的targetを付け替えない。Per-wordの緩い上界を勝手に2や観測sample最大でclipしない。

Pで確認できるcompiler成功は**その1 wordだけ**。`all_q_compile_success_qualified=false`、`calibration_ready=false`を保持する。既存`calibration_inputs`は、別途all-q compiler-domain前提がない限り校正へ進まない。有限wordの選び方もIID sampleではなく、期待bias推定へ流用できない。

## 未承認科学契約と最小実行

[共通SOURCE](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/SOURCE.json)は旧157 fileを保持し、新module3・runner・testを加えた162 file。入力48件は旧保存結果／固定候補／Stage R headとpopulation／B2証拠／shellに限定する。科学SOURCEはこの技術版の公開commitとmanifest/file hashで固定する。旧SOURCE・契約・raw・grant・ledger・marker・STOPは変更しない。

| 契約 | 数値資格・範囲 | 最大calls | Fresh output |
|---|---|---:|---|
| [H：Head bridge](research_c_head_bridge_contract_unapproved_20261104.json) | Arb384、3 head対、両順序、sector36、13round・B2三項接続 | **26** | `artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01` |
| [P：1-word literal](research_c_stage_p_literal_contract_unapproved_20261104.json) | Arb192、full512全列、C/R/D二項、actual inverse、shell・単一occurrence信号 | **41** | `artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01` |

Hの内訳はmetadata2、保存配列group1、identity1、model1、invariance1、head pair3（各2指数）、products1、kernel1、round13、B2接続1、summary1。原Gを再生成せず、保存配列を比較する。Model準備は共通tail位相のbinding用で回路buildではない。

Pの内訳はmetadata/input/candidate/head/model/catalog/binding各1、必要basis5・head相対transition最大4、raw build1・P0 compile1・record2、literal reference1、二leg2・triangle1、actual inverse build/record/certificate各2、shell load6・certificate group3・finite signal1・summary1。各pair certificateは全512列を含む。Shell groupは3組の2×2比較を含む。QPY出力はraw/compiled/各actual inverseの4件。Cache共有により実績が上限未満となる場合も、残枠で候補を追加しない。

Pは旧65-call提案をそのまま実行せず、Stage R保存head再利用・不要なground/law/全raw9因子/numerical pilot/metricsを除いた専用契約とした。対象wordは既存提案どおり`CDF_proposed:F6:0`と`CDF_proposed:F7:6,7`の交互16回。旧4-wordを再実行・poolせず、新samplingなし。

[H passive](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/preflight_H.json)／[P passive](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/preflight_P.json)は、bytes/JSON/packageだけでPASS。科学call・配列decodeは0、fresh出力は両方未作成。入口は既定passiveである。

```bash
python scripts/prepare_research_c_head_bridge_stage_p.py --stage H
python scripts/prepare_research_c_head_bridge_stage_p.py --stage P
```

別途その契約への実行許可が与えられた場合だけ、対応する一方へ`--execute-science`を付ける。H/Pは独立して停止し、互いを自動起動しない。新しいfit/optimizer、rank探索、Stage R再実行、PF/RTE再評価、sampling、Stage C、期待bias校正、shots/総費用、重要科学レビューは0。

Correctness・input/SOURCE・basis・phase・gate型・compileのFAILは後続依存を止め、完了したhead pair／certificateとpartial failure・journal・STOPを保存する。Boundが大きいことは精度達成とは区別して記録する。Retry/resume0、phase fit・閾値緩和・head置換・別wordへのfallbackは行わない。正常終了もmandatory STOP。

## 人工検証・性能・保全

[人工43 PASS](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/tests.txt)、[関連回帰込み355 PASS／7 deselected](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/regression_tests.txt)。完全一致／微小差／phase差／非Hermitian head、leakage・誤identity・B2二重計上、per-word target/raw/angle、cap2禁止、full512 artificial controlled-phase、actual inverse、shellのphaseと不適切なsector縮約、未知誤差、partial STOPを確認した。人工H pipelineは26 calls、P pipelineは実artificial build/compile/record/inverse/shellを接続し、重いcolumn engineだけをmockしたrouting testを含む。実Arb二legは小さな人工分子、full512 column engineは人工gate列で別に確認している。

除外7件は、消費済みStage R V1/V2へfreshを要求する旧cold2件、既知の誤schema fixture1件、前段から記録済みの歴史的SOURCE drift依存4件。旧テスト／契約を改変せず、新H/Pのcold preflightと回帰に置き換えた。全履歴契約の包括的PASSとは主張しない。

CPU affinity32、空きRAM約52 GiBを確認した。Hの独立3 head pairはprocess-local Arbで並列化し、Pは既存のspin-block／全column process poolを再利用する。Pool数は実quota・affinity・空きRAMから決め、一つのconcurrency層を使う。Qiskitの既存build/compile/actual-adjoint排他を維持する。新しいCPU/RAM/worker/wall capや共有venv/OS/cache/他jobの変更はない。実H4のP包囲runtimeは今回測定しておらず、speedupや所要時間を認定しない。

[技術照合](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/verification.json)で旧固定205 fileを起点と照合し、両科学出力が未作成であることを確認した。[技術STOP](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/STOP.json)は実H4科学0を記録する。旧Stage R 256/256結果と旧10-call失敗はどちらも保全した。

公開時にremote SHA/tree、公開本文、SOURCE・契約・主要blobを照合する。**照合後mandatory STOP。** 最小の次段階はHの26-callでhead transportを測ること。Pの41-callは独立した有限word包囲の検証として選べる。ユーザー／研究Cが実行対象と承認範囲を判断し、Codexはその固定範囲への別途明示許可後だけ実行する。全law compiler前提と校正法の判断は引き続き残る。
