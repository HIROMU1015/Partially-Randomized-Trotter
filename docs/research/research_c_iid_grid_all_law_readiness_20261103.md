# 研究C：IID finite-bitモデルに条件付けたreference bridge・全law・RPE精度評価の技術具体化

2026-11-03。ユーザーが指定した**event/occurrence/shotごとに独立な一様53-bit乱数モデル**を採用した。
**Stage A（現在候補のreference bridge）は技術的GO／未承認38-call実行契約が完成。**
Stage Bは`P0_UNIFORM_NOT_ESTABLISHED`、Stage Cは`NOT_READY_BOUND_INPUTS`。
実PCG64 joint stream、厳密CA、actual CAは未資格。実H4の新TV・bridge・compile・sampling・RPE評価は実行していない。

## 固定対象・SOURCE・既存証拠

対象はendpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` のPRだけ。
直鎖H4・隣接1.00 Å/STO-3G、Nα=Nβ=2、sector36、number CDF rank8、元frame順LD2、partial ordered S2、
s1/r32/K0、δ=h=.1、m0..12、CA=.00159 Ha/DF reserveCA/8、own-DF ground oracle、状態準備費用除外。
保存P0（Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/MPFR160 declared phase/既存Gaussianとrelative fusion）、
controlled full512、head/X/Y shell込みfenced serial会計を変更しない。ENTRY/XDF/CDFへの新検証は対象外。

- 起点：[前回技術準備](research_c_all_law_signal_ca_readiness_20261103.md)、commit `3a784888a188db946681a6143c5e18479ddf284a`、実装 `ff80264447158c02d1f5d2f075e06f2ee8c970a4`。
- 新PR下流結果 `beb8de5cd8b71f80bb69556f92299435ff431c3e`、科学SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`。
- [探索的な対XDF/CDF価格統計](research_c_pr_xdf_cdf_statistical_comparison_execution_20261103.md)の結果 `2f1c86f161fb76e84dfa6b31f6912eb8a01ccb8f`、科学SOURCE `e2884a071d60ff8d2ffcacc98705e2e5cfafb206` は無変更。
- [固定Evidence Map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/169e327ae50960173bb2690ab98ea92c85e7a5ed/docs/research/research_c_near_objective_xdf_evidence_integration_20261103.md)の旧Claim・未資格を保持した。

正式共通指示の原本 `research_project_instructions_unified_2026-10-25.md` を読み、§8/§8.4/§10を適用した。
科学実行や重要科学レビューは技術準備とは別の許可範囲。承認ID・証明・原文保存・新grantを要求しない。

## 採用したIID-gridモデルとPCG64の違い

各消費drawは独立な整数 `U∈{0,...,2^53−1}`。`U/2^53`を既存NumPy1.26.4のカテゴリ選択写像へ入れる。
Event内のK0 order選択とatom選択、別event、別coherent occurrence、別measurement shot、X/Y測定は契約上の独立抽出を仮定する。
Actual inverseを使う場合は同じ実現回路のinverseであり、inverseだけを独立再抽出するモデルではない。
PCG64実装・過去のseed/cohortは保全するが、この抽象IIDモデルをPCG64の数学的保証と同一視しない。
`PCG64_joint_stream_error=None`／`PCG64_joint_stream_qualified=False`を維持し、0を代入しない。

既存[C2 categorical](../../src/trottertracks/c_pr_aware_df_stage_c2.py)の`analyze_table/choice_grid/tv/upper_float`をそのまま使用する。

1. 理論pは現在のsaved-atom係数の絶対値を正確な有理数として正規化する。
2. Stored binary64 probabilityとsamplerのmass再正規化を分離する。
3. 実際のbinary64 `cumsum` とCDF終端での再正規化を再現する。
4. `searchsorted(...,side='right')`を使うため、bin iは`cdf[i−1]≤U/2^53<cdf[i]`。
5. `count_i=ceil(2^53*cdf[i])−ceil(2^53*cdf[i−1])`、`q_i=count_i/2^53`がIID-grid法。

2^53格子の全列挙や乱数samplingは不要。ライブラリの`choice(size=0)`は引数検査だけで、乱数drawは0。
正確なzero coefficient、stored floatへのunderflow、CDFで消える小さいmass、gridで0になる正確率を別々に記録する。
Zeroに見えることを根拠にideal atomを削除しない。今回の実H4のp/q/TV値は未計算。

独立なkernel再抽出では`E[U_n...U_1]=(E[U])^n`。同じwordをcoherentに繰り返す`E[U^n]`とは一般に異なる。
価格校正の同一平均を全roundの費用へ共用することも、実際の量子回路のwordを再利用することとは別である。
保存native32のseed分離・correctnessは、この独立性モデルの実PCG64による実現証明ではない。

## Stage A：現在候補用reference bridgeの技術接続

[新adapter](../../src/trottertracks/c_iid_grid_bridge.py)は旧SOURCEを編集せず、以下を接続する。

`saved raw CAR finite-RTE reference → exact saved-atom ideal IID mean → finite-bit IID-grid mean`。

再利用する実関数は、Stage Bの`prepare_arm/configure/atom_operators/candidate_factor`、
C2の`analyze_table`、旧Bridgeの`atom_audit/event_mean/compare`、既存`_ordered_exponentials`、
`finite_rte_attenuation`、既存array loader/save、`Calls/resource_plan`。
旧Bridgeの`heads/kernel_bridge/round_bridge`はh1/40・r16・s4やideal pだけを固定しているため、そのentryを現在候補へ流用しない。
追加部分はh/2=.05、r32、n=2^m、saved corrected kernelの再利用、p/q両平均と出力の接続に限る。
新native builder・PF/RTE生成engine・確率engineを実装していない。

### 数理上の対象と実装

Signed atomをP_j、非identity係数をc_j、`lambda_exact=Σ|c_j|`、`p_j=|c_j|/lambda_exact`とする。
K0 eventは`E_j=cos(phi)I−i sin(phi)P_j`、`phi=atan(tau)`、tauは保存configのlambda_floatとh/rから定義される。
厳密なunitary atomと同じ角度の下では、

`mu_p=Σp_j E_j=(I−i tau Σp_j P_j)/sqrt(1+tau^2)`。

Kernel平均は`M_p=exp(−i h c_identity) S_rev mu_p^32 S_fwd`、grid側は重みをqへ置いた`M_q`。
HeadのG0には元g補正κと核scalarが既に含まれるため、両half-headでscalarは一度、tail identityだけを別phaseへ一度含める。
Saved raw CARは`R_m=A_m C_saved^(2^m)`、理想atom/gridモデルはそれぞれ`M_p^(2^m)`／`M_q^(2^m)`。
保存groundによるcomplex matrix elementをX/Y期待値と対応させる。

Lambda_exact/lambda_float、CAR atom係数和、角度・cos/sin、normalization、identity/scalar、frame/spin/bitの対応には
数値的な差や不確かさがあり得る。そのため式の対応だけでbridge差を0とせず、現在候補の9因子、216 signed atom、
λ/τ/identity/K0、kernel、13roundのnormalization/signal replayを明示的に検査する設計にした。
**技術的なparameter routingと再現可能な数値処理が完成したのであり、実H4のbridge一致や誤差上界はまだ取得していない。**

旧`event_mean`は内部field名が`ideal_probabilities`だが、処理は一般の重み付きK0平均である。
Grid側にはqを入れた新しい一時viewだけを渡し、元のp表は変更せず、保存出力にはgridモデル名を明記する。
実H4のfinite-RTE/PF kernelや誤差を再生成せず、保存corrected kernelを読む。Head・event平均・その反復は今回Stage Aとして新たに評価する対象である。

### 入出力・資格

入力は既存のh/g/scalar、固定endpoint J/U、同candidateのsaved CAR generators、corrected kernel、own-ground、13round scalar。
NPZのmember名はcentral-directory metadataだけを確認し、今回payloadはdecodeしていない。
実行時はJ/Uの完全一致、shape/finite、ground norm、9因子対応を先に確認する。

出力はexact p/q/grid counts/TV、216 atomの数値検査、p/q event mean、kernel、13roundのoperator2/Frobenius/complex-signal距離、
保存raw signal再生差、normalization経路差、条件付きcategorical上界とprovenance。
数値Frobenius上界はrepresented matrixのpilotであり、reference計算・減算・norm等の丸め包囲ではない。
`reference_numerical_uncertainty`、`rigorous_bridge_bound`、native synthesis、actual-law error_evidenceはnullを保持する。
単に観測したbridge距離を、無条件の実装精度上界としてStage Cへ自動注入しない。

## Categorical差の伝播と誤差会計

同じexact unitary trajectory map、ideal shellでの`|z(word)|≤1`、独立抽出という条件で、
`|E_q z−E_p z|≤2 TV(p,q)`。N_m=32×2^m個のeventには`min(2,2 N_m eta)`のcoupling上界を使う。
既存`c_all_law_signal_ca.categorical_preview`を再利用し、旧16×4の乗数は使わない。
m12はkernel4,096 occurrence、event131,072。Raw complex signal単位なのでA_mを再乗除しない。

Native shellの不完全さは別のsynthesis項、saved PF/corrected-RTEは既存accuracy内でA_m倍、DFはenergyでHaとして一度だけ加算する。
Bridge項にPF/RTEそのものの近似誤差を再加算しない。
PCG64 joint-stream差は今回のIID-gridモデルの**対象外で未資格**であり、既知0という扱いではない。

| 成分 | 処理・残る資格 |
|---|---|
| PF / corrected finite-RTE | 保存referenceを保持。新しい誤差評価なし |
| Native synthesis | 最終whole-P0のall-word/occurrence/shell boundが必要。現在未取得 |
| Categorical | IID-grid内の正確なTVとcoupling。実PCG64保証ではない |
| Reference bridge | 現r32/s1/h=.1のp/qとsaved CARを数値評価するStage A。未実行 |
| Reference numerical uncertainty | Bridgeに未包含の部分だけ別管理。Unknownを0にせず、必要なら数値的guardの根拠を準備する。厳密包囲を必須とはしないが未評価を消さない |
| PCG64 joint stream | 未資格・対象外。IID-gridと一致したと表示しない |
| DF / RPE statistical | 既存の一回加算／beta・alpha・shots政策を維持 |

## Stage B：Whole-kernel P0 uniform評価の到達点

前回の有限primitive/transitionとcompiler監査を再利用した。Hybridを第一候補とするが、
raw finite familyとwhole-P0の全word一様誤差を混同しない。
Raw構造では9 unique basis、4 directed head transitions、tail最大32 run。
Basis出入りはtail最大64＋head4＝最大68 instanceと4 relative transitionだが、これは**raw構成数**で、
展開後のelementary gate数やP0再合成誤差の上限ではない。

追加の[静的source確認](../../artifacts/research_c_iid_grid_all_law/2026-11-03/technical/p0_static_constraints.json)では、
Qiskit1.3.0 `DoWhileController` のsource-default guardが1,000 iterationで、超過時は`PassManagerError`になることを確認した。
これはcompilerの既存guardであり、新しいCPU/wall capではない。
有限回で返ること、全wordで成功出力を返すこと、各run/translationのrewrite数、一様operator誤差は別の問題である。
`Optimize1qGatesDecomposition.run`はRust処理へ渡し、event境界を越えるmatrix結合・再合成・位相更新を含む。

A（構造的一様法）は全到達input、run長、翻訳・fusion/iteration経路、数値再合成/phase/referenceの誤差を覆う必要がある。
B（有限family診断）は特定のboundary/sign/phase/fusion条件を検査できるが、全wordを覆わない。
C（hybrid）はAを主証拠とし、Bを実装整合性検査に使う。今回の人工cross-event P0一致もBであり、Aの証拠ではない。
Control相対位相、actual inverse、coherent反復、head/X/Y、full512または正当に不変性を示した領域も必要。
MPFR160/seed17/opt1/approximation_degree1、finite32残差、単なるunitary距離≤2では有用な一様誤差を得たとしない。

**結論：`P0_UNIFORM_NOT_ESTABLISHED`。** 入力領域・変形回数・一様再合成/phase/reference誤差と全word成功の論証が未完了。
Compilerをevent別compile/barrier化等へ変更して救済していない。Stage Bのscience callsは未固定で、数値を恣意的に入れない。

## Stage C：IID-gridを既存accuracyへ接続する方法

[条件付きadapter](../../src/trottertracks/c_iid_grid_accuracy.py)は新しいmodel識別を持つ。
既存`c_all_law_signal_ca.attach_evidence`のidentity/13round/単位/二重計上/coverage検査を再利用し、
actual streamの前提とstream差は**未資格のまま**、別のIID-grid資格を作る。
既存actual-law packetの`error_evidence`はNoneを保持し、IID-grid recordを旧`c_pr_aware_df_stage_c.qualification`へactual-law証拠として渡さない。
有限family・未知のsynthesis/reference成分・欠損roundではIID-grid資格も未成立。

既存`RoundEvidence/assess_round`のlaw識別は限定されているため、旧SOURCEを変更せず**reference transportの計算**として再利用する。
4つの既知の非PF/RTE raw成分（synthesis、categorical、reference bridge、reference uncertainty）の合計を既存の加算raw-error slotへ渡す。
内部reference lawは`ideal_finite_RTE_law`で、そのreference自体とのself-law差を0とする。
これは未知のPCG64差やcategorical差を0にする操作ではない。Categorical等は合計に全部含め、元の成分名/値を各planに残す。
計算結果のobserved modelは明示的にIID-gridへ戻し、`engine_reference_law`とslot対応も保存する。
`actual_implemented_law`というtagはこのengine呼出にも使わない。

計算式は元と同じ`delta=A*(eps_PF+eps_RTE)+eps_synthesis+eps_categorical+eps_bridge+eps_reference`。
Knownな4成分が揃う前に呼ばないので、unknownを加算から落とす経路はない。
`assess_energy`ではDFを一度だけ加算し、旧数値pilot・own-ground/alias・beta/alpha条件を保持する。
一部roundがradius/phase条件不成立なら後続とenergyを止め、r/h/K/CA/誤差上限/solver政策を変えない。

これにより、旧ideal、IID-grid数値的十分条件、実PCG64、厳密保証、actual CAを別々に保持する。
必要な数値入力が人工的に揃った場合だけ`IID_GRID_CONDITIONAL_NUMERICAL_SUFFICIENT_CONDITION`となる。
現実のStage B boundとreference uncertainty等は未取得なので、今回の実H4 Stage CはNOT_READY。
`actual_CA_achieved/rigorous_CA=None`、`PCG64_joint_stream_qualified=False`。実測やsimulationの達成を認定しない。

## 実行Stage・38-call限定契約

[未承認契約](research_c_iid_grid_all_law_contract_unapproved_20261103.json)は`UNAPPROVED_EXECUTABLE_STAGE_A_ONLY`。
Stage AはP0の未資格と独立にreferenceの数値対応を調べる限定科学実験であり、Stage B/C/Dへ自動進行しない。

| Stage A登録単位 | 最大calls |
|---|---:|
| Saved metadata / identity / input / sector / array binding / model replay | 各1、計6 |
| Current modelと保存CARのfactor binding | 9 |
| Law準備・保存λ/τ/identity/K0結合 / atom context / exact choice-grid解析 / 全216 atom audit | 各1、計4 |
| Ideal p / grid qの有限event平均 | 2 |
| Head / kernel bridge / 共有round-power cache | 各1、計3 |
| Round bridge | 13 |
| Summary | 1 |
| **合計** | **38** |

Model replayは既存one-body基底の小さな対角化を含み、physical/own-DF ground eigensolveではない。
全216 atom audit/meanは旧Bridgeと同じgroup単位、9 factorと13 roundは別々のcall。
4本のpower chainは12段のsquaring、計48積を共有し、各roundで同じpowerを再生成しない。
新しいPF/corrected-RTE kernelやPF/RTE誤差は生成せず、head/event/reference powerの再生をbridgeとして明示する。

Stage Bはmethod/全word coverage未確定でcalls=null。
Stage C adapterはqualification1＋round最大13＋energy1＋summary1＝最大16 callsで、全Stageの上限ではない。
Unknownなら計算へ進まず、最初の不適格roundでpartialを保持する。
Stage D（実PCG64、厳密包囲、shot-level simulation、actual CA）は別途判断。
旧1,231-call・native32・旧47-callを新実行の予算や許可へ転用していない。

## 人工検証・SOURCE・passive preflight・資源

[テスト記録](../../artifacts/research_c_iid_grid_all_law/2026-11-03/technical/tests.json)：**新30＋既存152＝182 unique PASS**。
53-bit境界/整数計数、zero/underflow、不正確率、独立再抽出とkernel再利用の差、signed atom/scalar/K0、
r32/s1/h.1とλ/τ/identity binding、p/q meanとraw CAR期待値、PF/RTE/DFの二重計上防止、
cross-event P0・Gaussian/relative/control phase/actual inverse・coherent反復/shell、有限family資格拒否、
IID/actual/rigorous/actual CAの分離、unknown null、phase不成立STOP、drift/fresh/retry/resume、
mock38-call Stage Aとpartial/STOPを確認した。実H4の数値値を新しい計算fixtureにせず、metadata/hashだけを読む試験と分けた。

[技術SOURCE](../../artifacts/research_c_iid_grid_all_law/2026-11-03/technical/SOURCE.json)は154file、保存inputは19file。
旧150 SOURCEはそのまま保持し、4つの追加code/entry/testだけを加えた。
[Passive preflight](../../artifacts/research_c_iid_grid_all_law/2026-11-03/technical/passive_preflight.json)は数値libraryのimportを禁止してもPASS。
Science call0、Stage A fresh出力不在、旧STOP/source/input、model/38-call scopeが一致する。

- Branch：`codex/research-c-iid-grid-all-law-readiness-20261103`
- Worktree：`/tmp/prt-c-iid-grid-all-law-readiness-20261103`
- 実装固定commit：`55fc3f3015e95bf181d4dcf155d74c90a19abc3e`
- SOURCE SHA256：`95897050fb5a36ede0f60126da9fee3d4458fcb83d9c7afcb08b4f3a0cccd38a`
- 契約SHA256：`f9328123c2655ec36ba590556231b9f19ab17bee68f065c0cb422d2d09f2bee2`
- Future fresh output：`artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01`（未作成）

既定の読取り確認：

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B scripts/run_research_c_iid_grid_bridge.py
```

**別の明示許可を受けた場合だけ**同じcommandへ`--execute`を付け、固定Stage Aを一度実行する。
Source/input driftやfresh既存なら拒否し、retry/resume0。Flagは誤起動防止であり、承認証明ではない。
正常完了/FAILともpartialとSTOPを保持し、Stage B/Cや追加samplingへ進まない。

CPU32・available RAM約53GiBを確認。任意のCPU/RAM/worker/wall capは設けない。
既存SectorCAR・operators・head cache・4 power chainを共有し、13独立roundを既存resource_planで並列化する。
BLASは自job内だけ調整する。今回のtestsはtask-local cache/BLAS1/bytecode・pytest cache無効で、共有環境を変えていない。
未測定speedupを主張せず、重い全word列挙を既定にしない。

[技術STOP](../../artifacts/research_c_iid_grid_all_law/2026-11-03/technical/STOP.json)は新実H4科学call0。
旧SOURCE/raw/契約/grant/ledger/marker/STOP、dirty checkout、共有venv/OS/global cache/他jobを保持した。
Model採用と技術準備を研究概要・当日ノートへ追記するが、旧Claim・数値結果・Evidence Map・validation manifestは変更しない。
既知のmanifest旧schema FAILを修復したり、global validation PASSと主張したりしない。
公開後はremote SHA/tree・公開本文・主要blobを照合する。

**Mandatory STOP。** 次の担当はユーザー：固定38-call Stage Aの数値reference bridge実行を許可するか判断する。
許可後のCodexは同じSOURCE/input/契約だけで一度実行し、数値観測・条件付きTVと未資格を報告してSTOPする。
Whole-P0一様法、Stage Cの不足bound、実PCG64、厳密保証、shot-level/actual CA、重要レビューはその許可に含めない。
