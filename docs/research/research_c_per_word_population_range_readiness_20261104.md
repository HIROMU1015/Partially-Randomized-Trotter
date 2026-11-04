# 研究C：per-word誤差包囲・population rangeの技術具体化

2026-11-04 JST。**Compiled→raw→literalのper-word包囲器と、全q-supportのnorm/range評価器を接続した。人工41件と適合回帰を検証し、未承認Stage Rを256 logical unitsで具体化した。** 現H4の誤差・range・分散・CIは新規評価していない。校正法、threshold、N、δ_calは未採択。Stage P/C/Aを自動実行しない。

## 起点・SOURCE・保全

独立branch `codex/research-c-per-word-population-range-readiness-20261104`、起点 `bd560a45410409573ae849ca5aaf4050343910bf`。[前段の期待bias設計](research_c_iid_grid_expected_bias_readiness_20261104.md)と旧未承認契約を原本のまま継承した。正式共通指示2026-10-25版の原本§8.1・§8.4・§10を確認した。技術実装・人工検証・科学契約準備だけが今回の許可範囲であり、承認証明・承認ID・原文保存・新grantは作らない。

| 固定証拠 | 結果commit | 科学SOURCE・状態 |
|---|---|---|
| 固定4word | `3649b75af191916daf44abb91e14d34c43b83e8f` | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`、実装`6e8475cc871916473d48fd38fb4b4a0297e84d15`、86 COMPLETE/STOP |
| B2 reference上界 | `b7a3db09e395d36965e5b9dc28ae4a299db3b2f0` | `01baf2b5e400773e00a5ae548941265f85b9882b`、27 COMPLETE/STOP |
| Stage A IID-grid | `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53` | `ddd8333488e07a3b3af9095b5cb193ef9f90b3c3`、38 COMPLETE/STOP |
| B1 Whole-P0 | `69c9a764aee66cf2ca79ab29bd3145a027385bcc` | 実装`5fc9e6a18e016bfa0725d18edd1c7fc4589a0075`、uniform未成立、準備のみ |
| B1/B2後続統合 | `891aa78f47e66184eaf8a6d527ca8773ba6bbdd0` | 実装`c22d79cfd5c6a1202587c2211031626cb8023df3`、旧`90a2a7b...`追補、NOT_READY |

前段の[公開本文5件のblob照合](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/public_inputs.json)と固定metadataを再利用した。今回は直接起点の公開報告とB2実装SOURCEも取得し、固定blobと照合する。契約・SOURCE・STOP・候補入力の照合はbyte/JSONだけで、保存NPZ/QPYを科学的にdecodeしない。旧科学結果・SOURCE/raw・grant/ledger/marker/STOPを変更せず、Evidence Mapの科学Claimを更新しない。

対象はLinear H4・隣接1.00 Å・STO-3G、Nα=Nβ2、sector36/system256/controlled512、number CDF8、元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1、m0..12、CA=.00159 Ha、DF reserve=CA/8。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。元g補正κ/scalar/frame順、own-DF ground oracle・準備費用除外を維持する。P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、既存relative head/full-frame tail/basis_key共有、fenced serial、actual inverse・head/X/Y shellも変更しない。

## 数学的targetとheadの適合性

第一estimandは前段のまま `Δ_m=|E_q z_compiled,m−E_q z_literal_atom,m|`。Per-wordはC=actual compiled gate列の理想数学作用、R=raw gate列の理想数学作用、D=diag(I,T)。Tは同じ32event・signed CAR Z/ZZ、元head順と逆順、tail identity phaseを含むliteral reference。観測したbinary64 Operator行列を真の作用と同一視しない。

**Headの定義をSOURCEから明示した。** 固定4wordの`c_pr_integrated_performance.native`→`stage_b2.reference_context`は、元h/g/J/Uから既存`raw_base/raw_frame`でfull256のG0..8をbinary64生成し、そのG0..2をreferenceのheadに渡す。一方、B2の`c_reference_bound_contract._head_task`は保存sector36 generatorのbit値をexact dyadicとして指数化する。新評価器では前者の**同じ独立full-CAR producerの出力bit値**を数学的headとし、入力をexact J/Uで再計算したheadやHermitianに射影したheadへ黙って置き換えない。これは既存full-reference targetへの適合である。

Stage Rはfull head3件と保存sector head3件のbit同一性を記録する。同一ならこのhead identity legを再利用でき、異なるならB2接続に追加head bridgeが必要。相違を丸めで消したり、B2のtargetを付け替えたりしない。Head bit不一致だけでは新literal rangeの算定自体を無効にしないが、accuracy transportの資格は保留する。

Literal real Uは保持する。Polar Uや修正κを科学targetへ導入しない。非直交Uに対するZ/ZZは厳密involution/unitaryと仮定しない。CとRは受理済みreal-angle理想native gateの積なのでunitaryだが、全q-wordでP0コンパイラが成功するというtotalityは別前提である。

## Per-wordの二leg

新実装は [c_per_word_population_range.py](../../src/trottertracks/c_per_word_population_range.py)。既存`c_pr_aware_df_fixed_certificate`のgate記録・Arb192理想gate作用・recentered column radius、`c_pr_aware_df_fixed_car`のMatrix/Arithmetic/pack/unpack/full_partition/column comparator、`stage_b.atom_operators`、`rte.event_unitary`、既存head schedulerを呼ぶ。新しい回路生成engine・CAR engine・数値シミュレータは作らない。

| Leg | 実装と資格 | 今回取得したもの／未取得 |
|---|---|---|
| C→R | `fixed_certificate.compare(compiled_record,raw_record)`を再利用。Arb gate evaluation＋全入力列の再中心化半径、outward Frobenius≥operator2 | 人工入力で包含を検証。実H4の新boundは未取得 |
| R→D | `_literal_block`で全spin-count blockのliteral atom・head exp・event積をArb包囲。既存full-controlled column actionでRとの距離を評価 | 新adapter実装・人工検証済み。Polar bridge加算なし、非unitary Dにcap2なし |
| C→D | 同じraw record digest・target/input/head/word/angle/phase identityで二legの和 | 三角不等式の包囲経路を実装。これは各指定word用で、全word synthesis保証ではない |

H4の全256 Fock基底は25個のspin-count不変blockへ**漏れなく分割**でき、最大block36。Head/atom全体をblockごとに包囲するのは、個々のGaussian gateをphysical sectorへ射影する検証とは異なる。Raw/compiled側は全512入力列・全出力成分を評価し、control=0 identity/control=1 T、相対位相、sector外作用を保持する。Headがこのblock分割に厳密に不変でなければ明示FAILする。

`build_literal_reference`はJ/U・frame・signed atom・support・event順・保存角度bits・h/r・full head bitsをbindする。Headの時間係数はh/2、forward/reverseの順序は既存scheduler、tail scalarは一回、核scalarはG0内のまま。CAR atomの符号とZZ順序を変更しない。

`compare_literal`は全columnの数値半径を既存Arb comparatorへ含め、outward Frobeniusへ換算する。C/R同士のcap2は理想unitary gate同士に適用できる。R/Dと最終triangleにはcap2を適用しない。人工の大きな非unitary targetで真の誤差が2を超えるケースを通し、上界を2へclipしないことを確認した。

Arb精度は既存192bit、python-flint0.8.0/FLINT3.3.1、binary64入力はexact dyadic。`acb_mat.exp`とcos/sin・積・normの区間半径を捨てない。[python-flint公式API](https://python-flint.readthedocs.io/en/latest/acb_mat.html)は現在0.9版の資料だが、今回試験したruntimeは既存0.8.0のまま。旧fixed-CAR公開APIはpolar targetと既存κ Hermiticity条件へ進むため、その最終target APIは使わず、適合する内部算術とcolumn作用だけを薄く再利用した。

資格は、observed residual／numerical uncertainty estimate／validated per-word enclosureを別kindで保持する。今回の人工PASSは現H4のbound取得を意味しない。B2の平均sector reference上界をper-word full512へ移送しない。

Actual inverseは同じ実現回路のactual inverseを検査する。X/Y信号のbra側の随伴はBorn則の一部で、inverse samplingを追加しない。Head/readout shellは別segmentで、将来Stage Pの三つのshell certificateを一回だけ会計する。Raw/compiled inverseの復帰検査と前向き信号の包囲を混同しない。

## 全supportのnorm/range

`grid_support`は保存catalog216件とexact 53-bit q countの対応を検査する。Populationは**qが正の確率を持つlength32 wordすべて**。Count=0というexact記録があるカテゴリだけを除外できる。微小正count=1も保持し、確率閾値やunderflow観測でカテゴリを削除しない。現在候補の正support数を今回新しく集計してはいない。Stage Rでは216件全部のevent boundを記録し、最終maximumは正q-supportに限って取る。

Uのspatial列uに対して、CAR annihilatorのnormは||u||、density `n_u=a†(u)a(u)`はspectrum {0,||u||²}、従って

`||Z_u||=max(1,|1−2||u||²|)`。

Gram `UᵀU`と各列の二乗normをexact dyadic rationalで計算する。FloatでIを引いた残差に置き換えず、polarも取らない。ZZ atom AにはZノルムの積bを使う。Support2で同spinのとき、CAR関係より

`||[n_u,n_v]|| ≤ 2|u†v| ||u|| ||v||`、従って `||A†−A|| ≤ 8|u†v| ||u|| ||v|| = k`。

異spinのdensityは可換なのでk=0。Signed係数の符号はこのnormを変えない。イベントE=cosθ I−i sinθ Aでは、**cos²θ+sin²θ=1という恒等式を保持して**

`||E||² ≤ 1+sin²θ(b²−1)+|cosθ sinθ| k`。

Arbで右辺を上向き評価する。単純な`|cos|+|sin|b`の32乗を既定にせず、Uがexact orthogonalならb=1,k=0からnorm≤1を正確に返す。非直交列・同spin ZZの非Hermitian性も許す。これは全合法atomの構造的範囲であり、観測sample最大ではない。

Head Gの非Hermitian部分を `A=(G−G†)/2` とすると `||exp(−itG)||≤exp(|t| ||A||)`。全spin blockのexact anti-partのFrobenius上界を取り、head半時間h/2のnorm H_iを計算する。Hermitianである場合だけH_i=1となる。入力Gの対称化・κ補正追加はしない。

従って全q-wordについて

`||T(w)|| ≤ H_0² H_1² H_2² · (max_positive_q ||E||)^32`、`a=max(1,この積)`。

D=diag(I,T)なので `||D||≤a`。成功した理想P0出力Cはgate定義から `||C||=1`。このCが全wordで定義されるというtotal compiler premiseが揃えば、**0≤d(w)≤M=1+a**。M=2とするのはa≤1を根拠付きで得た場合だけである。

Stage RはDの全word normを評価できるが、コンパイラの全domain totalityを認定しない。Compile失敗wordを黙って棄却・再抽出してqを変える運用も認めない。`calibration_inputs`はこの前提が不足すればconfidence算定へ進まない。

13roundに対してD積のnorm≤a^(2^m)、理想shellのraw軸差範囲は±(1+a^(2^m))。Kernel期待operator差を伝播する場合は前段の両norm/shell式を再利用する。新`propagate_calibrated_mean`は同じscalar式をArbで包囲し、m12のFraction巨大化を避ける。非常に大きな有限値はdyadic mantissa/exponentで保存し、表現できない旧slotをnullのままにする。これは誤差clipではなく、structured endpointとNOT_READYを返す出力方式である。

**数学的に有効な有限Mと、RPEに有用なMは別。** a−1が小さくてもa^4096と校正幅の反復増幅を確認する必要がある。現H4のa/Mは未取得なので有用性を数値認定していない。

## 校正法の再比較

新しい統計engineを別に作らず、前段`mean_upper/paired_round_upper/failure_accounting`、旧`zero_failure_tail_limit`を使用する。Empirical Bernsteinだけは既存実装にないため、interval誤差を含む最小scalar adapterを追加した。

| 方法 | 対象と必要資格 | Confidence・N依存・実行負担 | 今回の判断 |
|---|---|---|---|
| Threshold tail | d≤tの包囲判定＋全population M。True dを含むsample intervalと事前固定t/Nが必要 | Bad0なら既存exact binomial上界p≤1−δ^(1/N)、μ≤t+(M−t)p。O(M log(1/δ)/余裕)。各wordにnative/reference certificate | Straddle/未判定をgoodにしない。Bad/straddle>0では既存Hoeffdingの保守的indicator上界へ接続できるが、採択前にその分岐も固定する。全bad0を事前仮定しない |
| Variance-sensitive | d∈[0,M]、enclosureを用いたsample variance上界 | Maurer–Pontil Thm4：mean上端+sqrt(2 V_upper log(2/δ)/N)+7M log(2/δ)/(3(N−1))。Varianceが小さくてもrange/N項が残る | 小varianceの資格を旧4word/native32から推定しない。新adapterは人工検証済み、科学method未採択 |
| Direct paired round bias | 各mのsigned X/Y差の数値interval。Range±(1+a^(2^m))、oracle/state/shell対応が必要 | 52tailのunion等。Kernel→roundのoperator telescopeを避けられるが、各sampleで2^m個のfresh occurrence、全round合計8191。Matrix-free actionでもbuild/compile負担は残る | 前段adapterを再利用。Per-word operator boundをround signal evaluation誤差へ同一視しない。現時点の初回候補には重い |

Variance adapterではsample interval中心x̂_iとradiusη_iを使い、centered projectionの収縮性から `sqrt(V_true)≤sqrt(V_mid)+sqrt(sum η_i²/(N−1))` とする。Meanは各上端の平均なので数値的不確かさを別に取り落とさない。新しい分散値をH4で計算してはいない。式の一次資料は[Maurer–Pontil (2009), Theorem 4](https://arxiv.org/abs/0907.3740)。固定N以外のoptional stoppingや複数法の事後best選択を認める式ではない。

前段δ_cal=.05の楽観設計例は**既存値の引用**：kernel Hoeffding約128.7億標本、zero-bad tail約55.5万標本、direct約1457万kernel occurrence。これはa=1、raw room=.25、数値/shell差0などの設計例で、現H4の実効roomや真の分散を示さない。新しいa・sample原価が未取得のため、今回N/threshold/δを確定しない。

同じM、同じ極小varianceを仮定した比較でもempirical Bernsteinのrange/N項は消えない。Tailとの優劣はgood thresholdと実variance・enclosure幅・per-word原価を取得してから判断する。都合のよい誤差相殺、母分散、fresh性を仮定して実行可能と扱わない。

## Confidence・accuracy・未資格事項

RPE α_total=.01、α_axis=.01/26、β_total=π/6、β_stat=π/12、CA=.00159を保全した。Calibration δ_calは別管理。校正event上でRPE failure≤.01、校正込み無条件failure≤min(1,.01+δ_cal)。新joint .01への再配分は未採択。

Kernel校正event一つを全13roundへ伝播するならδを13回加算しない。Directの各round/axis tail配分はunionで同時confidenceを作る。Calibration sampleとmeasurement shotは別。実PCG64の再現可能なsimulation streamを数学的IIDの証明としない。固定4word/native32をfresh cohortに再分類しない。

`calibration_inputs`はtarget/head/word/angle bits、per-sample enclosure、population Mの導出、compiler-domain premiseを確認して前段confidence関数へ渡す。ゆるいsample upperが別に証明したMを超える場合は、明示したinterval交差だけを許す。Sample最大からMを定める処理を拒否し、observed residualだけならUNRESOLVED。

Whole-P0 uniform synthesis、probabilistic expectation、finite-word correctness、B2 mean reference、actual PCG64は別資格。前段の別期待値qualificationを保持し、all-word slotへ偽装しない。PF/RTEはA倍を一度、categorical p→q・reference bridge・reference numerical uncertainty・native synthesisは互いに異なるraw transport leg、DFはenergy側、RPE統計はmeasurement failure側。B2合計と二項、direct totalと各leg、既包含rounding radiusを二重加算しない。

B2のsector mean上界は引き続きper-word reference不確かさを覆わない。Literal categorical unitary premiseもtrueにしない。Norm-growthを使う非unitary couplingへ拡張する技術余地はあるが、今回その科学boundを評価していない。Head同一性が未確認、total compiler premise・shell certificateも未取得なので、**Stage C ready=false、actual/rigorous CA=null**を維持する。

## 未承認Stageとcall内訳

[未承認契約](research_c_per_word_population_range_contract_unapproved_20261104.json)。科学permissionは今回なく、実行済みcalls0。専用entryはdefault passive、`--execute-stage-R`は固定Stage Rへの別途実行許可後だけ使用する。

| Stage | 固定・準備状態 | Callと出力 |
|---|---|---|
| **R：Population range** | **実行可能な未承認契約、256 units**。同じ保存endpoint/元入力/atom/table/angle、Arb192。回路・sampling・PF誤差・shotsなし | a/M、全support、13round norm growth、head3件のB2/A同一性、SOURCE/calls/STOP。全word compile totalityの証明は含まない |
| P：Per-word | 新しい1wordの**65-call案、非実行**。F6辞書順最初ZとF7辞書順最後ZZを16回交互。構造だけで選び、旧4wordと不一致をmetadataで確認 | 48 setup＋9 native/check/metrics＋literal reference1＋二certificate2＋triangle1＋shell3＋summary1。専用science runner・post-R head hash binding・判定政策を実行前に固定する必要あり |
| C：Calibration | 非実行、N/δ/method/threshold/call上限null | R/Pの数値資格・原価とtotal-domain条件を確認後に設計 |
| A：RPE accuracy | 非実行、call上限null | 全13roundの適格raw biasとdisjoint transportが揃った場合だけ旧accuracy算術へ接続 |

Stage Rは **setup13＋head norm3＋frame Gram8＋event norm216＋population組立1＋13 growth＋summary1＋support1=256**。Setup13の内訳はevidence/input/candidate/native model/CAR各1、raw head factor3、candidate bind1、catalog1、saved grid1、saved sector heads1、head binding1。Word集合の全列挙はせず、216eventだけで全wordのsubmultiplicative normを作る。R完了後にP/C/Aへ進まない。

Raw head3件は独立に並列化し、affinity/quota/実空きRAMから既存resource_planを使う。BLAS/OpenMP等とのoversubscriptionを抑えるため自jobのthreadpoolだけを調整する。Scalar Gram/eventの小計算はArb process-local precisionを保持し、起動overheadを避けて逐次。重いStage Pは既存25block/512column並列・immutable head/atom cache・local block matrix actionを再利用する。Nested process poolを同時に多数起動せず、一層の並列計画とする。CPU/RAM/worker/wallの人工capなし、共有venv/OS/global cache/他job/lock変更なし。

Stage Rの主データはhead3枚×256²（complex128で約3MiB）とCAR bilinear/cache。Arb full512密unitaryは作らない。Stage Pのfull-column gate作用・per-word referenceは既存finite pilotより重く、現候補の一件原価は未知。固定4wordの既存wall約58.95秒をArb処理の見積もりへそのまま流用しない。

SOURCE/input/位相/control/target/数値非有限/非不変blockでcorrectness FAILなら依存taskを止め、開始済みを回収し、partial evidenceとSTOPを保存する。Normが極端に大きければstructured endpointまたは明示STOPを保持し、precision変更・clip・unknown→0・別条件fallback・retry/resumeをしない。Bad-tail threshold超過は統計上のbadで、correctness FAILとは区別する。

## 人工・回帰確認と技術成果物

新人工テストは、両leg・literal非unitary・signed Z/ZZ・relative/head方向・control phase・head/event順・actual inverse・shell・全512列・数値半径包含・target拒否、exact/非orthogonal Uのnorm、同spin commutator、exact q-count support、非Hermitian head、巨大growth、M=2/sample max誤用拒否、tail bad/straddle、variance interval、confidence・identity/fresh/STOPを含む。人工の一体Stage Rは256/256 unitsで完了し、native build/samplingを呼ばないことを確認した。

**新41件＋適合回帰226件=267 PASS、旧4件を理由付きで除外**。[tests.json](../../artifacts/research_c_per_word_population_range/2026-11-04/technical/tests.json)／[tests.txt](../../artifacts/research_c_per_word_population_range/2026-11-04/technical/tests.txt)を実際の結果正本とする。旧SOURCE依存4件は直接起点でも失敗することが前段で確認済みのため同じ理由で除外し、旧契約/STOPを修正しない。[前段の起点再現ログ](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/historical_parent_checks.txt)。ローカル人工PASSはCIの外部再現やH4の科学PASSではない。

[技術SOURCE](../../artifacts/research_c_per_word_population_range/2026-11-04/technical/SOURCE.json)、[passive preflight](../../artifacts/research_c_per_word_population_range/2026-11-04/technical/passive_preflight.json)、[静的確認](../../artifacts/research_c_per_word_population_range/2026-11-04/technical/static_verification.json)、[技術STOP](../../artifacts/research_c_per_word_population_range/2026-11-04/technical/STOP.json)を保存する。科学fresh出力stageR/P-run01は作らず、新実H4数値評価/array decode/compile/sampling/CI/shots/重要review0。既存ファイルへ差分がないことを確認し、公開後はremote SHA/treeと主要本文blobを照合する。

## 次の判断

推奨は**Stage Rの256-call限定科学検証を先行**すること。理由は、重いper-word native/certificateを追加する前に、現候補のaとm12のnorm-growthが有用な範囲にあるか、B2のhead identityが再利用できるかを判定できるため。

次担当は**ユーザー／研究C**：固定Stage R契約の実行可否を判断する。明示許可後に**Codex**がfresh StageR-run01を一度実行し、結果・partial evidence・callsを公開してSTOPする。Stage Pの方法/判定資格、calibration法/N/threshold/δ_cal、新joint failure budgetは別の未採択事項。今回の技術準備だけでその採択や実H4計算を開始しない。**公開照合後mandatory STOP。**
