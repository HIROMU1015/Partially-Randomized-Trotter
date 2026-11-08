# 次の最小検証：現wordのdefectと保存recordのinterface

2026-11-08 JST。基準 **f0b4b143e9c01115cd010f4dd9f88354ba06c3b4**。本書は未実行の提案である。今回許可された静的調査は[REPORT](REPORT.md)の時点で完了しmandatory STOP。以下の科学数値評価・回路fixture実行は、固定scopeへの別途明示承認が必要。

## 1. 共通の入力・境界

固定科学条件はREPORT §1を引き継ぎ、旧SOURCE、C0 299 word、43基本word、85 certificate、S1/S2、V0/V1とSTOPを保全する。中央precision探索・S1再実行・S2 frontier再計算を含めない。

[STATIC_FINDINGS.json.xz](../../../artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz)に保存したarchive/member/context、literal、native位置、word hashを使用する。Archiveは基準commitの保存版に限定し、target literalをdecimal再丸めしない。A1とB1は独立して判断でき、結果を得るために片方から他方へ自動拡張しない。

再利用SOURCEは以下である。

- [静的crosswalk](../../../scripts/audit_pr_reference_s3_s2_static.py)：固定template／record／contextの読取部分。実行済みartifactのSOURCEを後から変更しない。
- [exact_word_matrix](../../../src/trottertracks/track_c_h4_synthesis.py)：保存wordのexact cyclotomic表現からinterval行列への変換。Wを含む。
- [N4-B consumer検証](../../../scripts/validate_pr_reference_n4b_consumer.py)の`compressed`と[CT1](../../../src/trottertracks/pr_reference_ct1.py)の`apply_batch`：native gateのwire対応・疎適用の参考。現在のnumpy complex128計算自体はstrict証明に使わない。
- [N3-B1b-0 runtime検証](../../../scripts/verify_pr_reference_n3b1b0_runtime.py)、[lower_native](../../../src/trottertracks/track_c2_b_verify.py)、[consume_native](../../../src/trottertracks/pr_reference_common_word_consumer.py)：named word／inverse／wire／terminalの既存処理。
- [N5-B1 scalar検証SOURCE](../../../scripts/verify_pr_reference_n5b1_scalar.py)：common-joint-scalarを分離できる条件の参照。旧fixture一式は再実行しない。

追加する場合も各pilotの局所runner／fixtureに限る。実装後に科学的に意味を持つ入力と誤差方式を固定し、人工testを通す。既存のdouble中点化や1-qubit RZ用`strict_guard`を、multi-qubitの厳密defect保証として流用しない。

## 2. A1：現wordの2 pairに対するinstruction-body defect

### 固定対象

43 literalが対応する3 SOURCE basisのうちID辞書順で最初の

`df-basis-8b103ae5327e20e021dcbb2eb64919b67de85b194d4d6c38b6093414f37e2fbe`

を使う。これは最良／最悪basisの選択ではなく、有限interface診断の決定規則である。

| Arm | 保存archive | Member／context |
|---|---|---|
| B′ | `artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz` | `catalogue/rare0_Y0.json`、`[false, false]` |
| Reference | `artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz` | `catalogue_boundary/reference_ccx1_rare0_Y0.json`、`reference_ccx1_rare0_Y0` |

各々の`basis:<ID>`と`basis_inverse:<ID>`の8-qubit保存native命令列を対象とする。Current C0＋N4のpositive sequenceを使い、negativeは全wordのactual adjointにする。Word再選択、新word、precision変更、native再compileは含めない。

### 必要な科学callと実装差分

科学評価は4 instruction body（2 arm×forward/inverse）を構成し、2件の

`eta_body = ||C_body - A_body†||_2`

の**interval上界**を得る範囲に限る。保存4 bodyに現れるdistinct positive sequence hash集合を開始前に固定し、各々を`exact_word_matrix`で一度だけ変換する。SXはHSHのexact表現、X/CXは同じwire／controlのexact演算とする。

提案する最小実装は、既存疎gate適用を8-qubit interval行列へ拡張し、mpmath1.3.0・192-bit固定で積を囲い、差行列のinterval Frobenius上界をspectral上界として出す方法である。Wordのmatrix変換は必要だが、新たなbackend strict guard呼出しではない。Float SVD／phase-fit／最良scalar最適化は使わない。計算順とintervalの包含性を人工exact例で検査する。これは**今後実装する数値runnerの設計**であり、今回の静的helperに数値経路を追加して実行したものではない。

保存local loweringにnative global phaseの対応がない場合、得られるのはinstruction-bodyの上界だけである。Full strict `eta=||C-A†||`は未資格に残す。Common scalar条件が別にSOURCEから証明できるかを記録しても、未知prefactorを0と置かない。

### 判定とSTOP

- Finite上界を得た場合：2 pairのbodyに限定した保証。共役式に必要なeps_f／eps_D／W／full phase／全contextは別途未評価と明記する。
- 上界が大きい／intervalが広い場合：この固定診断では鋭い保証を得られない。実誤差下界やRPE失敗とは扱わない。
- Missing word／phase／仕様不一致：該当資格をUNQUALIFIED。補完・再合成・再compileしない。
- 2 pairを完了又は入力不足で停止したらmandatory STOP。精度倍増、他basis／context、全support、RPE、T会計へ自動拡張しない。

このscopeは現構成でdefect評価の経路を作れるかの診断であり、S3の全law採択ではない。A†を直接生成するadapter案は別の構成変更であり、A1に混ぜない。

## 3. B1：保存候補1 recordの有限interface検証

### 固定対象

43 frame literalの辞書順最初のkeyをinterface fixtureとして選ぶ。Guard／Tを再比較して最適候補を選んだものではない。

- Literal：`0x1.0b1d9883f9213p+1`。
- Archive：`artifacts/track_c2_b_pilot/2026-11-06/raw_synthesis.tar.xz`。
- Member：`synthesis/1c223a1ac0ccd6b7e2729dae93da988f4bf9d3bbc10ac1bba0500ed4ebc3029f.json`。
- 元spec：`0x1.0b1d9883f9213p+1:10^-13:pygridsynth2.0.0:strict:seed17`。
- Positive sequence SHA256：`981736be8f2688787aa2e1da4598b9d42b016abf32171fcc27038f88f55c0155`。保存global_W=7。
- SOURCE対応例：B′ `catalogue/rare0_Y0.json`／context `[false, false]`、`basis:df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2`、native position423、wire `[2]`、positive literal。全対応はSTATIC_FINDINGSを用いる。

この1 recordは保存frontierに含まれる候補であり、43 literal全体の新policyには採用しない。B′／Referenceに同じliteral共通selectorと同じrecordを与える。

### 必要な実装・科学call

現DictionaryProvider、cache、旧certificateを変更せず、fixture専用のliteral→元record tupleの小さなoverlay adapterを用意する。元-13 spec／guard／T／W／sequenceを保持する。Sequence-only carrierの-11 keyはtransport上の約束としてのみ使い、backend recordの仕様を-11へ書き換えない。

必要な局所fixtureは固定4構成とする。以下の`a`は上のliteralであり、記載順が時間順である。

1. 1 qubit、native列`RZ(+a, q0)`をstandaloneでemitする。
2. 1 qubit、native列`RZ(-a, q0)`を同じpositive wordのcomplete actual adjointとして一度だけemitする。
3. 3 qubit／1 clbit、`reset(q0), reset(q1), reset(q2), SX(q0), CX(q0,q2), RZ(+a,q2), CX(q0,q2), SX(q0), measure(q0,c0)`。
4. 3と同じ列で`RZ(+a,q2)`だけを`RZ(-a,q2)`にする。

この人工native入力のglobal phaseは定義として0とする。これはSOURCE templateの欠けたglobal phaseを0と補う操作ではない。3／4ではreset prefixとmeasure terminalをunitary bodyから分け、同じnamed列を独立emitした参照とinplace consumer出力を比較する。

既存runtime／consumerのfixture処理を再利用し、named gate列・Wを含む行列・wire・reset/measure terminal・inplace join・一度だけのnegative inversionを照合する。元recordの保存strict guardは再計算せず引用する。Runtimeのglobal scalarを分離する際もN5-B1の同一joint構造・sink条件を満たすことを必要とし、相対位相をfitしない。

これらの回路fixture build／simulationを伴うcorrectnessは今回未実行で、別途承認対象である。Native compile・新規合成・strict guard・元word取得・T再集計は不要。SOURCE対応位置はmetadata根拠として保持し、whole m17をbuildしない。

### 判定とSTOP

- 4構成で適合を確認：この保存recordの局所interfaceを資格できる。新43-key policy／S1上界改善／RPE成功とはしない。
- 元spec、W、inverse、named gate／wire／terminalに不一致：adapter経路をSTOP。Spec偽装、phase無視、再合成で埋めない。
- Emitter精度が不足：有限runtime結果とstrict保証を分けてUNQUALIFIED。新guard計算へ進まない。
- 固定fixture完了後mandatory STOP。43-key適用、D word選択、価格最適化、全18round総Tへ進まない。

## 4. その後にGPT／ユーザーへ戻す判断

**A1/B1の承認は未受領で、実行待ちである。** 両方を実行するか、一方のみを実行するか、又は科学的判断のため保留するかをGPT／ユーザーが選ぶ。

A1はfull phaseの欠損を解消しないため、厳密etaとcommon-scalarで十分な信号保証を区別する判断が必要。Basis全体のactual adjoint化を優先するなら、旧compiler diskを自動継承しない独立のSOURCE／構成資格が必要となる。

B1後にrecord選択を行う場合は、SOURCE basis maxへの寄与とDの保存guard/T tradeoffを扱う有限policy／目的量を先に固定する。現word非劣のDを「無料で精度改善できる」と扱わず、両armへ同じ選択機会を与える。43 literalが3 basisに存在するだけではS1最大上界の改善を保証しない。

Domain（B′ scalar／Reference vacuum／event到達性）、central、emitter、shell、terminalは残る。全support policyが資格されるまで新しいjoint disk・保存626/630 shots条件・全18round総Tを計算しない。追加合成は既存recordと構造保証で不足する対象を特定してから判断する。局所保証改善のために総Tが悪化する案を無条件に採用しない。
