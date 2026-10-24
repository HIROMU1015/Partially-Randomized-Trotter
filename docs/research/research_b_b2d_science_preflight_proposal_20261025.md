# 研究B：B2-D限定科学検証の実行前提案

2026-10-25 JST。**条件案・接続準備であり、科学実行承認・grant・科学的GOではない。** 今回は5方式のinterface・未起動の数値backend・人工testを追加し、実Hamiltonian生成、operator／signal評価、回路build／compile、数値再解析を実施していない。

## 1. 起点・既存資産・現在の到達点

正式起点は `5d3ec883eeeabf69abfa128c9e169c59fc572149`。その[B2-D adapter][ADAPTER]・[未承認の旧草案][DRAFT]を保全し、[限定設計][DESIGN]のA0–A4を共通評価へ接続する。旧草案の未採択欄を無言で書き換えず、本書を新版の**提案**として追加する。

PFの固定出所は `HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae`、`20d64c2fea1ed2d2188777df4f3d962ee94df486`。published YP8m8の6関数は旧adapterを変更せず再利用し、A1–A3に必要な既存3係数関数だけをbody変更なしで移植した。新しい係数探索・corrector設計・分割最適化は行わない。

今回の追加SOURCEと実行した人工testの入口は [`b2d_evaluator/README.md`](../../src/trottertracks/b2d_evaluator/README.md)。共通運用指示2026-10-23版は実参照済みで、SHA-256 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` は既読版と一致した。研究概要・既存科学SOURCEが前段から不変であることも確認し、不要な再レビューをしていない。

研究CのC1圧縮DF最優先は維持する。通常のB2の次の判断は**研究Bチャットとユーザー**へ返す。優先順位変更・研究間の重大調整・中心方針の採否が必要な場合だけ研究統括へ戻す。

## 2. 共通評価器の範囲と、実行前に残る接続

| 層 | 実装・資格 |
|---|---|
| 純粋な列定義 | A0は現forward／reverse D sweep、A1–A3は同じblock順のstandalone D半時間発展、A4は旧 `DRun` の `P K^n_D P†`。signed time・tail barrier・数学的積／append順を明示 |
| 共通評価DAG | 登録gridから参照・mean・error・signal処理の依存関係と共有keyを作る。ここはmetadata処理だけで科学計算をしない |
| 未起動のdense backend | 承認済みのsector行列・state・symbolic tailを受け取り、既存finite-RTE moment・signal関数へ接続する実装を用意。numpy／scipy／trotterlibのimportはbackend生成時に限定。今回生成・実行していない |
| Input package | fixture生成・sector変換・block行列materializeの経路は未接続。新SOURCEの `PreparedDenseInputs` へ同じsector順序の行列・stateを渡す準備が科学実行前に必要。旧helperを直接呼ぶと回路もbuildするため、その呼出しは採用しない |
| 実行guard | 未設定条件、異なるSOURCE／grid／call範囲、承認・新grant参照の欠落をbackend生成前に拒否。入力配列hash・H=D+R+scalar・非可換性等の数値確認は将来のbackend内gate |
| 承認管理 | guardは呼出し元の承認情報との一致検査であり、外部ledgerの未使用grant・実ユーザー承認・fresh outputを証明しない。新grant発行／消費やscience launcherは今回追加していない |

これにより5方式の最小評価経路は用意できたが、**実データ接続と数値的正しさは未検証**である。独立参照との照合を実行しないと解消できない事項をmock PASSで解決済みとしない。

## 3. Fixture・分割・sector・stateの提案

候補は起点commitの [`tests/test_df_partial_s2_repeated.py`][FIXTURE]、blob `0c92ae9c1e8c5721d9c8629a83435b855f5d6043` の `_nontrivial_basis_case`。固定synthetic one-body＋3 DF factor、2 system modes、`L_D=2` の定義を使用する案である。分子geometry・AO basis・分子DF rank探索は非該当。

| 項目 | 静的に分かること／推奨案 | 未確認・必要な承認 |
|---|---|---|
| Hamiltonian identity | 元関数のconstant、one-body、3因子・係数を変更しない。生成せず定義を参照した | 実際のHamiltonian hash、sector行列・準備・tail hashは未生成。承認後にbinding |
| Split／順序 | 元helper同様のweight-ranked prefix `lambda_frobenius_squared`、`L_D=2` を推奨。one-bodyを含むD blockを固定順で共有 | selected indicesは今回rank計算していない。generation-prefixと同一視せず、解決した順序を記録 |
| D内部非可換性 | 非対角係数と異なるbasisを持つ定義 | 既存testはbasisと既存回路を照合するが、必要sectorのcommutator非零を保証しない。数値確認は未実行 |
| D/R非可換性 | 残差を含むfixtureである | `[H_D,H_R]` のoperator norm確認は未実行。atolを越えない場合は未判定STOPし、別fixtureへ自動移行しない |
| Sector | **N=1の全粒子数sector**を第一候補とする。2 modeのN=0/2の自明sectorを避け、block／tail／stateを同じ順序に限定するため | 今回新たな提案であり未採択。sector保存・leakage、bit順とblock表現の同値性はInput package接続で確認が必要 |
| Input state | **mode0を占有した固定occupation状態**を全arm共通の第一候補とする。exact ground stateを作らず同じphysical状態を使う | OpenFermion／Qiskitのbit順による座標をloaderで明示。signalの識別力は未検証。state変更を結果後に自動追加しない |
| Exact参照 | 同じsectorのH、D、retained Rのmatrix exponentialを診断参照に使う。無料の量子oracleではない | exact行列とmatrix exponentialも科学call。全arm共通で共有し、今回は実行しない |

現backendは用意されたinput packageのshape・Hermiticity・norm・identityを確認するが、ラベルだけでparticle sectorの正しさやbasis変換の同値性を認証しない。この不足を埋める接続SOURCEと参照検査が実行前に必要である。

## 4. 時間・反復・tail条件の具体案

| 条件 | 推奨案・出所・理由 | 残る不確実性／資格 |
|---|---|---|
| Macro幅δ | **`{+0.09,-0.09}`**。元fixture testが使うsigned幅を候補にする | ±は二つの絶対時間幅ではない。正時間で機構診断、負時間でsigned処理を照合。8次の誤差がnoiseより大きいかは不明。次数fitを目的にしない |
| Macro反復q／T | **`q∈{2,3}`, `T=qδ`**。元fixture testの反復を使い、短い連続stepの境界を識別 | 長RPEの代用ではない。新B2の条件としては未採択 |
| D内部n_D | **A1–A4共通で `{1,2}` の固定比較**。1は既存PF full列APIの単回、2は最小の複数内部反復を表す構造診断案 | n_D=2は新提案であり性能最適値の根拠はない。同じ機会を4armに与え、optimizer・事後選択は導入しない。A0にn_Dを付けて重複計上しない |
| Tail r_RTE／K | **r_RTE=2、K=2**。元helperと同じ候補 | 新B2で必要精度を満たすか未検証。Kの自動増加やlaw変更はしない |
| RTE tolerance metadata | 元helperの **`truncation_tolerance=1.0`** を候補とする。Kを明示固定するAPIのmetadataとして使う | physical operatorの許容誤差や科学的改善閾値ではない |
| Scalar／identity | 元helperの **`extract_identity_phase`**。Hamiltonian constantと抽出identityをphysical時間に対する位相として保持 | scalarをPF stage数だけ掛けない。event phaseと混同しない |
| Threshold | **coefficient_atol=0**。既存preparationのdefaultに合わせ、retained tailとphysical Hの差を新たに導入しない | first-stage実装は非零thresholdを拒否。別policyは追加SOURCE／承認が必要 |
| Sampling／seed | **analytic finite-RTE operator moments、trajectory samplingなし**。seed41は元helper由来のmetadata候補 | q回は独立occurrenceのfirst momentとして扱う。seed41から独立性を証明しない。有限bit samplerのbias評価は含めない |
| Finite-bit policy | **analytic_moments_no_sampling**。理想finite targetの診断に限定 | finite-bit certificate、weighted measurement、shot実行は第二段階 |
| Precision | **complex128**を最初の接続候補とする。既存小規模DF／RTE参照と整合 | 新構成でのroundoff余裕・高精度一致は未確認。自動precision昇格・追加計算をしない |

上記は研究Bチャットで採択するための具体案であり、実行契約の確定ではない。識別力不足なら**INCONCLUSIVE**として返し、時間点・fixture・familyを結果後に増やす標準手順にしない。

## 5. 評価grid・公平性・出力schema

| Arm | Gridと作用列 |
|---|---|
| A0 現partial-S2 | δ2種×q2種＝4行。`R_D(δ/2) tail F_D(δ/2)` を保持 |
| A1 standalone D-S2 | δ2種×q2種×n_D2種＝8行。同じ方向のD半時間発展でtailを挟む |
| A2 Yoshida4 | 同じ8行。既存係数を再最適化しない |
| A3 published Y8m10b | 同じ8行。legacy m8への置換なし |
| A4 published YP8m8 | 同じ8行。旧adapterの `P K^n_D P†` を変更せず使用 |

**合計36行**。これは静的な登録点の個数で、保存科学結果の再集計ではない。各rowは同じresolved input fingerprint、tail identity、state、scalar／threshold、Tとδを参照する。固定n_Dごとの記述的な比較とし、各armのbest点を異なるbudgetで最適化した結果とは呼ばない。

出力schema `b2d_five_arm_evaluation_v1` は次を分離する。

- source commit、設定・input fingerprint、D block順、tail／state／sector identity、graph fingerprint、承認／grant／output参照、登録call数。
- correctness欄：inputのH identity、D内・D/R非可換性、A4の独立数学的P/K積・actual inverseの参照残差。**旧validatorとの数値同値性を認定する欄ではない。**
- row欄：arm／family／係数出所、δ／q／T／n_D、D半時間operator誤差、tailを外したD全時間誤差、exact-D/R outer誤差、ideal-tailとphysical Uの差、corrected meanとphysical Uの差、Taylor bias、normalization整合、raw／corrected normalization。
- signal欄：同じstateのphysical／ideal-tail／corrected／rawのX・Y・半径、条件を満たす場合だけprincipal phaseと参照phase距離。
- 判定欄：現実装は **記述診断のみ・科学的GOなし**。native／shotsはscope外、終了はmandatory STOP。

A0の片側sweepはstandalone D半時間発展ではないため、A0のD-half errorは未定義としてnullにする。全armについてtailを外したD全時間の近似も評価し、異なる時間のD誤差をそのまま順位にしない。signal半径・位相をkernel固有値誤差と混同しない。corrected meanはunitaryやchannelではなく、normalization補正とprocessor補正も別である。

## 6. 科学callの登録案と共有

以下は**未起動DAGのdispatch単位**を上記gridと3個のD block labelから数えた値である。実行済み科学callでも、native gate数でも、全ての内部NumPy／SciPy呼出し回数でもない。Input package作成はこのDAGの外側であり、別に登録する必要がある。

| Dispatch種別 | 提案gridでの個数 | 内容・共有 |
|---|---:|---|
| validate_input | 1 | 同一入力のshape・配列hash・state norm・H identity・非可換性。内部の交換子norm処理を含む |
| exponential | 410 | block/timeとexact D／R／physical Hのkeyを共有。uniqueなreference exponent node |
| rte_moments | 2 | signed δごとの既存finite RTE config／corrected／raw mean。全arm・q・n_Dで共有 |
| moment_field | 4 | 2δ×corrected／rawの取得。新しいmatrix exponentialではない |
| a4_reference_check | 4 | 2δ×2n_D。adapter展開とは別に数学的S2／P／K積を構成する数値gate。内部の追加exponential・積はbackend cacheを共有 |
| product_append | 94 | 同じordered依存列を共有。tailをまたぐprocessor削除はしない |
| physical_scalar_phase | 56 | step／参照のscalar位相を一貫して付与 |
| power | 112 | 共通stepのq反復等。独立occurrence meanの積という条件を前提とする |
| normalization／rescale_raw | 4／36 | 共通normalizationと、各rowのraw mean再補正整合 |
| operator_error | 182 | D／outer／physical／Taylor／normalizationの差を区別。共有する差は一度だけ |
| signals | 36 | 各rowの4対象の共通state信号と半径・位相診断 |
| **計** | **941** | 承認後にこのDAGへdispatchするなら941単位。mockでは同数のラベル処理だけを行った |

CorrectnessのA4数学的積参照はfull arm評価前に置く。旧PF validatorのfull listを直接appendする経路との数値同値性は前段から未確認であり、このgateが通っても自動認定しない。旧validator照合を追加するなら、その対象・追加callを別に登録する。

**まだ積算できない範囲：** fixture生成、split／basis／sector解決、D block／retained tailの行列化、粒子数・basis順の独立照合、source／array identity binding、必要と判断する独立precision参照。これらのInput package接続が未完成のため、**第一段階全体の科学call総数は未確定**。941を全job予算やexactなexpm総回数と誤記しない。A4参照内の追加cache missも実接続時の低層call登録へ含める。新grantはそのscopeを確定するまで発行しない。

## 7. 数値精度、科学的判定、未判定条件

| 条件 | 提案／未確定 |
|---|---|
| Correctness数値atol | **1e-12を候補**。元fixtureの既存operator照合に使われる値からの出発案。旧assertのrtol込みPASSを新経路の根拠にはしない。本backendは絶対operator normで確認するため、適用性は未検証 |
| Signal誤差の報告 | 同じstateの複素差を直接保存。operator normと別に記録し、位相誤差だけを選んで改善としない |
| 半径の有効下限 | **未確定**。旧RPEの半径条件・予算を無根拠に移送しない。`radius_floor=None` のままでは実行guardが拒否する |
| Principal branch余裕 | **未確定**。`branch_margin=None` のままでは実行guardが拒否する。現在のbackendは点ごとのprincipal branch診断のみで、δ窓を追跡したunwrap／energy推定をしない |
| Noise判定 | correctness残差と比較する誤差差が同程度ならINCONCLUSIVEとする案。complex128の数値残差を厳密な誤差certificateとせず、識別閾値・必要な独立precision checkは採択が必要 |
| 科学的materiality／GO閾値 | **未確定**。まず記述診断だけを採択する案を推奨。実装の `decision_policy` はその案だけを受け、科学的GOを自動生成しない |

将来の解釈案は以下のとおり。これは数値閾値の正式採択ではない。

1. **継続候補**：D改善がfull operator／同じphysical signalにも残り、強い既知baselineとの差がnoise・branchの曖昧さでは説明されず、端点費の第二段階を調べる情報価値がある。科学的GOや総資源優位とは別。
2. **限定NO-GO候補**：D改善がfullに残らない、または同じscopeのYoshida4／Y8m10b等に追加価値がない。この固定入力・時間・family内の判断で、processed PF一般の不能とはしない。
3. **INCONCLUSIVE**：非可換性・identity gate不成立、8次誤差がnoise-floor、半径・branch条件不成立、precision照合不足。時間窓やstateを同じ許可で拡げてpositive例を探さない。

## 8. 実行承認、grant、結果保存、STOP案

| 項目 | 研究Bチャットへ返す案 |
|---|---|
| 条件採択 | fixture／N=1／固定occupation state、±.09／q2,3／n_D1,2／r2／K2、threshold0等をユーザーが採択するか判断 |
| 残るSOURCE | Input package接続と独立basis／sector identity検査を完成し、新しい科学SOURCE commitを固定。現branch名だけで承認しない |
| 未確定数値条件 | radius／branch、数値noiseと必要precision、記述診断か科学的閾値付き判定かを実行前に決める |
| Grant | **新規・未使用の限定grantを別承認後だけ発行する案**。SOURCE・resolved inputs・登録grid／call・出力scopeに結び付け、既存grantを流用しない。今回は発行・消費なし |
| Retry／resume | **一回の限定実行・自動retry／resumeなしを提案**。技術的不具合時も原記録を保存してSTOPし、SOURCE修正・新attemptは別判断。未採択 |
| Output案 | freshな `artifacts/research_b_b2d_stage1/<execution-date>/<fresh-run-id>/`。設定、resolved identity、登録callと実績、全row、gate、不一致／未判定、環境・SOURCE provenance、STOPを保存。今回directoryやscience artifactを生成しない |
| Scope外 | Native circuit build／compile、full controlled wrapper、finite-bit certificate、weighted moments／shots、全RPE資源、C1共同計算。第一段階の名目で追加しない |
| STOP | 最初のcorrectness不一致・未採択条件発見・call逸脱でSTOP。登録比較が完了しても公開照合後mandatory STOP。追加grid／family／係数探索は自動開始しない |

実行guardは承認参照、SOURCE、graph fingerprint、call種別／個数を照合するが、外部grantの真正性・未使用性を判定する仕組みではない。未来のlauncher側で既存運用と接続する。新しい汎用grant管理を今回作らず、実行条件を変更した場合はfingerprintと承認範囲を結び直す。

新計算契約ではphase／total wall-time cap、恣意的CPU／GPU／memory／worker capを既定にせず、結果前にcall／output／log／instruction scopeを固定する。計算前に実資源と支配負荷を確認し、cache・既存sector／matrix-free等を適合する範囲で再利用する。小規模DAGをserialに扱うのは参照共有と依存関係のためで、科学runの速度倍率や資源上限を今回決めたものではない。既存固定契約・共有環境・他ユーザーを保全する。

## 9. 完了状態と次の担当

**今回完了：** 5arm列、共通評価DAG／設定／出力schema／実行guard、未起動dense backend、人工test、条件・call案。旧adapter15件＋新規14件＝**29件PASS**。数値backend・fixtureは起動していない。

**次は研究Bチャット／ユーザー**が、条件候補と未確定欄、Input package接続、科学SOURCE・grant・全call範囲を確認し、必要な実装と限定科学実行を別途明示承認する。研究Aはnormalization・半径・finite-bit・range／shots、研究Cは将来DF表現・native合成・controlled位相を分担する。通常のB2進行を統括へ報告することは必須としない。

旧草案・SOURCE・raw・grant・ledger・marker・科学契約・旧STOPとC1最優先を保持する。今回追加した準備物のremote SHA・公開内容一致を確認後、**mandatory STOP**。YP8m8の優位、独立新規性、科学的GOを認定しない。

[ADAPTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/5d3ec883eeeabf69abfa128c9e169c59fc572149/src/trottertracks/b2d_adapter
[DRAFT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5d3ec883eeeabf69abfa128c9e169c59fc572149/docs/research/research_b_b2d_science_validation_contract_draft_20261024.md
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1e2feb033bd7857482db09dd1f3d5b8692f100bf/docs/research/research_b_b2_limited_design_20261024.md
[FIXTURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5d3ec883eeeabf69abfa128c9e169c59fc572149/tests/test_df_partial_s2_repeated.py
