# 研究C：Compiled→rawの局所support包囲とcompiler abort scope

2026-11-05 JST。技術実装・人工検証のみ。新しい実H4のArb評価・profiling・sampling・回路build/compile、期待bias校正、shots/energy、Stage C、actual CA判定、重要科学レビューは実施していない。
正式共通指示原本 `research_project_instructions_unified_2026-10-25.md` §§8.1・8.4・10を適用した。

**結果：全列伝播を局所行列の包囲へ置き換える、有望な限定経路を実装した。実H4での構造対応・上界の鋭さ・速度は未検証。**
人工full512の共通prefix/suffixなしの例では約162倍のwall短縮と十分小さい上界を得た。一方、telescopingが相殺を失って粗くなる反例も保存した。速いだけの粗い上界を成功とは扱わない。

## 1. 起点と固定条件

独立branch：`codex/research-c-compiled-raw-certificate-acceleration-20261105`。
起点は指定されたCodex-C統合 `325f07ae02d8872d4fe322d6ad6e08c256e0bef7`。
[A/B/C統合報告](research_c_iid_calibration_integration_20261104.md)と、その薄い接続adapter・A/B原SOURCE・未採択政策をそのまま保持した。
7-call result `c0213176181b0f84eca0cf96464246d6a14e63a7` は別のresult branchとしてimmutable参照し、履歴をmergeしていない。
その[固定報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0213176181b0f84eca0cf96464246d6a14e63a7/docs/research/research_c_fixed_pair_batch_pilot_execution_20261105.md)・STOP・照合・資源metadata・C/R certificateの原版だけを今回のtechnicalディレクトリへbyte一致で保存した。

| 固定資料 | commit / 役割 |
|---|---|
| A/B/C統合 | `325f07ae02d8872d4fe322d6ad6e08c256e0bef7`。接続実装の起点 |
| 高速Arb準備A | `ce3185513578a8329efe364c748b5ce2aacdda45`。batch・失敗率adapterを再利用 |
| 7-call実行 | `c0213176181b0f84eca0cf96464246d6a14e63a7`。C/R 1173.086秒、inverse 1.734899秒を引用 |
| Stage P | `c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e`。指定1-wordの証拠 |
| Compiler全word調査 | `60dcaa94fd44675bc3184e409aa4ebdef7a07dbf`。全word成功は未資格 |
| Stage R | `1c0e1d1b4967efecb08dd3d8e7a747eba6c83039`。literal norm/rangeのみ |

対象はLinear H4・1.00 Å/STO-3G、number CDF8/LD2、partial S2/s1/r32/K0、h=δ=.1、controlled512、Arb192。
Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
既存P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、Gaussian/relative fusion、IID53-bit law、own-DF oracle・状態準備除外、CA/RPE政策は変更しない。

## 2. ボトルネックの切り分け

保存7-callではC/R上界 `1.585733058571e-11`、wall1173.086秒、共通prefix/suffix0だった。
Inverseは上界 `5.542118759907e-15`、共通11,386 gatesの除去で1.734899秒。後者の効果をC/Rへ移せない。
保存資源観測では32 worker、job全体RSS合計約1.5GiB。旧実績との差は、同時点のpaired benchmarkではない。

今回の保存record調査は**opcode・support arity・件数の読み取りだけ**で、角度文字列を数値へ変換せず、行列・回路・新しい誤差を評価していない。
正本は [stored_record_structure.json](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/stored_record_structure.json)。

| 保存record | 総gate数 | 内訳 |
|---|---:|---|
| compiled | 11,386 | RZ5738、SX2742、CX2894、X12 |
| raw | 2,633 | XX+YY1728、RZ640、CRZ48、P9、CRZZ112、CX96 |

rawは最大3qubit support、compiledは1/2qubit。したがってgateを名前・位置だけで一対一対応させる方法は適合しない。

既存 `c_arb_column_batch` は全d入力列・全出力を伝播するため、主要なentry処理は概ね `O(d²(G_C+G_R))`。d=512で各gateにつきpacking、FLINT積、scatter、midpoint、列norm/recenter、Fraction加算が必要になる。
Cacheが避けるのは局所gate行列の生成であり、これらの全列走査は残る。Batch幅を増やしても更新対象entry総数は減らない。
Process間では入力recordをinitializerで渡し、列結果を回収する。メモリが足りていても、Python/FLINT境界と丸め半径更新は多数残る。

**実H4の新profilingはしていない。** 人工4qubitのcProfileでは、旧経路の累積時間約0.094秒中、`midpoint_batch`約0.080秒、`_norm`約0.064秒だった。入れ子なので加算しない。
Gate行列生成だけが主因だとは言えず、recenterとPython/Fraction操作が大きい。この人工比率を1173秒の実H4へ定量移送しない。
新経路の同じ人工profileは約0.014秒。全profile・CPU/RSS・反復値は [artificial_benchmark.json](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/artificial_benchmark.json)。

## 3. 比較した方法

| 方法 | 保証・phase/full512 | 費用・保守性・実装判断 |
|---|---|---|
| 最終C/Rの全列Arb192 | 既存保証をそのまま維持 | 任意の適格pairに使えるがd²伝播が支配。既存方式を保全 |
| Gate cache・batch幅・allocation改善 | 同じ全列計算・phaseを維持可能 | 定数倍の候補。新しいArb native extension等なしではPython走査総数は残る。今回主実装にはしない |
| 共通unitary・内部の厳密可換構造 | norm不変性・位相の明示保持 | Prefix/suffix0でも、disjoint support順序や既知のinverseは利用可能 |
| **局所supportの誤差＋unitary telescope** | embeddingのoperator2 norm不変性で全512。Arb192で局所誤差とphaseを包囲 | 構造対応が成立する場合に採用。相殺を失い上界が粗くなる可能性あり |
| P0各passの誤差証明 | 全変換・phase更新を包囲できれば全wordへ候補 | 現在の保存pairには各pass traceがない。Rust synthesis・translation・declared phase等を厳密同値と仮定できず、今回は完成しない |
| Tensor/block/matrix-free | 全Fock分解の数学的根拠があれば候補 | 局所supportのtensor embeddingは採用。粒子sectorへの個別gate射影や有限state probeへ変更しない |
| State/signalだけの評価 | 別scopeのboundとして候補 | 現C/R operator guaranteeとは異なる。今回置換しない |

Installed Qiskit1.3 sourceを静的確認した [compiler_static_inventory.json](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/compiler_static_inventory.json) にpass構成と主要source hashを残した。
固定wrapperは既存のdefinition pre-unrollとMPFR160 phase accumulatorを通す。PresetにはInverseCancellation、UnitarySynthesis/HighLevelSynthesis/BasisTranslator、Optimize1qGatesDecompositionとfixed-point loopがある。
分析passのDepth/Size等、厳密gateのinverse cancellation、数値積・Euler再合成・phase加算を区別する。
「各passが無誤差」「1000反復以内に全入力で成功」とは認定していない。新方式は各passを証明したのではなく、**返された固定C/R recordを独立に比較**する。

## 4. 新しい局所評価器と導出

実装： [c_compiled_raw_local.py](../../src/trottertracks/c_compiled_raw_local.py)。旧SOURCEは変更しないopt-in経路。
既存の `fixed._validate` / `fixed._local`、Arb192、`z._norm` / `_real` / `_fraction`、process resource planを再利用した。
旧column APIは全列enumerationを証拠形式に含むため、局所解析による同じoperator保証をそこへ偽装せず、新しいproof kindにした。

### 4.1 Raw primitiveの局所bridge

XX+YY、CRZ、RZZ、CRZZの既存Qiskit定義は**候補**として1qubit/CX列へ展開する。U/U3はP–RY–P候補へ展開する。
その候補が正確であるとは仮定しない。元primitiveの既存独立Arb行列と候補の2/4/8次元全行列を比較し、definitionのbinary64角度・phase差と丸めを含む局所上界を得る。
候補の各gateは登録されたunitary primitiveであり、再帰cycle・非有限値・未知gateは明示的に拒否する。
Definition phaseはbinary64へ再加算せず、保存floatのexact dyadic和として全体へ持ち上げる。局所bridge誤差は重複primitiveごとのcacheと実occurrence数を区別して合成する。

### 4.2 厳密な構造対応

Disjoint supportのgateは可換である。Wire上の依存を保つDAGから決定的なCX順序を作り、対応CX間の1qubit列を取り出す。
Support上に介在gateがない厳密なinverse pairだけを消去する。共通部分回路がなくても、一方の複数RZが他方の融合RZへ変わるような対応を扱える。
構造対応がなければ `NOT_APPLICABLE_NO_SHARED_CX_SKELETON`、上界null。全列計算への自動fallbackはない。
Control-diagonal/target-XをCXの先へ送る追加の厳密変種も人工検証したが、因子の切り方によって上界を悪化させ得る。今回の限定契約は **commute_CX=false** に固定し、実H4結果を見て切り替えない。

### 4.3 SU(2)と位相の保持

各1qubit列を `U=e^(iα)V`、`V∈SU(2)` とする。αは登録gateのdeterminantから導くexactなπ有理倍＋dyadic角度和であり、数値的に自由なphaseをfitしない。
必要な局所の中心符号 `s=±1` は、VをsVへ置き換えると同時に**対応するπを全体phaseへ必ず加える**。これは数学的に同じoperatorの表示変更であり、phaseを捨てる操作ではない。
連続phase fitはせず、未補償global phase・controlled相対位相は誤差として残る。

SU(2)の2行列の差は2つの特異値が等しいので、

`||V−W||₂ = ||V−W||F / √2`。

局所積は各gate後にmidpointへrecenterし、外向きFrobenius半径を加算する。真の差のFrobenius包囲を得てから外向き√2で割るため、midpoint自体のdeterminantを1と仮定していない。

### 4.4 全registerへの合成

Unitary積のtelescopingと `||A⊗I||₂=||A||₂` を用いて、

`||C−R||₂ ≤ E_translation,C + E_translation,R + Σ e_local + |exp(iΔΦ)−1|`。

ΔΦには保存recordのglobal phase、candidate definition phase、全α、全中心符号のπを一度ずつ含める。共有CX因子を同じ順序で保持し、二重計上しない。
この導出は全2^n次元operatorに成立し、n=9で全512入力状態・全出力を覆う。有限probeの一般化でもsector限定でもない。
**列をenumerateしていないので `columns_evaluated=0` と明記**し、別の `coverage_proof` を保存する。
下界0は一般的な自明下界であり、未知上界を0で埋めることとは異なる。

Unitary同士の上界2は使用できるが、非unitary literal target DはこのAPIで拒否する。
将来C→Dへ接続する際も、R→Dは既存literal evaluatorの別legを保ち、外側triangleを2へclipしない。
旧C統合のcolumn専用checkerは新proof kindを意図的に拒否することをテストした。**旧per-word校正schemaへ接続済みとは扱わない**。

## 5. 人工benchmarkと限界

[Benchmark script](../../scripts/benchmark_research_c_compiled_raw_local.py) は分子の保存recordを読まない。全例で同一の人工pairを使い、isolated processで各3回、old/new順序を交互にした。
旧方式はbatch16・既存process plan、新方式は局所task量に応じた同じresource plan。小さい局所task集合はspawn費用を避けて逐次、独立taskが多ければprocess並列にする。恣意的CPU/RAM/worker/wall capは追加しない。
開始時CPU affinity32、available RAM約56GB、load約1.14。環境・共有venv/cache・他jobは変更していない。

| 人工例 | old中央値秒 | local中央値秒 | old/local | old上界 | local上界 | 解釈 |
|---|---:|---:|---:|---:|---:|---|
| full512・共通prefix/suffixなし・RZ融合 | 2.08023 | .01281 | 162.33 | 5.43896e-16 | 4.16334e-17 | 小さい有効上界を維持 |
| 3qubit raw primitive＋固定P0 wrapper | .04108 | .03400 | 1.21 | 8.64604e-15 | 9.32635e-15 | 少し粗いが小さい。普遍的高速化ではない |
| 可換誤差が全体で相殺する反例 | .01134 | .01043 | 1.09 | 2.26898e-57 | .249837 | **速くても校正に有効とは扱わない** |
| full512・共通gate列＋global phase差 | 1.57691 | .01322 | 119.31 | 2.05795e-11 | 9.09495e-13 | phaseを捨てず局所operator boundを使用 |

最初の例のCPU中央値（親＋子合計）はold56.17935秒、local.012814秒。
最大親RSSは約101–102MiB、oldの最大child RSSは約99MiB/個、新方式はこの例ではchildなし。これはprocessごとのhigh-water markであり、aggregate peak RAMの実測ではない。
構造対応・primitive展開・Arbを含むevaluator全体を計時し、warm-upは共通runtime importのみ。新方式からstructural plan費用を除外した比較にはしていない。

反例はcontrol RZがCXを挟んで相殺するケースで、defaultの局所和は相殺を失う。明示的な追加可換変種ではこの人工例の上界0を厳密に回復できたが、他の分割には不利となり得る。実H4での事後的な変種選択は未承認・未実施。
以上は人工の技術結果であり、**実H4・未知word・母集団のspeedupや鋭い上界の証拠ではない**。

## 6. Compiler失敗scopeの具体化

[c_compiler_abort_scope.py](../../src/trottertracks/c_compiler_abort_scope.py) はAの `failure_rate` / requests / confidenceと、Bの保存outcome、CのSOURCE/word/occurrence照合・status変換を再利用する。
新しいsampler、continuing scientific runner、fallback circuitは実装していない。以下は第一候補の技術設計であり、**正式採択はしていない**。

| 項目 | 固定すべき内容・今回の扱い |
|---|---|
| Raw failure | SOURCE/inputのcorrectnessを通過した後の登録raw build failure。全Nを分母 |
| Compile failure | Raw成功後のP0例外・guard failure・不適格native output。raw成功件数を条件付き分母 |
| Certificate failure | Compile成功後の数値不能・certificate FAILをcompiler failureへ混ぜない |
| Correctness FAIL | SOURCE/identity/math不整合は常に依存処理STOP。record-all政策でも継続しない。分類UNKNOWN/FAILではconfidence上界を出さない |
| 非停止・判定不能 | INCOMPLETE/UNKNOWN。失敗0や成功へ付け替えず、固定N confidenceはnull。任意timeoutを新設しない |
| 固定N | 全planned sample/occurrenceを保持。失敗標本も保存し、同じwordの別drawを除外しない |
| 初回失敗STOP | 未訪問occurrenceはUNKNOWN。完了した固定N実験へ昇格しない |
| 再抽出等 | discard・置換・並べ替え・重複sample ID・retry/resumeを拒否。同じwordの正当な別sample IDは保持 |
| p_fail confidence | 0件でも `p_U=1−δ_compile^(1/N)>0`。非零kは既存の保守的Hoeffdingで、exact CPと呼ばない |

既存 `process_sample` は初回失敗STOPであり、それをrecord-all runnerへ変更していない。
将来の継続runnerは、元wordをcallback前に保存し、raw/P0を各1回だけ試し、known algorithmic failureとcorrectness/unknownを分類する必要がある。
後二者はSTOPとpartial保全、採択された前者だけに固定N内の継続を許す。科学SOURCE不一致まで単なるBernoulli failureとして数えてはいけない。

### 要求数と確率の分離

旧独立kernel-occurrence政策では

`T_draw = Σ[m=0..12] 2^m (shots_X[m]+shots_Y[m])`。

今回の実shots・Tは未計算。Actual inverseは同じrealized recordから生成し、別word抽出0。Shellの追加compileは別会計。
Cache hitはcompile invocationを減らしてもdraw multiplicityを減らさない。同じSOURCE/candidate/law/word/angle/phase/P0/environmentに対する決定的outcome map、または正当化されたIID joint lawが必要。seed17だけで保証しない。
非決定的・履歴依存のcompile/cache、非停止の打切りを導入するなら別scopeを定義する。
固定の要求数上限Tに対するoperational abortはunion bound `min(1,T p_U)`、独立性もあれば `1−(1−p_U)^T`。観測後の小さいTを無断で代入しない。

δ_compile（推定confidence）、p_fail（母率）、operational abort、δ_cal（bias校正confidence）、RPE測定α=.01を別々に扱う。
共通校正eventを13roundへ再利用するときは同じeventを重複加算しない。新しい配分や総failure=.01を採択していない。
仮のunion合成にもRPE scope premiseが必要で、今回のadapterは無条件RPE failure上界をnullに保つ。

### Estimandと追加条件

成功wordだけのlawはq|Sである。失敗wordを記録から消して元qのbiasとして扱わない。
`E_q[z_D]=(1−p)E[z_D|S]+pE[z_D|F]` から、別途 `|z_D|≤Z_D` が資格付けされればselection shiftは `2 Z_D p` 以下。
例えば成功条件付きのbiasを使うなら、このtransportと、randomな成功標本数に対するconfidence条件が必要になる。固定N全試行の記録は残す。
分析上のzero-padding/killed signalは別の数学的estimandであり、fallback量子回路や元native無条件平均を自動的に定義しない。
Abort時には実運用のenergy出力自体がない。成功条件付きRPEとunconditional abortの会計を接続しなければactual accuracyを主張できない。
旧all-success calibration gateは変更しておらず、failure-aware estimator・law transport・continuing runnerは未実装/未資格。

## 7. 次の最小科学検証と契約

[未承認6-call契約](research_c_compiled_raw_local_pilot_unapproved_20261105.json) を用意した。
保存済み同じ固定32-event wordのC/R pairだけを対象に、**新方式が実H4で適用可能か・十分鋭く安いか**を調べる。新sampling/native build/compile、inverse再評価、R/D再計算、全law校正は0。

| logical unit | 最大 |
|---|---:|
| 保存C/R・旧上界・word記録load | 1 |
| SOURCE/record/phase/word identity | 1 |
| Primitive candidate展開・構造plan | 1 |
| 局所Arb包囲・telescoping | 1 |
| 旧包囲との整合・鋭さ・費用比較 | 1 |
| Summary | 1 |
| 合計 | **6** |

構造対応がなければsummaryを含む最大4 callsで上界null・mandatory STOPし、局所Arb・dense fallbackを実行しない。
対応する場合は、同じfull512数学的targetをArb192で包囲する。新上界が保存7-callのexact上界以下かを事前比較規則とし、valid-but-looserをcorrectness PASSと有用な高速化から分離する。
計時は新しいcandidate展開/planとcertificateの合計を保存旧C/R wallと比較し、plan費用を隠さない。新しい対照再実行はしない。
SOURCE/input/runtime/fresh不一致は科学前STOP、数学的矛盾・非有限値はpartial/journal/STOP保全。retry/resume0、rank/word/method変更・旧予算の流用なし。

Fresh：`artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01`（未作成）。
Entry：[prepare_research_c_compiled_raw_acceleration.py](../../scripts/prepare_research_c_compiled_raw_acceleration.py)。Defaultはpassiveのみ。**別途明示許可後だけ** `--execute-local-pilot` を使用する。今回は呼んでいない。
[Failure/abort設計契約](research_c_compiler_abort_scope_unapproved_20261105.json) は非実行。failure policy・N・δ・abort許容・cache/非停止政策・call上限はnull。

実H4で未確認なのは、expanded rawとcompiledのCX skeletonが対応するか、局所和が旧上界程度に鋭いか、unique primitive/block cacheとprocess overheadを含めて安いか、の3点。
これを確認する前にfresh IID-gridの大きいNを固定しない。校正pipelineへの新proof kind接続も後続の技術課題として残す。

## 8. 検証・保全・公開

新規人工テスト55 PASS。局所phaseの解析的包含、旧Arbとの整合、global/controlled phase、非可換順序、共通部分なし/あり、融合/cancel、局所誤差蓄積、parallel/serial一致、非unitary/非有限値、SOURCE/record/plan identity、全register誤昇格、旧column schemaへの偽装拒否、未承認実行拒否、人工runnerの正常6-call/未適用4-call/FAIL partial STOPを含む。
関連回帰290 PASS、旧消費済みH/P fresh namespaceのcold preflight2件だけを理由付き除外した。旧契約・SOURCE・テストを変更していない。
これらはlocal人工検証であり、実H4性能・外部再現・CIの主張ではない。

証拠：[新規テスト](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/new_tests.txt)、[回帰](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/regression_tests.txt)、[passive preflight](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/passive_preflight.json)、[SOURCE](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/SOURCE.json)、[保全](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/preservation.json)、[STOP](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/technical/STOP.json)。
既存179 SOURCEファイルを保全し、新規8ファイルを加えた187件を固定した。旧A/B/C/P0・Stage P/R・Head/B2のSOURCE/raw/契約/grant/ledger/marker/STOPを変更しない。
7-call原版の参照・保存結果と、今回のtechnical commitを区別する。旧科学SOURCEを今回commitへ付け替えない。
公開後remote SHA/tree・報告本文・主要blobを照合する。

次の担当はユーザー／研究C：未承認6-call pilotを実行するか判断する。その許可がある場合だけCodexが固定範囲を一度実行する。
Compiler failure scope・confidence/abort配分の採択、fresh校正、重要科学レビューは別の判断・許可である。
**Mandatory STOP。新しい実H4科学計算へは進んでいない。**
