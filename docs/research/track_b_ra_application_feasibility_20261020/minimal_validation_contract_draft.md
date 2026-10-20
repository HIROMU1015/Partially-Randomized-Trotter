# C1：保存H4 DF coherent signalの最小検証契約案

**DRAFT_BLOCKED_PENDING_INPUT_GUARANTEES_AND_APPROVAL。将来の計算契約草案であり、実行許可ではない。** 2026-10-20 JST。本書は候補1件のみを記述する。新規sampler、案A/B、全round RPE、Full-N、追加Pilotへ進まない。

## 1. 問いと最小範囲

仮説候補は「実DF basis付きinvolutionの短い時間発展で、return集約の量子単価・weight/十分N・古典取得費用の交換が、強い既知対照に対して実用的な選択を変える領域を持つか」である。RAの勝利を前提にしない。新samplerの新規性や一般分子の優位を問う実験ではない。

提案する最小固定点は、既存development H4の**一つの時刻・一つのsplit・一つの外側PF schedule・m=5**。同じ点でordinary/partial/closed P3/Full/closed P5を比較する。これは機構分離のcomparisonで、q/r/mを最適化した最終best-to-best比較ではない。後者なしに実用最良・PR全体優越を主張しない。[調査報告](README.md#4-強い先行法との比較)のbaseline不足を結果の後で省かない。

### 固定案と未確定事項

| 項目 | 草案の具体案／承認前に必要な取得物 |
|---|---|
| Hamiltonian | PR-2の保存development linear H4 1.00 Å、STO-3G、DF rank12、8 system qubits。snapshot SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、H hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。今回は本体未開封。再生成で代用しない |
| 表現・split | one-body/constantは既存規約、生成順DF prefix L_D=3を決定論側、残りをbasis付きsigned I/Z/ZZ involution tail。identity policy・threshold・exact component ordered list・basis operations/hash・partition hashは既存入力から別途固定する必要がある |
| state | 上記snapshotと結合した保存数値sector-ground state。厳密固有stateとは呼ばない。state bytes/hash、bit/occupation順、normの保証、準備回路または明示されたcost/errorモデルは**未取得**。無料のground-state oracleとはしない |
| 真の推定対象 | `z_H(T)=<ψ|exp(-i T H_DF)|ψ>`のRe/Im。DFモデル固定のcharacteristic functionであり、電子系の正確なenergyやground-state phaseを推定したとはしない |
| 時間・外側 | T=0.8 Ha^-1、既存二次対称PF、q_outer=1、δ=T/q_outerを固定する案。決定論generatorの順・各tail occurrenceの係数/時刻・相対位相をfreeze。q=1で保証不足なら停止し、無断でqを増やさない |
| tail | 各random occurrenceをr_RTE=4 microstepsに分割、各step `P_5(-i σ x R)`、`x=λ_R |δ_tail|/r_RTE`。全occurrenceで0≤x≤1を認証。r_RTEとq_outer、Taylor m、RPE roundのmを混同しない |
| 精度候補 | complex ε_total=0.05は既存PR-2契約からの設計候補。軸ε=ε_total/√2。採否とdeterministic bias配分、finite-bit/angle/native precisionは**未採択**。既存H160/K256やnative1e-6を一般分子へ自動流用しない |
| failure候補 | 一つの比較packet全体でα_total=0.05を提案。全採択armのX/Y、resource tail、入力/数値certificateに必要なfailureを事前配分する。arm数確定前にα_axisを固定したとしない。PR-2の各軸.025を全arm同時保証へ転用しない |
| 出力 | 精度certificate、全trial/accepted会計、T/CX/1Q/depth/workspaceのresource vector、setup/online/保存費用、未知項、比較範囲と不成立原因。今回これらを計算しない |

T、q/r、精度は[既存PR-2契約][PRCONTRACT]内の値・候補から選んだ草案であり、今回新しい性能閾値を正当化したものではない。特に旧K=2/4とG6の奇数mは同じ記号ではない。m=5のpaired finite targetに相当するordinary cutoffのAPI対応を結果前に確認する。旧signal・旧shot・旧compile表を新lawの結果として転用しない。

## 2. 比較対象と公平性

同一finite targetを比べる機構層では、ordinary、partial-return+tail、closed P3+tail、既存general Full、closed P5を必須とする。m=5でclosed P5を除いたFullの勝利は成功扱いにしない。canonicalでも実digital law/weightとroot誤差の保証を要し、理想B一致だけでNを同一としない。[G7][P5][W2B]

同一exponential taskのapplication層では、同じH/state/T/ε/α/accessで、既存DF ordinary PR、deterministic symmetric PF、適切なPauli収集CTS/SCUを強い対照候補に含める。Pauli展開・coherent embedding・厳密精度証拠の取得が未完なら、その欄はUNKNOWNとしてapplication優位の判定を保留する。SCUのchannel shot式を一枝coherent signalにそのまま使わない。qubitizationは同じPREPARE/SELECT等を取得できる場合の別resource境界として扱い、未取得なら「全Hamiltonian simulation法への優位」を主張しない。

既存D1の参照event全列挙・全key取得は分子用の実行計画として未承認である。全supportのnative平均/上界、rare event、event-dependent angleのcoverageを、どの既存保証・取得経路で閉じるかを実行前に固定する。限定標本のCPU/native平均や有限cacheの成功だけでは全task費用・biasを認証しない。別の非自明な費用保証やアルゴリズムが必要なら、今回の草案で補わず停止する。

全側に同じ合理的なcache、grouping、prefix/alias等のselector、completion、同一Q productの相殺、basis reuse、direct rotation実装と精度選択機会を認める。ordinaryに全語列挙を課さない。合法な既知最適化を適用できない場合は理由と影響を保存する。既存実装のliteral費用比較と、最適化後のbest-to-best費用を分ける。

最小固定点ではq/r/m探索をしない。この制限が強い対照の既存最良設定を外す場合、その既存設定は同taskへの接続条件を調べる参照として残し、探索不足を明記する。新しい探索や対照の追加計算は別scope承認なしに始めない。

## 3. 誤差・confidenceの契約

### 共通targetへ戻すledger

| 項目 | 認証に必要な条件／禁止する代用 |
|---|---|
| 入力 | exactに採用するH/state、DF truncation/threshold、係数rationalization、basis近似の差を明示。固定DFモデルだけをtargetにするなら物理電子Hamiltonianとの差は対象外と明示し、化学精度claimへ使わない |
| outer PF | 同じH_DFに対するsplit/order/timeの誤差上界。state-specificの経験的fitや保存数値biasはrigorous upper boundではない。strict認証へ閉じなければ`ENGINEERING_ONLY`とし、認証Nの結論は出さない |
| Taylor | 各stepはfinite polynomialで、expそのものではない。既存D1/D3の適用条件を確認し、非unitary finite meanの積で誤差を伝播する。`r × local error`だけで済ませない |
| finite bit / roots | 正有理p・positive support、全π>0、実proposal π、実係数c_tilde、`πW=c_tilde`をsource-boundで保証。proposal補正はroot/係数の近似biasを消さない。未知B_full/Zを無料で予算に使わない |
| angle | F3のatan/half-angle enclosureとnativeへ渡した代表角との差、σ・child・wordとの結合を保存。angle certificateをnative certificateと呼ばない |
| native/provider | controlled-Q、実際のadjoint、deterministic部、basis、controlled rotation、ancilla相対位相のstrict誤差。近似Qをexact involutionと仮定しない。合成primitive誤差はweighted meanと全pathへ伝播 |
| state/readout | 共通ψからの準備誤差、X/Y測定器・較正/論理ノイズのscope。条件付き理想測定モデルならその仮定を全側で揃える。未取得物理ノイズを0と認証しない |
| 統計 | fresh IID bit/trajectoryと測定モデル、全trial estimator、bias控除後の正の幅、moment/range、整数ceil、familywise配分を固定。SHA-counter prefix一致はIIDの実証ではない |
| PR/RPE固有 | C1は単一signalのみでphase/energy decoderを実行しない。phase/visibilityを報告するなら別に半径下界と位相branchの条件が必要。RPE全round成功・alias回避・energy精度は対象外で未保証 |

既存の[G6数学監査][MATH]・[有限bit監査][BITS]、[方法ノートv1.1 §§3–4,9–10][NOTE]、[D1会計][DACC]を参照する。これらの一般理論と固定実装の適用域を区別する。D1 SOURCEの入力固定チェックを解除して呼ぶのではなく、新しい契約でどの既存保証が使えるかを先に監査する。非自明な新しい合成保証・定理が必要なら`UNRESOLVED_PROOF_REQUIRED`として停止し、別の証明承認を求める。今回それを証明しない。

### estimatorと十分N

同じ軸・全path一trialについて、正当zeroは0、acceptedは**実proposalに対応するpath weight × X/Y測定値**とする。denominatorは全古典trial数。acceptedだけの平均へ再正規化しない。全random drawsを量子実行前に確定できる場合に限り、どこかでzeroならstate preparationを遅延して省略できる。技術failureはzeroではない。

既存G7/D1型の十分予算を採用するなら、全pathに有効な二次moment上界V、絶対range R_max、deterministic bias上界b、残余s=ε_axis−b>0とα_axisから、既存Bernstein型の

`N = ceil(log(2/α_axis) [2 V/s² + 4 R_max/(3s)])`

を評価する契約を結果前に固定する。log/root/ceilも外向きに扱う。これは十分Nであり必要最小Nではない。理想single-step Fullの`B_ord B_full`、canonicalの`B_full²`、実digital certificateの`κ B_ord U`を混同しない。独立path積のmoment/rangeとnonunitary biasを戻し、真の未知normalizerや参照全event表からNをoracle計算しない。[G7][DACC][W2B]

比較側にも同じconfidence設計を与える。採択前にBernstein/Hoeffding等の保証policyの差と利用可能な情報を固定し、同じ十分条件での比較と各法に最適な既知保証を許した比較を区別する。既存PR-2の保存signalを使ったNを、未評価G6への認証Nとして貼り付けない。

## 4. 全資源の契約

CPU秒、bit演算、bytes、T/CX/1Q、depth/workspaceを別座標で報告する。換算モデルを事前採択しない限り一つのscoreへ足さない。

| 層 | 必須の支払・保存項目 |
|---|---|
| 入力/setup | 既存snapshot/state/registry取得、正規化・identity移動、exact入力証拠、Green kernel、U、P5群/completion、実law、preflight、cache key/内容と償却回数 |
| online classical | raw reject/coin reject/accepted全経路、selector・係数/root/angle・実weight、F3再query/hash/serialization、bit長、監査・IPC/I/O・finalization。拒否は無料ではない |
| 全trial | axisごとのN、全draw数、実完了数、terminal prefix。accepted quantum callsと別列。未完走runへの事後conditioningで全N confidenceを維持しない |
| quantum | accepted path全体のcontrolled deterministic部、Q/rotation、basis/inverse、helper、相対位相、state preparation、X/Y/readout。全q_outer・全tail occurrence・全r_RTEを含める |
| native | strict primitive単価、key取得費用、event依存価格、合法boundary/basis cancellation、T/T†・CX・1Q・depth・workspace。Tを含む1QとTを重複加算しない |
| stochastic cost | expectation、accepted数tail、worst/hard上限を別にする。期待受理数をhard capとしない。未取得のμ、native単価、prep/readout費用はUNKNOWN |
| storage/Evidence | Result/Event/πW/angle/provider certificate、全trial分類、full/compactのschemaと再構成限界、capacity/失敗時prefix、保存・再監査費用。全child再構築が監査側に残るなら計上 |
| 償却 | 何回/何target/何axisでsetupと合成cacheを共有できるかを事前固定。RAと対照へ同じ再利用機会を与える。入力・精度・source変更後の不正cache reuseは拒否 |

既存[W2c][W2C]の会計に従い、古典は`setup + N E[c_classical/trial] + 保存/終端費用`、量子は`N E[1_accept (C_path + P + readout)]`を、各axis・segment・roundの対象に応じて積算する。X/Yを一つのNと数えない。量子前判定ができないstreaming方式なら拒否prefixで既に払った量子費用も必要で、この式の単純な適用を保留する。C1のroundは1時刻のみ。将来RPEでは全roundのN、時間、準備、失敗配分の和が必要で、最大roundだけの価格で代用しない。

古典費用を将来測定すること自体も別実行承認を要する。16,896 PilotのCPUや保存bytesを本H4/全Nへ比例外挿しない。最小検証は新しいPilotや全Nの実行を自動的に含まない。

## 5. 成功・停止条件

**技術成立の条件**：同じH/state/T、全側のexact/digital law、targetへのbias、正の残余幅、confidence、provider/価格scopeが閉じること。未取得が残ればresource comparisonは部分結果に留める。

**研究上の情報価値を持ち得る結果**は、(a) 強い低次数対照を含めてもnative資源または準備・償却条件の明確なtrade-offが残り、その原因をreturnと実basis/角度価格へ帰属できる、または(b) 同じ条件を閉じると利益が消え、どの費用・保証が消したかを証拠付きで限定できる場合である。既存D系列の単なる繰返しや小さいpoint差だけなら、追加研究の価値は未確定とする。

materiality、費用推定の必要な精度、許す価格/準備/償却領域、主要resource座標、application層で必要なbaseline被覆は**GPT・ユーザーが実行前に採択する未確定事項**。旧PR-2の10%や別Pilotの件数を自動流用しない。勝つseed/precision/価格を結果後に探さない。

次の場合は新計算への準備または許可された計算を停止し、その理由を保存する。

1. snapshot/state/component/basis同一性、再取得経路、positive rational入力、必要providerが揃わない。
2. q/rを固定するとbias予算が尽きる、または認証できない。無断の次数/step/precision/target変更で救済しない。
3. canonical/CTS/通常PFの強い対照や合法cacheを省かないと結論が成立しない。
4. 実π/weight/ordered phaseと測定を結べない、未被覆のnative角度・rare eventが残る、技術障害をzeroへ置換する必要が生じる。
5. 新非自明保証、案A/B、新selector、新相関law、別分子/held-outを必要とする。
6. 同じtoy結果の反復以上の問いが残らない、またはユーザーが投資価値を認めない。

停止は不利な科学結果と技術的未確定を区別する。固定class内の負結果は保持し、全RA/全PRのno-goへ一般化しない。

## 6. 実行前に別途必要な承認・取得物

| 担当 | 次の仕事と必要な承認 |
|---|---|
| ユーザー | C1を検討するか、既存記録で区切るかを判断。必要ならGPT詳細レビューの開始を明示承認 |
| GPT | 上記仮説が既存PR-2/P5/D系列を超える問いか、最小固定点・strong baseline・成功/停止条件・誤差/failure配分・主要費用を科学的に評価。採択はユーザー |
| Codex | **別の技術準備指示後のみ**、既存入力の取得許可範囲・sidecar仕様・source固定・保証適用条件・native取得計画を具体化。科学module実行や入力開封を含めるならそのscopeを明示。新証明は別承認 |
| ユーザー → Codex | adapter実装/人工検証、native回路/合成取得、科学計算・保存値再解析は各々明示範囲で別承認。実行前に新SOURCE/input/policy、固定契約、保全・停止・retry条件を固定。既存grant/claim/STOPを再利用しない |

今回、上の未取得物は取得せず、SOURCE・grant・marker・runnerを作成しない。恣意的なCPU/RAM上限は設定しない。将来実行も共有環境に干渉せず、既存契約の資源条件を無断変更しない。**本草案の公開後mandatory STOP。**

[PRCONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/pr2_matched_accuracy_resource_contract_v1.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_p5.py
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/handoff.md
[W2C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/handoff.md
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[BITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[NOTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md
[DACC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
