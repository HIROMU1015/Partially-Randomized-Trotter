# H6純RTE/Hybrid：全lawへの実装誤差転送の技術調査

2026-11-02 JST。読取基点は`71432728ee0140fb3c1f8bf30ecf51b2faba86ec`。
**転送不等式とrestrictedな証明経路は既存資産を再利用できる。ただし、現行opt1の全law実装誤差上界は、現在の証拠では確定できない。**
主な不足はnativeの集約相対位相と、compilerの角度binding・1Q数値再合成に対する一様な誤差評価である。
不可能性や予算超過を示したわけではない。今回はSOURCE読取・既存証拠の対応付けだけで終了し、証明や数値評価を無条件に継続しない。

## 1. 固定対象と今回行ったこと

linear H6／隣接1.00 Å／STO-3G／中性singlet、保存Pauli-JW918非identity項＋constant、system12/full4096。
T=.8/S2/q8/K2/P=0、central rotation、Full13qubit/ancilla wire12、cosine/+Im sine。
DF rankは適用外。保存compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、追加backend/coupling/layout/routing指定なし。

| ID | 固定候補 | L_D/r | event位置数 | 比較上の役割 |
|---|---|---|---|---|
| R64 | H6_p000_q8_r64_K2 | 0/64 | 512 | 純RTE。下記2比較で共用 |
| H32 | PR_p0078_q8_r32_K2 | 78/32 | 256 | R64との既存有力候補比較 |
| H64 | PR_p0078_q8_r64_K2 | 78/64 | 512 | R64との同q/r/K対照。辞書・lawは異なる |

新しい状態作用・QPY deserialize・sampling/RNG replay・回路生成/compile・誤差数値評価・N/G/感度/価格再集計・人工数値testsはすべて0。
既存科学moduleをimportせず、Python ASTの比較、テキスト・保存JSONの読取、文書整合性確認だけを行った。
局所診断の[REPORT][LOCAL]と[STOP][STOP]、[証拠調査書](h6_pure_rte_hybrid_implementation_evidence_inventory_20261102.md)、
既存H4の[R2転送設計](../non_df_h4_central_r2_design_20261025.md)と[R2-C限定証明](../non_df_h4_r2c_compiler_phase_proof_20261025.md)を再利用した。
今回の調査は独立した科学レビュー・新しい数値証拠・証明完成の認定ではない。

## 2. SOURCEと保存証拠の接続

| 役割 | 固定版／参照 | 今回確認した範囲 |
|---|---|---|
| R64の元実装 | SOURCE `e755ecf581600192fd2745d7066e410cebd933d3`、結果`10cb3e5cca7834ebf75301bd91bd093ca51d4a6b` | append・trajectory・event/sampler・law exporter・tight平均・wrapper/compilerの該当関数を現checkoutとAST比較 |
| H32/H64の元実装 | SOURCE `887b69e449cb3c378286f09d0d7626599850fb67`、結果`00958ae1945c780718b2deef07d5dab1096a5cd8` | 同じ関数群をAST比較。元2 SOURCEとも上記関数は一致 |
| D2 central control | `h4_d2_pauli_control._real/_tokens/_rotation` | 元2 SOURCEとAST一致。`central_control_body`はP2でhelper抽出済みだが、tokenからのemit loop・phase転送・returnのASTは元実装と一致 |
| selected-target作用 | SOURCE `9d392de7c8c6f8ae35faba9aeb1c8c6f6a215fef`、結果[71432728…][LOCAL] | `h6_selected_implementation.selected_action`の保持operand・product順・符号・scalar所有を読取。再作用なし |
| 保存law/target/QPY | [固定INVENTORY](../../artifacts/h6_pure_rte_hybrid_evidence_audit/2026-11-02/INVENTORY.json)、[局所preflight](../../artifacts/h6_selected_implementation/2026-11-02/run01/preflight.json) | 過去のarchive/member/QPY照合と6 Fullの意味論照合を引用。今回新たにQPYをdecodeした検査とは数えない |
| restricted compiler | 既存[R2-C SOURCE一覧](../../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/upstream_sources.json)と`upstream/` | 保存された公式Qiskit1.3.0 Rustのphase setter・Euler/opt1処理、既存環境のPython level1/plugin/dispatcherを読取 |

元SOURCEとのAST比較はコメント・位置情報を除く静的な一致確認で、実行結果やbackend全体の同値証明ではない。
特にP2の`source_tokens`のPython `%`によるphase復元を、元QiskitのRust `rem_euclid`に対する全域誤差証明へ置き換えない。

H6の保存targetは既に53-bit実効CDF質量と保存binary64 event角度を含む。
`target_semantics=C_EFFECTIVE_GRID_LAW_EXACT_RETAINED_PRODUCTS_NATIVE_STEP_SCALAR`であり、
P3のfinite-bit gap=0という出発点ではない。CDF/atan/normalizationの再計算やP3検証の再実装は不要。
全lawの確率は保存`component_mapping.exact_masses`と`order_mapping.exact_masses`を正本とし、
個別eventの名目`event_probability`や観測頻度で代用しない。IID uniform-wordという既存仮定を維持し、PCG列の独立性まで新しく保証したとはしない。
K2のsupportはorder0/2。order4は対象に加えない。

## 3. 再利用できる転送式と、局所η/δとの違い

以下は既存R2の条件付き不等式をこの固定familyへ対応付けたもので、H6の新しい誤差数値を評価したものではない。
同じ保存lawのtrajectoryをω、同じraw stateをψ、exact normをνとする。
`U_C(ω)`はselected-target unitary、`V(ω)`はactual binary64 operandで定義したnative body、
`W_in,a(ω), W_out,a(ω)`は末尾measureを除く両Full wrapperである。
同じwire/測定対応、unitaryなprefix、native control恒等式を全対象で示した上で、

\[
D(\omega)=\|V(\omega)-U_C(\omega)\|_2,\qquad
K_a(\omega)=\|W_{out,a}(\omega)-W_{in,a}(\omega)\|_2
\]

と置くと、

\[
\rho_a=\left|B_{saved}\mathbb E_p[m_{out,a}-m_{C,a}]\right|
\le B_{saved}\mathbb E_p\!\left[\min\{2,D(\omega)+2K_a(\omega)\}\right].
\]

body→Hadamard信号の係数は1、一般wrapper→Z期待値の係数は2。
B_savedは候補別に一回だけ、νは同一の分母とし、出力normで再正規化しない。
本書のD/Kはoperator normで、局所REPORTのB補正済み**信号差**η/δと同一量ではない。
full-wrapper全体のglobal phaseの非観測性と、controlled bodyのancilla-relative phaseを区別する。
後者をphase alignmentで消去しない。

全ωで`D≤D_*`, `K_a≤K_a,*`が示せればuniform式
`rho_a ≤ B_saved min(2,D_*+2K_a,*)`を使える。
また排他的stratumの全経路bound d_sと正しい実効質量w_sがあれば`rho_a ≤ B_saved Σ_s w_s d_s`を使える。
いずれも保存標本の最大・平均・SEではなく、全経路を覆う前提が必要である。

## 4. 各辺の実現可能性と不足

| 辺 | 既存資産で使える構造 | H6全lawに不足する条件 |
|---|---|---|
| target→nativeの制御意味論 | 任意nonidentity Pauliのbasis/parity/CRZ共役、chronologicalなcontrolled products、order位相、+Im wrapperの記号的恒等式 | 元生成algorithm全体への対応。selected QPYだけで全入力対応を認定しない |
| 保持角度 | α_j=exact(c_j)exact(δ)/2とβ_j=exact(fl(2c_j(δ/2)))/2の差。R64は保持和が空、Hybrid2候補は共通保持prefix | binary64倍/半分の適用域・丸め順とoperand差の包含。今回は差を未評価 |
| event角度・signed products | targetもnativeも同じ保存角度。有限・非overflow/underflow域で符号/2倍がexactなら当該rotation差は0。Pauli products自体はexact | 対象operand域の根拠。負product/order位相は下記phase項が所有し、別に二重加算しない |
| nativeの集約phase | scalar_stepはtargetにも保存済み。既存R2-Cの加算・remainder・windingの誤差分解を再利用できる | π定数/np.angle、float加算、CircuitData phase setterの実remainder誤差と全経路winding範囲。scalarを再度full-H誤差として加えない |
| native→compiledのlowering | 元default compiler/transpile呼出は一致。無coupling/initial_layoutのlevel1はlayout/routingを設定しない | 到達する等cost rule閉包、ParameterExpression/SymEngineのbinding、定数・相対phaseを含む数値意味論 |
| opt1の1Q再合成 | 既存の有限run/accepted rewrite構造とexact Eulerの補題 | actual Complex64行列形成、det/abs/arg/atan2、ZSXX簡約/angle snap/moduloの一様誤差。gate数を比較する`old_error/new_error`はoperator誤差ではない |
| 全lawへの集約 | uniform/stratum-weighted転送式 | 各ωを覆う上記D/Kと、正常有限処理・同じ測定意味を保つcompiler familyの定義。未生成QPYをすべて保存すること自体は必須でない |

R2-Cは**PROOF_INCOMPLETE**のままである。exact Eulerの特異点に対する補題があっても、
実際のRust/libmによる計算へ接続する未証明項は消えない。
現在のH6環境のPython dispatcherもaccelerated backendへ渡しており、PythonのSOURCE読取だけでその数値辺は閉じない。
これは実装が不正という判定でも、wheelの再ビルドや環境のbitwise一致を新たな普遍要件とする判断でもない。

native bridgeは、位相差d_phaseの評価が成立すれば、既存式
`D(ω) ≤ min(2, 2q Σ_j |β_j−α_j| + d_phase(ω))`へ接続できる。
compiled側は、lowering・accepted rewrite・phase更新の各operator差をtelescopingで合成する必要がある。
1Q run融合はevent境界を跨ぐため、孤立eventのcompile誤差を足すだけでは成立しない。
K2によってproduct数の範囲は狭まるが、この境界跨ぎや浮動再合成の課題は残る。

`h6_selected_implementation.program_resolution/rotation_coefficients`は**与えられた作用計算のengineering分解能**を出す。
compilerが任意入力をどれだけ変えるかというD/Kの証明ではない。
同様に保存law平均のtight L2包含やgiven-program Arbの評価誤差を、全law compiler誤差へ流用できない。
局所6組の小さい値からuniform係数を選ぶことも行わない。

## 5. 現時点の判断と、代替の限定scope

**判定：条件付き転送式は利用可能、現行opt1の非自明な全law数値上界・保存許容幅への所属は未確定。**
`whole_law_rho_upper / uniform_native_upper / uniform_compiler_upper / actual_N / actual_G`は未取得のまま。
全law保証が成立したとも、保証が不可能とも認定しない。
既存局所engineering確認と、仮想ρ=0・保存価格標本に条件付けた純RTE低Gはその範囲で保持する。

| ユーザーが選べる次の範囲 | Codexが行うこと | 得られるもの／残るもの |
|---|---|---|
| 現在の条件付き成果を使用 | 既存比較と局所確認をその限定範囲で引用 | 追加科学計算は不要。全law/価格tail/実装込み順位は主張しない |
| native bridgeだけを先に限定 | 元phase setterのremainder/winding・保持operandというscalar条件に対象を絞り、まず不足補題を整理 | compiler辺は未解決のまま。H6 operandから新上界を出すscalar数値評価は、SOURCE/精度/件数/STOPを別固定して明示許可後だけ |
| 現行compilerの証明を限定継続 | 既存R2-Cの到達lowering/binding、phase処理、2×2再合成に範囲を絞る | 最初はH6状態作用不要。前提が閉じなければ未確定を報告して終了。無条件に完遂しない |

この調査から直ちに必要と特定できた新しいH6状態作用・大量trajectory追加・compileはない。
解析可能な別compiler/templateへ変更する案は別の研究判断であり、旧compiled価格をその新familyへ流用できない。今回は変更しない。
law-weighted評価は先に局所上界の全経路妥当性が必要で、確率だけを精密化してcompiler不足を埋めることはできない。

全lawρが得られ、保存rho_guard内に入っても、ρ=0で算出したN/Gと費用順位の維持は自動的には保証されない。
必要なら、そのρを一度だけ誤差会計へ接続する別scopeのN/G感度確認を行う。
既存bias/numerical allowance/native semantic項との重複を確認し、同じ辺を二重計上しない。
真の期待compiled価格・Gの順位には、order2等の価格tail/選抜後不確かさが独立に残る。今回tail評価や新確率を計算していない。
P3は一区切りを維持し、DF B2・決定論的PF・RPE全roundという元研究の目的を置き換えない。

## 6. 保全・完了・次の担当

コード・tests・既存科学artifactは変更しない。既存manifest entriesは保持し、今回の非実行調査の対応だけを追記する。
旧SOURCE/raw/QPY/結果・既存P1/P2/P3の全STOP・局所診断STOPを保全する。
過去の人工PASSは過去の証拠で、今回の新testsや独立レビューと数えない。
文書整合性の技術確認はPASS：旧manifest209 entriesの内容不変、変更対象は本調査の文書/索引/manifestの7ファイルのみ、
既存科学コード・artifactに変更なし、局所STOPのbytes一致、調査書/新ノートのlocal link7件有効、`git diff --check`異常なし。
成果の作業branchは`codex/h6-whole-law-transfer-feasibility-20261102`。本調査書と関連文書のcommit/pushによって科学的な検証範囲やSTOPを変更しない。

現在の`STOP_H6_SELECTED_IMPLEMENTATION_LOCAL_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP`を維持して本調査を終了する。
**次の担当はユーザー**：条件付き成果を現状で使用するか、native scalar条件またはrestricted compiler証明の限定継続が必要か判断する。
**Codex**は選択された範囲だけを次指示後に進める。新しい数値誤差評価・状態作用・sampling/compile・N/G再解析は、その具体的scopeの明示許可前には開始しない。

[LOCAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71432728ee0140fb3c1f8bf30ecf51b2faba86ec/artifacts/h6_selected_implementation/2026-11-02/run01/REPORT.md
[STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71432728ee0140fb3c1f8bf30ecf51b2faba86ec/artifacts/h6_selected_implementation/2026-11-02/run01/STOP.json
