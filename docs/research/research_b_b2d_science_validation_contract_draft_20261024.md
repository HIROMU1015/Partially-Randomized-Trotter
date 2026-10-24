# 研究B：B2-D小規模科学検証契約草案

2026-10-24 JST。**未承認の契約草案。grant・科学実行許可・正式な研究機構採択ではない。** 今回完了したのは最小adapterと人工unit/mock確認までで、以下の科学処理は実行していない。

## 1. 目的・起点・今回の境界

D内部の既知processed PFによる精度改善が、PRのfull physical coherent signalへ残るかを識別する。YP8m8が最良であることや総資源利益を前提にしない。

- PRの正式設計起点：`1e2feb033bd7857482db09dd1f3d5b8692f100bf` の [B2限定設計][DESIGN]。
- 前段のaccess・保証・SOURCE調査：`a40a655b27da6e8528c3e6fba7d6d9158e7d3056` の [限定調査][SURVEY]。
- Published PF SOURCE：`HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae`、`20d64c2fea1ed2d2188777df4f3d962ee94df486`。
- 今回のadapter・人工検証・技術報告：[`b2d_adapter/README.md`](../../src/trottertracks/b2d_adapter/README.md)。科学SOURCEとして採用するcommit／追加接続は、実行承認前に別途固定する。branch名だけを科学SOURCE identityにしない。
- 正式共通指示 `research_project_instructions_unified_2026-10-23.md` は実参照済み。研究CのC1（圧縮DF）最優先を維持し、C1のSOURCE・結果・実行契約を変更しない。

今回の追加SOURCEは抽象block・signed duration・phase metadataを扱うだけで、Hamiltonian、unitary、state、Qiskit circuitを生成しない。science runner、grant、call消費処理も追加していない。

## 2. 候補fixtureと静的に確認できたこと

候補は起点commitの [`tests/test_df_partial_s2_repeated.py`][FIXTURE] 内 `_nontrivial_basis_case`。元ファイルのgit blobは `0c92ae9c1e8c5721d9c8629a83435b855f5d6043`。今回、この関数をimport・呼出しせずに定義と既存assert対象を読んだ。

| 項目 | 既存定義／今回の証拠範囲 |
|---|---|
| モデル | 固定synthetic one-body＋3 DF factor。分子geometry・AO basis・分子DF rank探索は非該当 |
| 係数 | constant `0.17`、one-body `[[0.4,0.13],[0.13,-0.2]]`、DF係数 `[0.6,-0.35,0.22]`、3個の2×2 G行列を元関数に保存。転記からHamiltonianを再構築していない |
| 分割 | `split_df_hamiltonian_by_ld(hamiltonian,2)`。既存defaultは `lambda_frobenius_squared` のweight-ranked prefixで、generation-prefixの先頭2個とは限らない |
| 正確な所属・順序 | selected fragment indices、one-bodyを含むD block順序、basis／tail／Hamiltonian hashは将来のidentity確認で記録する。今回rankingやHamiltonian hashを新規計算していない |
| 既存helperのtail設定 | `identity_policy='extract_identity_phase'`、`r_RTE=2`、K2、helperのtruncation tolerance `1.0`、seed41。これは旧helperの条件の静的記録で、新B2実行条件の採択ではない |
| 時間・反復 | helperの `step_time`／`repetition_count` は呼出し元から渡す引数。既存testには±0.09・反復2/3のcaseがあるが、新B2の時間窓・qへ自動流用しない |
| 既存assert | Dに3個の異なるbasis、既存partial-S2のraw／boundary-optimized／controlledのdense照合、scalar・event順序。今回testを実行しておらず、新B2列を検証するものでもない |
| 非可換性 | 非対角係数と異なるbasisを持つfixture定義であることまでは読取り済み。**D内部・D/R境界のcommutatorが必要なsectorで非零かは未評価**。関数名やbasisの相違だけで保証しない |

将来、非可換性が十分でない、sectorが自明、tailが消える等の条件が判明した場合はその事実を報告する。同じ許可で別fixtureを探索・置換しない。fixtureの最終採択と必要なidentity処理は科学実行の承認対象である。

## 3. 予定比較集合と列の定義

今回の指示で指定された予定集合は次の5方式。係数や比較集合を結果後に有利に変更しない。

| Arm | 一stepの違い |
|---|---|
| 現行partial-S2 | `R_D(δ/2) V_R(δ) F_D(δ/2)`。既存のforward／reverse sweepを保持 |
| 同配置の標準S2 | standalone D-S2で半時間発展 `G_D` を作り、`G_D V_R G_D` に配置 |
| Yoshida4 | 同じordered D block・同じ配置で既知Yoshida4を使用 |
| Published Morales Y8m10b | 同じD配置でpublished non-processed係数を使用。legacy m8とは区別 |
| Published YP8m8（B2-D） | `h=δ/2, τ=h/n_D`、`G_D=P_D(τ) K_D(τ)^n_D P_D(τ)†`、左右は同方向の `G_D` |

今回のadapterが実装するのはYP8m8の**記述列**であり、5armの科学評価器は未実装。現partial-S2とnested配置の差をYP8m8固有の利益に数えず、同配置S2との比較で分離する。n_Dや各baselineの内部反復機会、計算・情報budgetは結果前に公平に固定する。

同じsnapshot・D/R分割・block順序・scalar／identity／threshold policy・tail辞書・tail時間／`r_RTE,K`・入力状態／sector・physical時間・精度・confidence・費用scopeを使用する。exact行列やexact ground stateを参照に使う場合は全arm共通の診断情報とし、候補だけの無料量子access・無料state preparationには数えない。

## 4. 第一段階の評価対象

以下は将来計算する差の定義であり、今回その値・係数・bound・改善率を生成していない。

| 評価対象 | 目的・区別 |
|---|---|
| D内部operator | 各 `G_D(h)` と `exp(-ihH_D)` を比較し、D単独の改善を識別 |
| Exact-D/R outer分割 | `J(δ)=U_D(δ/2) U_R(δ) U_D(δ/2)`、`J(δ)^q` と `exp(-iTH)` を比較。Dをexactにしても残るbiasを分離 |
| Ideal-tail full列 | 各armにexact tailを入れたordered列を定義。現partial-S2とnested列は別々に扱う |
| Normalization-corrected mean | 実装するevent lawのordered meanと各normalizationから作る `A_a`。各armのideal-tail列との差が有限Taylor bias、physical Uとの差がtotal physical bias |
| Raw mean | `M_a=E[V_a]` と `A_a=N_tot M_a` を区別。corrected meanは一般にunitaryではない |
| X/Y coherent signal | 同じphysical状態ψについて `⟨ψ|A_a|ψ⟩` とphysical参照を比較。raw signalも別に保存 |
| 半径・位相 | 半径の下限・branch／continuity条件が成立する範囲でのみphaseを判断。小半径／分枝曖昧は未判定にする基準を事前に採択 |
| Spectrum診断 | 必要な場合に限り、kernel eigenphaseとfull fixed-state signalを別に扱う。Pによる相似変換をkernel固有値改善とはしない |

有限RTEのnormalization補正とprocessorによる補正は別である。各armでfinite targetが変わるため、旧finite-target一致や旧数値許容差をphysical bias・finite-bit allowanceの保証に転用しない。第一段階にshot sampling／N認証を自動的に含めない。

## 5. 固定済み事項と、実行前に採択が必要な事項

| 項目 | 状態 |
|---|---|
| 作業起点・published係数出所・5比較候補 | 今回の準備scopeとして指定済み |
| 旧helperのfixture定義と旧tail設定 | 静的な参照元は固定済み。新科学条件としての最終採択ではない |
| Adapter記述規約 | 数学的積を明示しappend順に変換。actual descriptor adjoint、残差barrier、scalar／event phase・normalization metadata保持を人工検査済み |
| 科学SOURCE・5arm evaluator／接続 | **未採択・未実装**。今回公開commitを参照して必要差分を別承認で固定 |
| Fixtureの採択・resolved Hamiltonian／split／basis identity | **未採択／未評価** |
| State・particle sector・exact-state参照の可否 | **未採択** |
| Physical時間T、delta窓、q、D内部n_D | **未採択**。旧test条件や人工testの有理数を実行条件にしない |
| Tailのr_RTE／K、threshold／identity policy、sampling／seed契約 | **新B2用には未採択**。旧helper値を採用するかを明示判断 |
| 各armの内部反復と情報／評価budget | **未採択**。候補にだけ有利な選択機会を与えない |
| Operator／signal数値許容差、precision、numerical-noise／branch判定 | **未採択**。mock PASSや旧test toleranceから生成しない |
| 科学的GO／NO-GO・materiality閾値 | **未採択**。D-only改善、full改善、baseline差を別々に判定する条件が必要 |
| 科学call数・入力／arm／時刻／状態の登録一覧 | **未採択**。science evaluatorとgrid採択後、必要callのscopeと件数を結果前に固定 |
| grant・retry／resume・停止／技術的不具合対応 | **新規の承認が必要**。旧grantやone-shotを流用せず、retry回数等を今回恣意的に指定しない |
| 結果保存先・出力項目・SOURCE/input provenance | **実行前に固定**。既存結果を上書きしないfresh出力。planned directoryの作成も今回不要 |

これらが未採択であることは草案の未完成を意味せず、実行許可と混同しないための明示的な未確定項目である。経過時間・branch公開・人工test PASSから承認を推定しない。

## 6. 必要となる科学処理と別途承認

1. **科学評価器の接続実装**：5armの参照・D記述列を同じblock定義へ結ぶ。今回の抽象記述列をQiskit gateやoperatorへ変換する処理は今回作っていない。既存 `finite_rte_signal_validation.py` は回路生成やeigを含むため、そのまま呼べば第一段階の許可範囲に入るとは仮定しない。
2. **Fixture identity／性質確認**：synthetic Hamiltonianの生成、split／sector解決、D内・D/R非可換性、exact参照の作成。分子生成・C1再実行は含めない。
3. **固定比較のoperator／signal評価**：採択されたtime／q／n_D／r_RTE／K／state一覧に限り、D、outer、full、corrected／raw、X/Y、radius／phaseを評価。必要なprecision確認と件数も承認scopeへ含める。
4. **結果固定と報告**：全arm・全登録点の成功／不一致／未判定を保存し、許容差や候補集合を事後変更しない。比較結果の集計はこの将来実行承認の明示scopeで行う。

今回これらの科学callは全て未実行である。将来の第一段階で回路buildを必要とする経路を採用する場合も、その必要範囲を明記して別途承認する。第一段階という名称だけで回路生成・compileを包括許可しない。

## 7. 結果から可能にする判断と、第二段階の境界

- D精度改善がfull physical signalへ残らない場合は、その結果をそのまま報告する。D/R bias・誤差相殺・noise-floor等を区別し、同じ許可で別時刻・別split・別fixtureを探索しない。
- Full改善があってもYoshida4／published Y8m10b等と同等なら、既知baselineに対する追加価値を認定しない。YP8m8が勝つまで候補を増やさない。
- 第一段階の改善はendpoint込みの総資源利益ではない。改善余地が残り、研究B／統括／ユーザーが必要と判断した場合だけ、第二段階を提案する。

第二段階は、native full wrapper・endpoint／境界費、controlled relative phaseの数値同値性、finite-bit bias allowance、weighted range／moment・IID・confidence・必要shot数、same-accuracy総資源を対象とする。対象SOURCE、compiler／synthesis、state preparation／readout、call数、統計資格を改めて固定して**別承認**する。第一段階完了から自動移行しない。

## 8. 保全・実行前条件・STOP

科学実行前に、上記の未採択欄を研究B／統括／ユーザーが判断し、必要な実装、科学call、grant、retry、結果保存・STOPを明示的に承認する。重要な理論・新規性・研究機構採択の詳細科学レビューも、必要な場合だけ別の開始承認を受ける。

将来の新計算契約にphase／total wall-time capや恣意的なCPU／GPU／memory／worker capを既定で設けない。計算前に利用可能資源と既存実装・支配負荷を確認し、適切に高速化／並列化し、出力・log・科学call・instruction制限はscopeとして固定する。旧固定契約のresource／retry／STOPを無断で変更せず、共有環境・他ユーザーへ影響する操作をしない。

固定SOURCE・raw・grant・ledger・marker・科学契約・旧STOPを維持し、C1との共同実装・合同科学計算を始めない。本準備の公開確認後は **mandatory STOP**。次は研究B／統括／ユーザーが未採択条件と別途承認範囲を判断する。

[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1e2feb033bd7857482db09dd1f3d5b8692f100bf/docs/research/research_b_b2_limited_design_20261024.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a40a655b27da6e8528c3e6fba7d6d9158e7d3056/docs/research/research_b_b4_b2_pr_connection_static_survey_20261024.md
[FIXTURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1e2feb033bd7857482db09dd1f3d5b8692f100bf/tests/test_df_partial_s2_repeated.py
