# 研究B：B2-D第一段階・限定科学実行契約案

2026-10-25 JST。**提案・実装準備のみ。未採択、科学実行不可、新grantなし。**
起点 `52ba88687f92d089921b207e60aec006f04141de` の[前案](research_b_b2d_science_preflight_proposal_20261025.md)を保全した新版。
今回の新実装・人工確認・未完成部分は[技術報告](research_b_b2d_input_preflight_technical_report_20261025.md)に記す。
本書は承認packetの `contract.json` でも実行承認でもない。

## 1. SOURCEと入力の固定案

| 項目 | 固定案・未確定事項 |
|---|---|
| Repository | `HIROMU1015/Partially-Randomized-Trotter` |
| 実装起点 | `52ba88687f92d089921b207e60aec006f04141de` |
| 旧adapter | `5d3ec883eeeabf69abfa128c9e169c59fc572149`、変更なし |
| Published PF | `HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae` の `20d64c2fea1ed2d2188777df4f3d962ee94df486`、係数関数変更なし |
| 科学SOURCE | **本準備を公開した最終commitを候補とし、研究Bで別途固定**。旧52baのまま今回の追加経路を実行しない |
| SOURCE closure | `preflight.source_inventory` で全 `src/**/*.py` とfixture定義のbytes hash。実行前にcommit、clean checkout、closure一致を検査 |
| Fixture | `tests/test_df_partial_s2_repeated.py::_nontrivial_basis_case`、blob `0c92ae9c1e8c5721d9c8629a83435b855f5d6043`。定義commitは52ba。helperのimport／直接呼出しなし |
| Hamiltonian | 同fixtureのconstant .17、2-mode one-body、3 DF因子・lambdaを不変で再構築する案。synthetic、geometry／AO basis非該当、分子DF rank探索なし |
| 分割 | `L_D=2`、`lambda_frobenius_squared`、降順weight-ranked prefix、同weight時original index順。generation-prefixではない |
| Ordered D | 既存preparationのone-body correction→選択DF fragmentの順を採用する案。**選択index・実際のblock hashは未生成** |
| Sector／state | N=1の完全粒子数sector、mode0 occupationの共通physical状態を候補とする。未採択・非可換性未確認 |

実入力hashを事前に架空固定しない。`unresolved_identity()` のH／preparation／tail／sector／state／array hashはnull。
承認対象は固定recipeと**登録SOURCEによる一回のruntime解決手順**とする案。
事前の `graph_layout` はD順序のslotとgrid・演算依存関係・correctness条件を束縛する。実入力を含む `graph.fingerprint` とは別物である。
生成後に選択index、D順序、scalar、正規化、sector basis、状態座標、shape、hashを保存し、resolved graphを同じlayoutと比較する。
このruntime binding方式も研究Bでの明示採択が必要。事前resolved input hashを要求する方式なら、別途入力生成の承認段階が必要である。

## 2. 入力経路とcorrectness gate

1. blob検査後、ASTからfixture最初の `DFHamiltonian` constructorのliteralを読む。これは今回実施した静的処理。計算時だけnumpy配列とHamiltonianを生成する。
2. `split_df_hamiltonian_by_ld` と `prepare_df_partial_s2` を再利用する。既存fixture helperのRTE request／繰返しbuilderは呼ばない。
3. `PhysicalSector.number_sector` のsorted OpenFermion JW basis順を全行列・stateで共有する。
4. full Hと各D blockを `dense_df_operator_in_sector` で行列化する。DF block単独ではconstant=0、one_body=0、one-body単独ではconstant=0・DF因子なしとする。`select_blocks` は元one-body・constantを保持するため単独Dには使わない。
5. retained tailは既存 `_normalized_symbolic_tail_in_sector` の基底group処理とbit-order permutationを再利用する。分母はretained symbolic Pauli L1 `exact_rte_lambda_r`。分割用ranking proxyではない。
6. OpenFermionではmode0 occupation bitは最上位。sorted basis内の該当座標へunit amplitudeを置く。既存のbasis変換を用い、state最適化・ground-state eigensolveをしない。
7. 既存 `DenseBackend._validate` がbytes hash、共通identity／順序、complex128、shape、Hermiticity、state norm、`H=sum(D)+lambda_R*R_normalized+(constant+extracted_identity)I`、D内／D-R非可換性を検査する。新たに同じnorm検査を複製しない。tail leakageだけloader側で既存返却値を照合する。
8. **全δ・全n_DのA4独立数学参照4件が完了してから、いずれのarmのoperator／signal比較にも進む。** 数学的 `P K^n P†` とactual inverseのgateであり、旧published validatorとの数値同値性の認定ではない。

**重要なscope依存：** 既存preparationはGaussian gate／basis表現を構築し、既存tail helperは `basis_change_unitary` 内で小さなQuantumCircuitとOperatorを生成する。
再利用経路に内在するこの入力基底処理を、科学call scopeへ含めて別途採択する必要がある。今回一切実行していない。
第一段階の名目でPF wrapper build、controlled wrapper、compile／transpile、resource pricingを追加しない。
内部基底回路も認めない契約を選ぶ場合は、入力経路の別実装・検証を先に準備し、同じgrantで自動置換しない。

N=1でretained residual、D内／D-R非可換性が残るかは**未確認**。backendが適格性を満たさない場合は保存・STOP。N=2、別fixture、別stateへの自動変更なし。

## 3. 比較条件の採択案

| 条件 | 候補 |
|---|---|
| Arm | A0現partial-S2、A1 standalone D-S2、A2 Yoshida4、A3 published Y8m10b、A4 published YP8m8 |
| δ／q／n_D | δ=±9/100、q=(2,3)、nested A1–A4のn_D=(1,2)。A0にn_Dを割り当てず4行、他各8行、合計36行 |
| r_RTE／K／seed | 2／2／41。seedはmetadata、analytic momentsのみでsamplingなし |
| RTE tolerance metadata | 1.0。physical correctness許容差ではない |
| Scalar／threshold | extract_identity_phase／coefficient_atol=0 |
| Precision | complex128。精度上昇、grid拡大、係数探索を自動実行しない |
| Numerical correctness | 絶対許容差1e-12を候補。**未採択**。input identity・state・非可換性の検出とA4数値参照に適用する案。物理的改善の閾値ではない |
| 主要signal診断 | corrected–physicalおよびcorrected–ideal-tailの**直接複素差**、各対象のX/Yと半径 |
| 位相 | 初回は `phase_diagnostics=False, radius_floor=None, branch_margin=None` を採択する案。位相はnull・NOT_REQUESTED。無根拠な正の閾値を補完しない |
| 位相を有効にする場合 | radius_floor>0、0<branch_margin<πの根拠と値を別途採択し、layout／契約へ束縛。principal phaseの点診断のみ。unwrap／energy推定なし |
| 判定 | descriptive_only_no_scientific_go。科学的materiality・GO／NO-GOの自動判定なし |

全armに同じH／D／tail／state／scalar／precisionを使用し、参照を共有する。A0 forward/reverseとnested同方向D半時間の配置差は保持する。
既存error／signal／normalization schemaを維持する。A0 D-half errorはnull。A4だけへの参照情報・最適化機会の供与なし。

## 4. 科学call登録案：全job総数は未確定

| 層 | 静的に数えた数・依存関係 | 残る登録 |
|---|---|---|
| Fixture／split／preparation | 各1回、上記順 | Gaussian基底生成・eigen分解・gate internalsの種類と回数 |
| Sector／state | N=1 sector 1回、occupation state 1回 | 実際のbasis／座標、正規化検査は数値gate |
| 行列化 | full H＋D3個のsector materialize計4回、grouped normalized tail 1回 | sector matvec内部回数、retained成分数・基底数と内部基底回路処理 |
| Input correctness | 1 dispatch。D pair交換子3組、D/R交換子1組 | Hermiticity／state／identity／leakage norm等の全低層内訳 |
| 共通DAG | 36行、941 dispatch。旧種別別個数は前案と同じ | 全job予算へ置換しない |
| Exponential | DAG410 key＋A4独立参照の追加38 key＝**backend cacheのunique key448** | この実装・3 D block・候補gridでの静的key数。入力内部のexponential等を含む総数ではない |
| A4 correctness | 4回、全arm比較前。参照内matrix multiplication1048回、cacheはDAGと共有 | 旧validator照合は未登録・未認定 |
| Operator積／power／norm | DAG product_append内matrix multiplication2276回、power112 dispatch、operator_error182 dispatch | A4内power4回、RTE内部power・norm等を区別して全scopeへ登録 |
| Finite RTE／normalization | moments2、field4、normalization4、rescale36、scalar phase56 dispatch | 既存RTE内部Taylor／power／正規化処理の全内訳 |
| Signal | signals36 dispatch、4対象×36=144 expectation呼出し | phase無効時はangle／phase-distance呼出し0。複素差・半径等の低層処理 |
| 保存・provenance | SOURCE／recipe／resolved input／graph／全row／gate／失敗情報 | 出力file・log／instruction scope、run実績とscope逸脱を検出するlauncher接続 |

`preflight.scope_proposal(graph)` は上表の機械可読案を返す。`fully_resolved=False`、11groupに未登録欄、未確定事項を保持する。
全groupでoperation・count・dependencyを固定した最終scopeを作り、科学的比較条件を変えない範囲で実行時実績の照合方法も確定する必要がある。
**これが済むまでgrant発行・科学実行をしない。** 未知の回数を0で埋めず、backend key448を全科学call数とも呼ばない。

## 5. Preflightと運用承認の接続

`verify_execution` はread-onlyで以下を拒否する：draft／未採択、条件／SOURCE／layout／call不一致、未確定scope、fixture blob変更、dirty SOURCE、依存version変更、fresh output不足、承認記録照合欠落、旧stage grant、消費済みmarker、retry／STOP違反。

既存 `pr_dictionary_validation.authorization` の `_canonical`／`_under`／`file_sha` とpacket／`consumed-<authorization digest>`規約を再利用する。
将来の実行launcherは既存 `claim_packet` を使う。今回claimを呼んでいない。既存packet・ledger・markerを読み書きする新しい汎用管理システムは作らない。

`authority.verify_approval` は**実ユーザーの別途承認を既存運用記録と照合する外部境界**。既定実装なし。
呼出し元の承認文字列や自己作成packetのhash一致だけでは通さない。mock authorityは明示され、科学claim／permitを得られない。
**本環境におけるproduction authorityの記録元・束縛方法は未確定**。この部分を完成した真正性保証と表示しない。
運用側で実承認・新grantを登録した後だけ、read-only preflight→既存atomic claim→claim後handle→入力生成→resolved settings／graph→評価の順で進める案である。
入力生成と評価は一度だけ。再取得・retry／resume不可。失敗時にmarkerを消さない。

## 6. 保存、失敗、inconclusive、STOP

Fresh output案：`artifacts/research_b_b2d_stage1/<execution-date>/<fresh-run-id>/`。今回は作成なし。
将来のlauncherは登録済み入力snapshot・SOURCE／環境・承認binding、resolved identity／basis順／array shape/hash、input gate、A4 gate、実call実績、全36rowとerror／normalization／signal、失敗理由・STOPを保存する。
数値入力検査またはA4 gate失敗、call／SOURCE逸脱、未採択条件発見では比較を始めずSTOP。途中結果と例外も保存する。
現 `evaluate` はin-memory APIであり、**失敗時を含む保存launcherと低層call実績照合は承認前の残作業**である。

非可換性不成立、residual消失、correctness残差・丸め誤差と同程度の方式差、precision資格不足、（有効化した場合の）半径／branch不適格はINCONCLUSIVE候補。
比較条件・精度を同じ承認で拡張しない。第一段階の記述結果からYP8m8優位、同精度総資源改善、QPE/RPE誤差保証を認定しない。

新契約ではphase／total wall-time cap、人工的なCPU／GPU／RAM／worker capを設けない。output／log／call／instruction scopeは確定させる。
支配負荷は承認前に既存timingと実資源から確認し、sector／cache／grouped tailを再利用する。小行列のDAGを無理にprocess分割せず、実負荷に応じて共有cache・vectorization・thread設定を選ぶ。
科学的条件を変える高速化、共有環境・他研究の設定変更はしない。今回32 CPU、利用可能RAM約52 GiB、GPU driver利用不可を確認し、stdlib人工testだけを独立processで並列実行した。速度倍率は主張しない。

## 7. 承認前に残る具体的作業・担当

1. **研究Bチャット／ユーザー**：fixture、N=1／mode0、固定grid、complex128／correctness atol候補、位相無効・直接複素signal主診断、runtime identity解決方式、内部Gaussian基底回路を含む入力scope、一回実行・retryなし・STOPの採否を判断する。
2. **Codex（今後の追加準備として提案、未実施）**：採択内容に沿い、内部基底／RTEの低層call scopeと実績照合、失敗保存launcher、実運用authorityの接続・環境bindingを完成し、最終科学SOURCE commitと全call／output inventoryを固定して返す。追加科学計算は行わない。
3. **ユーザー**：その具体物を確認後、新grantと別途明示的な限定科学実行承認を判断する。現草案の公開だけでは承認されない。

C1最優先を維持。通常のB進行は研究Bへ返し、統括報告を必須にしない。優先順位変更・重大な研究間調整・新中心方針だけ統括へ戻す。
本準備の公開内容・remote SHA照合後 **mandatory STOP**。詳細科学レビュー、科学計算、C1変更は開始しない。
