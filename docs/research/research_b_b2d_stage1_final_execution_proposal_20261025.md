# 研究B：B2-D Stage 1最終実行提案

2026-10-25 JST。**未承認の提案書。科学実行不可、grant未発行・未claim。**
起点 `94394357d503b7326378dd62f647487199593dd5` の[前案](research_b_b2d_stage1_science_execution_contract_proposal_20261025.md)を保全した新版。
今回の依頼は準備だけの承認であり、以下の科学条件の正式採択・実行承認ではない。

## 1. 完成状況と固定SOURCE候補

| 項目 | 状態 |
|---|---|
| 科学SOURCE候補 | **`97a449ad53c74361ffc5c4cffd6de5f3f0f0a72e`**。コード・test・機械可読scopeを先に固定し、この提案書は後続の文書commitに置いた |
| 全scope | 科学的な処理境界と固定recipeの合法なruntime展開を登録した `b2d_stage1_scope_v2`。定義・実績照合を実装、人工確認済み |
| Launcher／失敗保存 | 接続実装・mock確認済み。実科学経路は未実行 |
| Production authority | **未接続**。参照可能な既存SOURCE・接続ツールに、独立した実ユーザー承認記録の取得・照合providerを確認できなかった |
| 実行契約全体 | **未完成・未採択**。authority接続と実行環境の固定、最終条件採択が残る。科学実行可能と表示しない |

入口は [`scripts/run_b2d_stage1.py`](../../scripts/run_b2d_stage1.py)。`--inspect` はmetadataだけを出力する。
`--execute` は現SOURCEでは `PRODUCTION_AUTHORITY_UNCONNECTED` で入力生成・出力作成・claim前に拒否する。
callerが任意のauthorityオブジェクトを渡しても代替承認にはならない。
実providerの接続でSOURCEが変わる場合は、新しい科学SOURCE候補へ更新・再固定してから承認を判断する。
候補commitをcheckoutしたことだけで実行できる契約ではない。

旧adapter `5d3ec883eeeabf69abfa128c9e169c59fc572149` とpublished PF `20d64c2fea1ed2d2188777df4f3d962ee94df486` の係数SOURCEは変更なし。
SOURCE closureは全 `src/**/*.py`、fixtureファイル、今回のlauncher scriptのbytes hash。実行時は契約のcommitとclean HEADも照合する。
文書公開commitと科学SOURCE候補commitを混同しない。実行checkoutは契約で正式採択したcommitへ合わせる。

## 2. 固定候補条件と未確認事項

| 項目 | 候補 |
|---|---|
| Fixture | `_nontrivial_basis_case` のconstructor literal。blob `0c92ae9c1e8c5721d9c8629a83435b855f5d6043`、定義commit `52ba88687f92d089921b207e60aec006f04141de`。helper自体は呼ばない |
| Model | synthetic 2-mode one-body＋3 DF因子、元constant・係数不変。分子geometry／AO basis／分子DF rank探索は非該当 |
| D/R | L_D=2、`lambda_frobenius_squared`のweight-ranked prefix。one-body→選択DF因子順。generation-prefixへの置換なし |
| Sector／state | N=1完全粒子数sector、sorted OpenFermion JW basis、mode0 occupation状態を全arm共有 |
| 比較 | A0 partial-S2、A1 standalone D-S2、A2 Yoshida4、A3 published Y8m10b、A4 published YP8m8 |
| Grid | δ=(+9/100,−9/100)、q=(2,3)、nested側n_D=(1,2)。A0は4行、他各8行、計36行 |
| Tail | r_RTE=2、K=2、seed41はmetadataのみ。analytic moments、samplingなし。RTE tolerance metadata=1.0 |
| Scalar／threshold | `extract_identity_phase`、coefficient_atol=0 |
| Precision／correctness | complex128、絶対atol=1e-12を候補として維持。物理的改善・科学的GOの閾値ではない |
| Signal | 同じphysical stateのX/Y、半径、corrected−physical／corrected−idealの複素直接誤差 |
| 位相 | 無効、radius_floor=None、branch_margin=None。principal phase／phase distanceはnull、unwrap／energy推定なし |
| 判定 | `descriptive_only_no_scientific_go`。自動順位・科学的GO／NO-GO・総資源優位の認定なし |

H／preparation／tail／array hash、選択factor index、実basis順・state、N=1でのresidual／非可換性、H identity、A4の数値correctnessは未確認。
実入力のhashを架空固定せず、承認したrecipeをclaim後に一回だけ解決し、provenance・scope receiptとresolved graphへbindingする。
事前graph layoutは順序slot・grid・演算DAG・数値条件を束縛し、resolved inputを含むgraph fingerprintとは区別する。

## 3. 全科学scopeの登録方式

[機械可読scope](research_b_b2d_stage1_scope_20261025.json)をそのまま契約へ束縛する案。
処理境界ごとのbudget vectorであり、入れ子の回数を足して「全jobの科学call総数」としない。
NumPy／BLAS内部の各演算は、それを内包する固定SOURCE関数のscopeへ含める。
無制限な追加処理は認めず、SOURCE・grid・recipe・合法な展開範囲・実績を照合する。

| 科学処理境界 | 登録回数／合法な展開 | 入力・依存・出力と実績照合 |
|---|---|---|
| Fixture生成 | input.fixture=1 | 固定blobのliteral→元DFHamiltonian。生成自体は今回未実行 |
| Split | input.split=1 | 同H→既存weight-ranked split。選択2因子／残差1因子のindexをreceiptへ保存 |
| Preparation／Gaussian基底 | input.preparation=1 | 既存one-body Gaussian生成1＋全DF生成3＋残差DF再生成1を内包。SOURCE・依存version固定、basis hash・operation数を記録 |
| Sector／ordering | input.sector=1 | 2 modes、N=1、basis indices=[1,2]、occupied modes=[0]。別sectorへ変更しない |
| Full H／D行列化 | input.full=1、input.block=3 | 同sectorの2×2行列。各materializeは既存matrix-free operatorの2列を実体化。D position 0,1,2を各一回 |
| Retained residual | input.tail=1 | 既存grouped tail。残差factor1、非identity成分1〜3、retained basis1に制限。Z0／Z1／Z0Z1だけ、抽出scalarは別扱い |
| 内部basis回路 | 上記preparation／tailに内包 | 必須のGaussian／Bogoliubov表現、basis QuantumCircuit／Operator、bit-order変換・leakageを含む。full PF wrapper／compileは含めない |
| State | input.state=1 | 同sector順のmode0 occupation vector。state最適化・ground-state eigensolveなし |
| Input correctness | DAG validate_input=1 | 既存backendの5行列Hermiticity、state norm、H=D+R+cI、D内3組／D-R1組の交換子検査。入力全体で共有 |
| Exact exponential／参照 | DAG exponential=410、backend.expm=448 unique cache key | [scope JSON](research_b_b2d_stage1_scope_20261025.json)のlabel/time keyのみ。A4独立参照の追加38 keyを含む。cache hitの再expmなし |
| A4数学参照 | DAG a4_reference_check=4 | 全δ／n_Dの独立P K^n P†・actual inverseを検査。全方式の性能比較に先行。旧validatorとの数値同値性を認定しない |
| A0〜A4 operator比較 | 共通DAG全941 dispatch | product_append94、power112、operator_error182、scalar phase56等をsource通り実行。node identity・依存・種別・回数を照合 |
| Finite RTE／normalization | rte.config=2、rte.moments=2 | 固定signed δごと。Taylor／power／normalizationを既存関数内へ含める。field4、normalization4、rescale36はDAG側の別境界 |
| Coherent signal | DAG signals=36、signal.expectation=144 | 4対象×36行。X/Y・半径・直接複素誤差。phase計算なし |
| 保存・STOP | 下記9ファイル以内 | SOURCE／契約binding、resolved input、実績、全行・部分結果、例外code、STOP。旧出力を上書きしない |

成分数・basis operation数を事前の架空数値で埋めない。固定recipeから合法域内でruntime解決し、recipe外の展開・重複・call超過を処理開始前に拒否する。
完走時は全境界の完了回数と3種類の入力resolution（preparation／sector／input）が揃うことを確認する。
失敗時は完走回数に補完せず、attempted／completed／failedと利用可能なreceiptを保存してSTOPする。

## 4. 承認・環境・grant接続

既存 `pr_dictionary_validation.authorization` のhash／packet／`consumed-<authorization digest>`／`claim_packet` を再利用する。
新しい汎用承認・grant管理は作っていない。旧grantの再利用・marker削除・旧STOP解除なし。

既存packetの `authorization.json`・`contract.json`・`user_approval.txt` とhash一致は、内容bindingの形式として利用できる。
しかし独立した実ユーザー承認の真正性を取得するproduction provider、そのtrusted record元・照合形式は今回確認できなかった。
そのため `production_authority()` はNoneであり、caller任意の文字列・file・authorityオブジェクトから承認へ昇格させない。
mock authorityはread-only packet検査に限り、science claim不可。実providerが利用可能になるまで**実行停止**を維持する。

将来の正式契約はSOURCE commit／closure、recipe fingerprint、layout、scope全文、fresh output、Python implementationと完全なversion_info、少なくともscopeに列記した9依存packageの実versionを束縛する。
現段階で実行環境のversion値は採択していない。qiskit-natureを含めてversion固定し、未導入時のGaussian fallbackへ実行中に移行しない。
最終採択文書と実承認記録を対応させ、既存運用で新規未使用grantを登録できた場合のみ、下記順へ進む。

1. 実承認記録・SOURCE／環境・契約／scope・fresh output・unused grantをread-only照合。
2. 既存 `claim_packet` でatomic claim。claim失敗時もmarkerを削除しない。競合outputへ書き込まない。
3. claim後handleに同scopeを接続し、固定入力を一回生成。resolved identity・scope receipt・graphを照合。
4. 入力correctness gate、**全4件のA4 correctness gate**を先行実行。
5. 全gate通過後だけ登録済みoperator／signal比較を実行し、全実績・結果を保存。
6. 完了・失敗いずれもmandatory STOP。retry／resumeなし。

今回、実承認record、実grant／ledger／markerは取得・発行・claim・変更していない。

## 5. Fresh output・失敗保存・precision

出力案は `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/`。今回は作成なし。
親directoryと未存在のfresh outputをpreflightで確認し、将来の承認済みatomic claimだけがoutputを作成する。
保存対象は `binding.json`、`inputs.json`、`inputs.npz`、`graph.json`、`events.jsonl`、`partial_results.json`、`result.json`、`status.json`、`STOP.json`。
失敗時は生成できたsubsetを保存する。入力前の拒否は理由codeをstdout／戻り値へ出し、秘密の承認本文・例外本文・他研究の記録を露出しない。

Log最大3126 record、1 record最大64 KiB、JSON文書最大1 MiB、総出力16 MiB、最大9ファイル、STOP用reserve256 KiBを提案する。
CPU／GPU／RAM／worker／phase・total wall timeには人工的な上限を設けない。
sourceの共有cache、grouped tail、sector行列化を再利用し、小行列の依存DAGへ不要なprocess複製をしない。速度倍率は未測定。

失敗codeは承認／SOURCE／grant／scope不一致、入力identity、sector／basis／state、residual／非可換性、H identity、A4 order／inverse、数値precision、途中例外、保存失敗を区別する。
実行開始後は有効・不適格両方のresolution候補、境界ごとの進捗、gate残差、完了した行・途中の数値診断を保持する。
保存失敗も成功扱いにせずSTOPを試み、戻り値に保存できなかったファイル名を残す。旧出力を上書きしない。

complex128の識別力は未資格。非有限数値等はINCONCLUSIVEとしてSTOPする。
有限な方式差が丸め・correctness残差と区別できる保証はなく、`precision_status=UNQUALIFIED_COMPLEX128_DESCRIPTIVE_ONLY` を残す。
曖昧な差を改善認定しない。自動高精度計算、追加δ、state／sector変更、科学的GO閾値は導入しない。

## 6. 承認前に研究Bへ返す事項

- **研究Bチャット／ユーザー**：表の入力・固定grid・complex128／atol候補・phase無効・runtime recipe binding・内部Gaussian basis scope・一回実行／STOPを正式採択するか判断する。
- **ユーザー／既存運用の担当者**：実承認記録を参照できるproduction authorityの既存取得元・形式・照合方法を指定する。自己作成packetだけでは代替しない。
- **Codex（次の提案）**：指定された既存providerが利用可能なら最小接続し、実行環境をbinding、必要なら科学SOURCEを更新・再固定する。新grantを発行・claimせず準備結果を返す。
- **ユーザー**：最終SOURCE・scope・承認記録接続・環境・新grant／STOPを確認し、別途科学実行を明示承認した場合だけStage 1を開始する。

[技術報告](research_b_b2d_stage1_final_preparation_technical_report_20261025.md)。既存46＋追加16＝62人工test PASS。科学的correctness・優位性の証拠ではない。
C1最優先を維持し、そのSOURCE・結果・branch・契約へ干渉しない。通常のB進行は研究Bへ返す。
公開内容・remote SHA確認後 **mandatory STOP**。今回の新科学計算・内部basis回路・科学backend・grant操作は0件。
