# B2-D Input package・preflight技術報告

2026-10-25 JST。**実装と人工確認のみ、科学実行0、新grant発行・消費0。**
固定起点 `52ba88687f92d089921b207e60aec006f04141de`、独立branch `codex/research-b-b2d-input-preflight-20261025`。
[限定科学実行契約案](research_b_b2d_stage1_science_execution_contract_proposal_20261025.md)は未採択で、残作業がある。科学実行可能との完成宣言はしない。

## 実装した接続

- [`input_package.py`](../../src/trottertracks/b2d_evaluator/input_package.py)：fixed blobのAST literal読取り、InputRecipe、nullの未解決identity、opaque metadata接続、lazyな科学loader。fixture helper自体はimport／呼出ししない。
- [`preflight.py`](../../src/trottertracks/b2d_evaluator/preflight.py)：固定grid、事前graph layout、全call scopeの不足拒否、clean commit／SOURCE closure／dependency、fresh output、新B2-D packet／消費marker、外部実承認照合境界。発行APIなし。mock authorityからscience claim不可。
- 既存 [`evaluation.py`](../../src/trottertracks/b2d_evaluator/evaluation.py) へ必要最小限の拡張：位相無効の明示policy、science permit／mock区別、全correctness gateの先行実行。DAGの演算node・依存関係・36行・941 dispatchを維持し、評価順だけ依存解決型へ変更。
- 既存 [`dense_backend.py`](../../src/trottertracks/b2d_evaluator/dense_backend.py)：科学import前のclaim確認、位相無効時NOT_REQUESTED／null。既存数値入力検査・normalizationを再利用。実backendはinstantiateしていない。

`PreparedDenseInputs` の各項目へ、preparation／tail／Hamiltonian hash、ordered blocks、full physical H、normalized retained tail、symbolic tail、共通state、sector／state identityを供給する経路を追加した。
実数値検査は既存backendに集約し、loaderで複製しない。scalarは既存 `context_from_preparation` のconstant＋extracted tail identityを使用する。
事前layoutからは未生成のinput identityを除き、runtime生成後にclaim handleへbindingし、resolved settingsと照合する。事前layout hashを実入力のgraph hashと混同しない。

## 再利用SOURCEと新規glueが必要だった理由

すべて起点52baの既存実装を静的に確認した。旧adapter5d3と固定PF20d64の移植関数は変更していない。

| 既存関数／SOURCE | 再利用方法・注意点 |
|---|---|
| `_nontrivial_basis_case` | constructor literalだけを固定blobから読む。helper内request／回路を起動しないためASTの最小glueが必要 |
| `df_partial_randomized_pf.split_df_hamiltonian_by_ld` | `abs(lambda)*||G||_F²`の降順、original index tie-break。generation orderを代用しない |
| `df_partial_s2.prepare_df_partial_s2` | ordered D specs、scalar、symbolic tail、準備hashを再利用。内部Gaussian基底処理は将来の科学scope |
| `df_hamiltonian.PhysicalSector.number_sector` | sorted OpenFermion JW number-sector basis。mode0最上位bitで共通occupation stateを置くglueを追加 |
| `finite_rte_signal_validation.dense_df_operator_in_sector` | full H・D単独を既存matrix-free operator経由でmaterialize。`select_blocks` はconstant／one-bodyを保持するため、各Dの引数を明示的に分離 |
| `_normalized_symbolic_tail_in_sector`／`_qiskit_to_openfermion_sector_permutation` | grouped basisとretained normalization、bit ordering、leakageを再利用。内部basis QuantumCircuitをscopeに含める必要あり |
| `DenseBackend._validate`／既存RTE／signal | H identity、Hermiticity、state norm、非可換性、moments等の計算を新実装しない |
| `pr_dictionary_validation.authorization` | hash／path／canonicalと既存packet／atomic claim規約。別研究の承認・grantをB2へ流用しない |

共通運用指示2026-10-23版は参照済み、SHA-256 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。
本段階で科学的判断・既存数値結果・証拠台帳の状態変更はない。旧草案・raw・grant・ledger・marker・mandatory STOPは保全した。

## 人工testと技術的に分かった範囲

Python 3.11.0rc1、`-S`、stdlibのみ。科学import barrierを使用した。

```bash
/usr/bin/python3.11 -S tests/management/test_b2d_adapter.py -v
/usr/bin/python3.11 -S tests/management/test_b2d_evaluator.py -v
/usr/bin/python3.11 -S tests/management/test_b2d_input_preflight.py -v
```

**既存15＋14件、新規17件＝46件PASS。** 旧mock approval helperに `mock_only=True` を付けた以外、既存人工testのassertionを緩和していない。
固定fixture bytes／literal、opaque D順序・scalar／tail・sector metadata、null identity、位相無効schema、全A4 gate先行と失敗時比較未実行、無承認科学経路拒否、未確定scope拒否、SOURCE／grid／call／output／旧grant／retry不一致、mock承認packetのread-only確認とclaim拒否を検査した。
mock signal schemaのtestは定数complex labelと注入stubだけを使い、physical state／行列／科学的expectationを生成・評価していない。
実packet、既存grant／ledger／markerにはアクセスしていない。人工packet・markerは隔離されたTemporaryDirectory内のmockであり、新science grantの発行・消費ではない。

資源確認：32 CPU、RAM約62 GiB（available約52 GiB）。`nvidia-smi` はdriver接続不可。3つの独立した軽量test processを並列実行し、CPU／RAM／worker上限や共有環境設定を変更していない。
科学test suite、fixture helper、numpy／scipy／Qiskit／trotterlib数値依存はimportしていない。実Hamiltonian・行列・expm・交換子・eigen・回路build／compile・保存科学結果再解析・詳細科学レビューは0件。

## 完成事項と未完成事項

**完成：** loaderの呼出し経路とmetadata組立、lazy分離、science入口のclaim境界、phase無効policy、全correctness先行、拒否条件を持つread-only preflight、call登録案、人工test、自己完結した契約案。

**未確認：** 実H／preparation／tail／array hash、選択DF index、N=1での残差・非可換性、実basis ordering／stateの数値同値性、H identity、A4参照残差、旧validatorとの数値同値性、complex128の識別力。

**実運用に残るもの：** 実承認記録のproduction authority接続、環境の採択とbinding、全低層call inventory・実績照合、失敗保存を含むlauncher。
これは架空の値や自己申告の承認文字列で埋められない。preflightはこれらを未確定のまま通してよい資格を与えない。scope草案をそのまま渡すと拒否する。
したがって**「本準備だけで科学実行preflightの全実運用接続が完成した」とはしない**。

DAG外入力と内部科学処理を11groupに整理。静的にbackend expm key448（DAG410＋独立A4追加38）、DAG product積2276、A4参照積1048、signal expectation144を数えられた。
これらはSOURCE descriptorからの計数で、実行済み科学call／実測性能ではない。総科学call数は未確定。新grantを発行できる段階ではない。

## 次の担当・STOP

研究Bチャット／ユーザーが[契約案の採択事項と残作業](research_b_b2d_stage1_science_execution_contract_proposal_20261025.md)を確認する。
Codexの次の提案は、採択内容に沿う全call／launcher／実運用承認接続の追加準備。これも科学実行とは区別する。
ユーザーが最終SOURCE・input resolution・grid・全scope・grant／STOPを確認して別途明示承認した場合のみ、第一段階へ進める。
C1最優先を維持し、通常のB進行を統括へ戻すことは必須にしない。
今回の必要差分だけをcommit／pushし、remote SHA・公開内容一致を確認後 **mandatory STOP**。YP8m8優位・科学的GO・同精度総資源改善は認定しない。
