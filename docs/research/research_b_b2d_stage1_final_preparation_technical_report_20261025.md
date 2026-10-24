# B2-D Stage 1最終実行準備：技術報告

2026-10-25 JST。準備のみ。科学計算・Hamiltonian／state生成・内部Gaussian回路・科学backend生成・grant発行／claim／消費なし。
起点 `94394357d503b7326378dd62f647487199593dd5`、専用branch `codex/research-b-b2d-final-preparation-20261025`。
科学SOURCE候補は **`97a449ad53c74361ffc5c4cffd6de5f3f0f0a72e`**。この技術報告と[最終提案](research_b_b2d_stage1_final_execution_proposal_20261025.md)は後続の文書commitで公開する。

## 完成と未完成

| 対象 | 状態 |
|---|---|
| 全scope定義・実績照合 | 実装・人工確認完了。科学的境界と合法runtime recipe展開を固定。全BLAS演算の個数を総科学call数とする方式ではない |
| Launcher・失敗保存 | 実装・mock確認完了。実入力／科学backendは未起動 |
| Production authority | **未接続**。既存の実承認取得providerを確認できず、架空のauthorityは作っていない |
| 実行契約 | 未採択・全体として未完成。実行可能とは報告しない。現CLIはclaim前に拒否 |
| 科学資格 | N=1適格性、非可換性、H identity、A4数値correctness・旧validator同値性、complex128識別力は未確認 |

## 再利用と必要な追加

既存のDAG／A0–A4列、InputRecipe／loader、`PreparedDenseInputs`、`DenseBackend._validate`、RTE moments・normalization、approval packet／atomic claimを再利用した。
旧adapter、移植済みPF係数、既存trotterlib科学関数、raw結果・旧契約・grant・ledger・marker・STOPは変更していない。

- [`scope.py`](../../src/trottertracks/b2d_evaluator/scope.py)：旧scope案の未登録内部演算を、今回指定された科学的処理境界へ整理。固定recipeのruntime domain、explicit expm key、DAG node／call回数と各入力resolutionを照合する。旧scope schemaも保全。
- [`launcher.py`](../../src/trottertracks/b2d_evaluator/launcher.py)／[`run_b2d_stage1.py`](../../scripts/run_b2d_stage1.py)：既存コードに無かった五方式評価の実行・保存glue。preflight→既存atomic claim→一回入力生成→resolved graph→全gate先行→比較→保存／STOPを接続。`--inspect` はmetadataのみ。
- [`errors.py`](../../src/trottertracks/b2d_evaluator/errors.py)：秘密の例外本文を出さず、入力・sector・H identity・A4・precision等の失敗codeを保存するための最小共有例外。
- 既存input／backend／evaluator／preflightを最小拡張し、境界の実績と途中のgate・operator／signal診断を保存する。科学的数値式・PF係数・比較grid・normalizationは変更しない。

新しいHamiltonian計算法、Gaussian合成、sampler、PF探索、grant発行機構を再実装していない。
共通運用指示2026-10-23版のhashは既読版 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` と一致した。

## Scope・Gaussian内部経路の静的確認

`prepare_df_partial_s2` はone-body Gaussian生成1回、`build_df_blocks` で元3因子のGaussian生成、tail抽出で残差1因子の再生成を行う。
`U_to_qiskit_ops_jw` はqiskit-natureのBogoliubovTransformを再利用し、既存basis fingerprint処理は局所Operatorも使う。
`_normalized_symbolic_tail_in_sector` はretained成分をbasisごとにまとめ、`basis_change_unitary` 内部のQuantumCircuit／Operatorとsector bit変換・leakageを使用する。
これらは将来Stage 1の必須入力処理としてscopeへ含めた。今回生成していない。qiskit-natureを含む実行環境は正式に固定する必要がある。

[scope JSON](research_b_b2d_stage1_scope_20261025.json)には10回の入力処理境界、941 DAG dispatch、448 backend unique expm key、RTE config／moments各2、signal expectation144を登録した。
親子が重複するvectorであり、合算した全job科学call数を主張しない。
その他のGaussian diagonalization／gate表現、sector matvec、norm、Taylor／power等は固定関数scope内へ含め、sourceと入力domainを固定する。
選択factor、retained成分1〜3／basis1、operation数はruntime receiptに記録。違法展開、未登録key、重複、回数超過、correctness前の比較、完了不足を拒否する。

## Production authorityの調査結果

`pr_dictionary_validation.authorization`、`rectangular_preflight`、`f2_preflight`、`pr_dictionary_binding.contract` には、packet内の承認本文／hash／SOURCE／未使用markerの照合形式がある。
参照可能なSOURCE、ローカル運用領域のprovider定義、接続ツールのmetadataからは、packetとは独立した実承認取得API・trusted record元・production verifier実装を確認できなかった。
既存の実grant／承認本文を探索・流用したり、C1の運用へ接続したりしていない。

従ってproviderはNone。callerの任意authorityを非mockと宣言しても、現launcher／非mock preflightには入れない。
mockはread-only確認だけを許可し、既存claim前に拒否する。実providerの取得元・照合方法をユーザー／既存運用担当者が指定した後に接続する必要がある。
これは未完成部分であり、packetのhash一致や本依頼を科学実行承認に変換していない。

## 保存・失敗・STOP

新規出力の9ファイルに、binding、raw input配列とprovenance、resolved graph、境界event、途中／全結果、status、STOPを保存する実装を準備した。
入力生成の途中でも取得済みresolution候補とcall失敗を記録する。A4失敗時はoperator／signal比較を始めず、比較途中の例外では完了行と途中診断を保存する。
承認前の拒否は安全なcodeを返し、承認本文・秘密の例外本文・他研究の記録を露出しない。
既存atomic claimのmkdir競合では相手のoutputへ書き込まず、markerを消さずSTOP。保存エラーも完了成功扱いにしない。
一回実行、retry／resumeなし。SOURCE・fixture・sector・grid・precisionを自動変更しない。

## 人工確認

CPython 3.11.0rc1、`-S`、stdlibのみ、科学import barrierを使用した。

```bash
/usr/bin/python3.11 -S tests/management/test_b2d_adapter.py
/usr/bin/python3.11 -S tests/management/test_b2d_evaluator.py
/usr/bin/python3.11 -S tests/management/test_b2d_input_preflight.py
/usr/bin/python3.11 -S tests/management/test_b2d_stage1_launcher.py
```

**15＋14＋17＋追加16＝62件PASS。** 新testはscienceのmatrix／state／unitaryを作らず、opaque label・固定SOURCE descriptor・mock receiptだけを使用する。
実atomic claimは呼んでいない。claim位置にはspyを置き、mock authorityが到達しないことを検査した。
隔離TemporaryDirectoryのmock pipelineでは36行・941 dispatch・448 expm descriptor・144 signal labelと保存schemaを確認した。数値expm／signal評価ではない。
SOURCE／scope／runtime mismatch、runtime domain、call超過、全A4 gate先行、input／A4失敗時の比較禁止、途中2行の保存、precision INCONCLUSIVE、秘密の例外本文の非露出、保存失敗時STOP、再実行による既存fileの非上書きを検査した。
`--inspect` もmetadataのみで動作確認。科学import／科学backend生成はなし。

利用可能資源は32 CPU、RAM約62 GiB（available約52 GiB）、GPU driver利用不可。
4つの独立した軽量test processを並列実行。CPU・RAM・worker数・wall timeへ人工的な上限や共有環境変更を加えていない。新しい速度倍率・科学性能は主張しない。

## 次の担当・mandatory STOP

研究Bチャット／ユーザーへ、[最終提案の採択項目](research_b_b2d_stage1_final_execution_proposal_20261025.md)と実承認provider取得元の未接続を返す。
次のCodex作業として提案するのは、指定された既存providerへの接続と環境binding、必要なSOURCE再固定まで。科学計算は別途明示承認後のみ。
C1最優先・既存SOURCE／結果／承認／STOPを保全し、公開SHA・内容照合後mandatory STOP。通常のB進行で統括報告は必須にしない。
