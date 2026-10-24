# B2-D Stage 1：一括実行準備の最終技術報告

2026-10-25 JST。**技術準備完了、実行未承認。** 新規科学計算、Hamiltonian／state／operator生成、内部Gaussian回路、科学backend実行、正式grant発行・claim・消費は0件。
起点 `a9b53854b283b7e84f7ffe5ba7284ab004c3fc57`、専用branch `codex/research-b-b2d-execution-ready-20261025`。
最終科学SOURCE候補 **`348e08cbc44fb0a81cfc4b6932914f7b146a5b9c`**。[最終契約・具体的なユーザー操作](CONTRACT.md)を一件に集約した。

## 完成事項・残る境界

- 手動承認接続：外部chat承認APIは参照可能な既存SOURCE／toolに存在を確認できなかった。ユーザーが実承認を照合してroot所有recordを発行する最小方式を実装。コード生成・packet hash・自己申告authorityを実承認にしない。
- 環境binding：既存のCPython 3.11.9 venvを変更せず再利用。全50 packageのversion／RECORD、interpreter、venv cfg、import元・環境変数を固定。依存requirements整合も確認。
- Scope／launcher：36行・941 dispatch・448 exponential key、全入力scopeを維持し、published規約の必須独立照合4回を追加。全correctnessの後だけ比較へ進む。
- 最終proposal：SOURCE closure122ファイルとscope／環境／出力を含む未承認packetを実際に生成。採択後contract digestを計算して承認文面・発行commandを完成。
- 実行を阻む残る操作はユーザーの最終承認と、ユーザー／管理者によるprotected receiptの発行。権限が無ければ管理者へのこの一操作の依頼が最小解消方法。Codexが保護を回避して発行しない。

独立した技術的blockerは残していないが、手動記録発行自体は未実施。手動方式はchat identityのプログラム的真正性証明ではなく、ユーザー確認とOS権限に依存する。
実科学動作・入力identity・非可換性・A4数値同値性・complex128識別力は未検証であり、準備完了から推定しない。

## 再利用SOURCEと必要差分

既存のInputRecipe／loader、5arm DAG、DenseBackend、finite RTE／normalization、scope実績、失敗保存、packet／marker／atomic claimを再利用した。
旧adapter、published係数、trotterlibの科学関数、既存raw・契約・grant・ledger・marker・STOPは変更していない。

| 追加・拡張 | 理由と範囲 |
|---|---|
| `manual_authority.py`／`issue_b2d_stage1_approval.py` | 旧packetのhash一致だけでは独立した実承認にならなかったため、ユーザーの明示的な管理者発行を記録する最小glue。大規模認証・新汎用grant DB・暗号鍵基盤は作らない |
| `environment.py` | 旧preflightの必須version条件を実在環境へ接続。既存authorizationのhash／canonicalを再利用し、科学依存はimportしない |
| `prepare_b2d_stage1.py` | clean固定SOURCEから未承認packetを一回作るmetadata入口。正式grantは発行しない |
| `_published_reference.py` | 実取得した固定upstreamの3 full列constructorをbody変更なしで移植。既存係数generatorを再利用し、独立した数学的full積をA4内で照合 |
| preflight／launcher／scope | manual provider、環境binding、追加gate実績を接続。fresh output親の作成は将来の承認確認後だけ。root publisher用Git安全指定はcommand単位としglobal設定を変更しない |

固定upstream `20d64c2fea1ed2d2188777df4f3d962ee94df486` のproduct_formula.py（blob `70d68691a1dcdedea5447aac4b8bbfa51141d4cb`）と旧validatorをGitHubから実取得した。
移植3関数のAST hashが取得SOURCEと一致することを確認。旧validatorの `unitary=gate @ unitary` と数学的list規約を区別し、legacy append operatorとの同値性は主張しない。
独立gate失敗時はpublished法の性能比較へ進まない。今回数値gate自体は実行していない。
共通運用指示2026-10-23版は実参照し、既読hash `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` と一致した。

## 実環境・未承認packetの確認

[environment.json](environment.json)はCPython 3.11.9、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Nature0.7.2、OpenFermion1.6.1、PySCF2.7.0等50 distributionの実metadataを記録する。
科学packageをimportせず、importlib.metadata／find_specでversionとimport元を確認し、純粋なpackagingのrequirements照合で整合を確認。環境を再照合して不変だった。
既存venv・他研究環境は変更せず、-I -Bを用いる。CPU32、RAM約62 GiB（available約52 GiB）、GPU driver利用不可。恣意的なworker／CPU／RAM／wall-time上限や共有設定変更なし。

実行checkoutは `/tmp/prt-b2d-stage1-science-20261025` にcleanな固定commitを用意しただけである。
未承認packet `/tmp/prt-b2d-stage1-draft-20261025` は [contract.proposal.json](contract.proposal.json) とsampleのみ。
正式採択後SHA-256は `26aaf07cf0e74eb60263d01a0bfa91ccc1a3c623d8795a18299aaaa3ac80ff6a`。
科学artifact directory、正式packet、protected approval directoryが未作成であることを確認した。正式発行scriptは一度も実行していない。

## 人工テスト

固定実行interpreterの `-I -B` と科学import barrierを用いた。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_b2d_adapter.py
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_b2d_evaluator.py
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_b2d_input_preflight.py
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_b2d_stage1_launcher.py
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B tests/management/test_b2d_execution_ready.py
```

**既存62＋追加12＝74件PASS。** 旧testの「providerがNone」という仮定だけを「登録manual providerはあるが記録なし／任意authorityではclaim不可」へ更新し、既存の拒否条件を緩和していない。
新testはdraft／採択bytesの区別、非管理者発行拒否、agent所有・symlink receipt拒否、承認／SOURCE／scope／環境／出力／packet pathのbinding、記録欠落、環境不一致、固定移植ASTを確認。
追加published gate4件の先行・失敗時比較禁止・STOP、既存atomic claimのO_EXCL拒否をmockで確認した。
人工packet・markerは隔離tmpのmockのみ。実grant／claimはない。mock backendのexponential／signalはlabel処理であり、実配列・state・unitary・科学operatorを生成していない。
science backendの実動作を人工PASSだけで実証したとは扱わない。

## 次の担当・STOP

研究B／ユーザーが[最終契約](CONTRACT.md)の科学SOURCE・入力・scope・環境・手動承認方式・grant／STOPを確認し、実承認と保護記録発行を行う。
その後、別途明示されたCodex実行指示で固定Stage 1だけを一回実行する。今回の一括準備を科学実行承認に変換しない。
C1最優先・他研究非干渉を維持。必要差分だけをcommit・pushし、remote SHA・公開内容一致を確認後mandatory STOP。
