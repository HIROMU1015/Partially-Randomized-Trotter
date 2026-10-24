# B2-D Stage 1：簡素化した科学実行契約・技術報告

2026-10-25 JST。**実装・人工確認のみ完了。今回、科学計算は実行していない。**
次にユーザーが研究Bチャットで明示的に科学実行を許可した場合だけ、Codexが本条件で一回実行し、結果を保存してmandatory STOPする。
承認原文の保存・承認ID・証明書・grant発行・packet・receipt・端末での事前操作は要求しない。許可は会話上の運用規則であり、プログラムで証明しない。

## 固定SOURCEと単一実行コマンド

| 項目 | 固定値 |
|---|---|
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| 作業起点 | `429ac59658b69252fe212501124cd6aa294da8f5` |
| 新科学SOURCE候補 | **`54008c697e686a42ec48941d25219d523bafd454`** |
| 科学checkout | `/tmp/prt-b2d-stage1-simple-science-20261025`、上記SOURCEのclean detached worktree |
| Fresh output | `/tmp/prt-b2d-stage1-simple-science-20261025/artifacts/research_b_b2d_stage1/20261025/simple-once`、今回未作成 |
| 実行入口 | `scripts/run_b2d_stage1.py --execute --source-commit … --output …` |

**ユーザーの明示的な科学実行許可後に、Codexが使うコマンド：**

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-b2d-stage1-simple-science-20261025/scripts/run_b2d_stage1.py \
  --execute --source-commit 54008c697e686a42ec48941d25219d523bafd454 \
  --output /tmp/prt-b2d-stage1-simple-science-20261025/artifacts/research_b_b2d_stage1/20261025/simple-once
```

`--execute`は明示的な起動指定であり、ユーザー許可の証明ではない。mode指定なしは引数エラー、`--inspect`はmetadata表示だけ。
`--source-commit`は計算SOURCEの整合確認に使う。実行checkoutのHEAD不一致・dirty SOURCEは拒否する。別の承認用契約ファイルやdigest入力はない。
旧科学SOURCE `91f45aeb50275bb6d22050a304d0a9b1ff025d70` と旧契約群は履歴として保全し、今回の将来実行条件だけを変更する。

## 科学条件

| 項目 | 固定条件 |
|---|---|
| Fixture／モデル | 既存 `tests/test_df_partial_s2_repeated.py::_nontrivial_basis_case` のliteral。Synthetic 2-mode one-body＋3 DF因子。分子geometry／AO basis／分子DF rank探索は非該当 |
| Sector／state | N=1完全粒子数sector、sorted OpenFermion JW順、mode0 occupationの共通physical状態 |
| D/R分割 | L_D=2、lambda_frobenius_squaredのweight-ranked prefix。one-body→選択DF順 |
| 比較方式 | A0 partial-S2、A1 standalone D-S2、A2 Yoshida4、A3 published Morales Y8m10b、A4 published YP8m8 |
| Grid | δ=±0.09、q=2,3。A1〜A4はn_D=1,2。A0は4行、他各8行、計36行 |
| RTE／normalization | r_RTE=2、K=2、既存finite RTE／analytic moments／retained symbolic Pauli L1 normalization。seed41はmetadata、samplingなし、truncation_tolerance metadata=1.0 |
| Scalar／精度 | extract_identity_phase、coefficient_atol=0、complex128、correctness絶対atol=1e-12 |
| 指標／判定 | Operator誤差、X/Y・半径・複素coherent signal直接誤差。phase diagnostics無効、radius_floor／branch_margin=None。記述的診断のみ、自動的な科学的GOや総資源優位の認定なし |

未生成の入力identityを架空値で埋めない。固定recipeによる一回の入力準備で、Hamiltonian・preparation・sector・state・array・選択因子・basis順を記録する。
入力identity、N=1での非可換性、A4の数値的正しさ、complex128の識別力は未検証。科学実行時の不一致・不適格・精度不足では保存・STOPし、別sector／state／fixture／精度／追加時間点へ自動変更しない。

## 科学scope・correctness・再現性

新SOURCEの `scope.registry(candidate_graph(source_commit))` が固定scopeを与える。承認情報を含まない科学的な境界・実績の構造である。
従来scopeとの差は入力生成開始の説明ラベルだけで、科学的条件・回数・budgetは不変。

- 共通DAGは36行・941 dispatch、共有exponential cacheは448 unique key。親子境界の回数を合計して全jobの科学call総数とは呼ばない。
- 入力境界10回：Hamiltonian1、split1、preparation1、sector1、full H1、D block3、retained tail1、occupation state1。
- 内部one-body Gaussian1・全DF Gaussian3・残差DF Gaussian1、basis QuantumCircuit／Operator、sector変換・leakage等を含む既存Input packageをそのまま利用する。
- Runtime domainは2 modes・N=1・D3・選択2／残差1、retained非identity成分1〜3／basis1。実成分・basis hash・operation数を科学provenanceとして記録する。
- RTE config／moments各2、signal expectation144、published.convention4。未登録・重複・超過・完了不足はSTOP。
- Input identity・sector／state norm・Hermiticity・H=D+R+cI → residualとD内部／D-R非可換性 → 独立A4数学積 `P K^n P†`／actual inverse全4件 → published full数学積全4件 → 全gate通過後だけ固定比較、の順序を維持する。
- Published係数・列constructorの起点は `HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae` の `20d64c2fea1ed2d2188777df4f3d962ee94df486`。数学的積とappend順を区別し、旧validatorの直接append operatorとの同値性は主張しない。

既存CPython3.11.9環境を変更せず再利用する。必須科学依存9種のversionとPython runtimeを確認し、実行時の全installed version（今回50種）・interpreter／import元・環境変数をprovenanceへ記録する。
必須versionはNumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0、Numba0.60.0、Matplotlib3.9.2。
既存environment JSONからPython／必須versionだけを再利用し、RECORD・interpreter bytes・venv cfg・設置先・thread環境変数等のbitwise一致は要求しない。依存versionを自動更新しない。

確認時は32 CPU、RAM約62 GiB（available約52 GiB）、GPU driver利用不可。既存の小さいsector・vectorized処理・共有cacheを利用し、CPU／RAM／worker／wall-timeへ人工上限を追加しない。
実行前に実資源を再確認し、C1最優先・共有環境／他ユーザー非干渉を維持する。

## Fresh output・失敗保存・STOP

科学preflightは読み取りだけ。通過後にoutputを排他的な`mkdir(exist_ok=False)`で新規作成し、衝突時は既存内容を書き換えない。消費grantや別markerは作らない。
一回のrun内で入力生成と評価の再入を拒否し、実行scopeを記録する。別namespaceでの再実行を含むretry／resumeはCodexの運用として禁止する。

保存対象は既存の9ファイル：`binding.json`、`inputs.json`、`inputs.npz`、`graph.json`、`events.jsonl`、`partial_results.json`、`result.json`、`status.json`、`STOP.json`。
bindingは科学SOURCE・recipe・scope・実環境・出力先を保持し、承認情報を含まない。全gate・実績、途中までのrow、失敗理由codeを保存する。
最大3134 event、record64 KiB、JSON1 MiB、総出力16 MiB、STOP reserve256 KiBを維持する。保存失敗時もbest-effortでSTOPを書き、完走扱いにしない。
正常終了・不適格・correctness失敗・途中例外ともmandatory STOP。Full controlled wrapper、native build／compile、PF係数探索、processor価格、shot認証、全RPE総資源評価は対象外。

## 撤去・維持した処理と人工確認

| 分類 | 処理と今回の扱い |
|---|---|
| A：不要な承認管理 | manual authority／receipt、承認本文保存・hash・参照、OS本人確認、grant／packet発行・照合・消費、承認用SOURCE closure一覧、承認証明guardを撤去。`manual_authority.py`、`issue_b2d_stage1_approval.py`、`prepare_b2d_stage1.py` を削除 |
| B：科学・実行管理 | 既存launcher・DAG・Input package・数値backendを再利用。科学SOURCE／fixture／identity／grid／scope・gate・provenance・fresh output・失敗保存・retry禁止・STOPを維持 |

`ExecutionApproval`とpacket前提の引数を削除した。run contextは入力identity・scope・一回の処理状態だけを保持し、承認証明に使わない。
Input packageの`preparation_receipt`は選択因子・basis等の**科学provenance**であり、承認receiptではないため維持した。
他研究の共通 `pr_dictionary_validation/authorization.py`、AGENTS.md、共通運用指示、旧契約／結果／grant／marker／STOPは変更していない。

| 人工test | 件数 | 結果 |
|---|---:|---|
| test_b2d_adapter.py | 15 | PASS |
| test_b2d_evaluator.py | 14 | PASS |
| test_b2d_input_preflight.py | 15 | PASS |
| test_b2d_stage1_launcher.py | 15 | PASS |
| test_b2d_execution_ready.py | 4 | PASS |
| test_b2d_simple_execution.py | 7 | PASS |
| 合計 | **70** | **PASS** |

旧80件から不要な承認専用testを整理し、科学的なtestを再利用・適合更新した63件と、簡素化入口のfocused test7件を実施。
各ファイルを固定venvの`python -I -B tests/management/<filename>`で独立並列実行し、科学import barrier・stub loader・mock backendを使用した。
明示指定なしの起動拒否、packetなしmock完走、SOURCE／grid／scope／依存version不一致、fresh output衝突、correctness先行、部分保存・失敗STOP、再入拒否を確認した。
PF列・係数・fixture・共通authorizationのbytes、数値backendの計算methodとDAG builderのASTが旧SOURCEと同じであることも確認した。
新しいclean科学checkoutでread-only preflightも通過し、科学packageをimportせず、科学出力directoryも生成していない。
人工PASSとmetadata確認は、実Hamiltonian・operator・signal・Gaussian回路の動作や科学的性能を検証した結果ではない。

次の担当は、**ユーザーが研究Bチャットで科学実行を明示許可し、Codexが上記単一コマンドで固定Stage 1を一回実行・保存・STOPする**。追加の端末操作や承認証明資料は不要。
今回は専用branchを公開・remote内容照合後、**mandatory STOP**する。
