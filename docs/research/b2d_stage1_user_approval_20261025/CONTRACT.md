# B2-D Stage 1：通常ユーザー方式の最終科学実行契約案

2026-10-25 JST。**未承認。今回完了したのは技術準備だけで、実承認記録・正式grantの発行、claim、科学計算は未実施。**
本契約は承認方式だけを簡素化する。旧SOURCE・契約・STOPは履歴として保全する。

## 固定対象

| 対象 | 固定値 |
|---|---|
| Repository／準備起点 | HIROMU1015/Partially-Randomized-Trotter／`ceb4c50a23c57743343e66cfc6aba69a9a4df3b8` |
| 新科学SOURCE | **`91f45aeb50275bb6d22050a304d0a9b1ff025d70`**。文書公開commitと区別する |
| 新契約原本 | [contract.proposal.json](contract.proposal.json)、`UNAPPROVED_TEMPLATE`、SOURCE closure122ファイル |
| 正式採択後のcontract SHA-256 | **`155728667abd5a663b44bc06b4e596ea6d64d529da6b02ba1be15ee8c050d4fb`**。`adoption_bytes` がproposal_statusを除きstatusをUSER_ADOPTED_STAGE1へ変えたbytes。未発行の将来contractのdigestであり、採択済みではない |
| 固定checkout | `/tmp/prt-b2d-stage1-user-science-20261025`、新SOURCEのclean detached worktree |
| 未承認draft | `/tmp/prt-b2d-stage1-user-draft-20261025`。proposalと未承認sampleのみ、grantなし |
| 承認方式／所有UID | `MANUAL_SAME_USER_OPERATIONAL_APPROVAL_V1`／1001（abe） |
| ユーザー専用記録先 | `/home/abe/.prt-b2d-stage1-approvals`。今回未作成。directory0700・file0600 |
| 正式packet予定先 | `/home/abe/.prt-b2d-stage1-approvals/stage1-user-20261025`。今回未作成 |
| Fresh科学出力 | `/tmp/prt-b2d-stage1-user-science-20261025/artifacts/research_b_b2d_stage1/20261025/user-approval-once`。科学専用親directoryから未作成 |
| Recipe fingerprint | `7538b7467849add337293a1719d32c5c963e28f7823bec7bedc5e924006ef3e9` |
| Graph layout | `7e6c5bb3822e743c0d5e62153b1cbb574508365a00b8d50c9fe757e6734fad5e` |

旧科学SOURCE `348e08cbc44fb0a81cfc4b6932914f7b146a5b9c`、旧digest `26aaf07cf0e74eb60263d01a0bfa91ccc1a3c623d8795a18299aaaa3ac80ff6a` は今回の実行対象ではない。
[旧契約](../b2d_stage1_execution_ready_20261025/CONTRACT.md)の管理者発行要件は、この新しい未承認提案では通常ユーザーの手動操作へ置き換える。旧文書を遡って書き換えない。
SOURCE closureは全src Python、fixture、prepare／issue／runの3入口、既存環境JSONを含む。SOURCE・契約・scope・環境・出力・承認方式を変える場合は、新たに固定して明示承認を受ける。

## 科学条件と未検証事項

| 項目 | 固定条件 |
|---|---|
| モデル／fixture | Synthetic 2-mode one-body＋3 DF因子。分子geometry／AO basis／分子DF rank探索は非該当。`tests/test_df_partial_s2_repeated.py::_nontrivial_basis_case` のliteral、blob `0c92ae9c1e8c5721d9c8629a83435b855f5d6043` |
| Sector／状態 | N=1完全粒子数sector、sorted OpenFermion JW順、mode0 occupationの同一physical状態 |
| Split | L_D=2、lambda_frobenius_squaredのweight-ranked prefix。one-body→選択DF順 |
| 5方式 | A0 partial-S2、A1 standalone D-S2、A2 Yoshida4、A3 published Morales Y8m10b、A4 published YP8m8 |
| Grid | δ=±0.09、q=2,3。A1〜A4のn_D=1,2。A0は4行、他各8行、合計36行 |
| RTE／scalar | r_RTE=2、K=2、analytic momentsのみ。seed41はmetadata、samplingなし、truncation_tolerance metadata=1.0。extract_identity_phase、coefficient_atol=0 |
| 精度／指標 | complex128、correctness絶対atol=1e-12。直接複素coherent signal誤差、X/Y・半径。phase diagnostics無効、radius_floor／branch_margin=None |
| 評価 | descriptive_only_no_scientific_go。科学的改善や最終総資源優位を自動認定しない |

実入力identity、選択因子・残差、N=1での非可換性、operator同値性、A4 correctness、complex128の識別力は未検証。
claim後、固定recipeを一回だけ生成してarray hash・basis順・state・resolved graphを記録する。未生成hashを架空値で埋めない。
不適格・不一致・精度不足なら保存・STOPする。別sector／fixture／状態／精度／追加時間点へ自動変更しない。

## Scope・correctness・保存

契約JSONのcall_scope全文を採択対象とする。旧[scope.json](../b2d_stage1_execution_ready_20261025/scope.json)と同一で、重複コピーは作らない。
親子処理のbudget vectorであり、各境界の回数を足して総科学call数とは呼ばない。

- 入力境界10回：Hamiltonian1、split1、preparation1、sector1、full H1、D block3、retained tail1、occupation state1。
- preparation内のone-body Gaussian1、全DF Gaussian3、残差DF Gaussian1、既存fingerprintのOperator処理を含む。retained tail内のbasis QuantumCircuit／Operator・sector変換・leakageも含む。
- Runtimeは2 modes、N=1、D3、選択2／残差1、retained非identity成分1〜3／basis1。Z0／Z1／Z0Z1の合法な展開に限定し、実成分・basis hash・operation数を記録する。
- 共通DAGは36行・941 dispatch。共有cacheの実exponentialは登録した448 unique key。RTE config／moments各2、signal expectation144、published.convention4。phase／samplingなし。
- Input identity・sector／state norm・Hermiticity・H=D+R+cI → residualとD内部／D-R非可換性 → 独立数学的A4 `P K^n P†`／actual inverse全4件 → published full数学積全4件 → 全gate通過後だけ固定比較、の順序を維持する。
- Published係数・列constructorの固定起点は `HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae` の `20d64c2fea1ed2d2188777df4f3d962ee94df486`。数学的列とappend順を区別し、旧validatorの直接append operatorとの同値性は主張しない。詳細な規約説明は[旧契約のscope節](../b2d_stage1_execution_ready_20261025/CONTRACT.md#3-全scopeとcorrectness順序)を継承する。
- 未登録・重複・超過・完了不足はSTOP。完走時はscope実績と全resolutionを照合する。

既存の9ファイル（binding、input provenance／配列、resolved graph、events、結果、status、STOP）と失敗保存を維持する。
最大3134 event、record64 KiB、JSON1 MiB、総出力16 MiB、STOP reserve256 KiB。途中失敗でも取得済み情報と理由codeを保存する。
Full controlled Hadamard wrapper、native build／compile、processor価格、shot認証、全RPE総資源評価は対象外。

## 環境と資源

[既存environment.json](../b2d_stage1_execution_ready_20261025/environment.json)と契約内environment_bindingを一致させて再利用する。新旧間で同一であることをmetadataのみで確認済み。
CPython **3.11.9 final**、実行interpreter `/tmp/pr-phase2df1pre-private-20261023/venv/bin/python`、常に `-I -B`。
50 distribution全version／RECORD、interpreter bytes、venv cfg、prefix、科学package import元hash、requirements整合を固定する。
NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Aer0.15.1、Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0、Numba0.60.0、Matplotlib3.9.2。
PYTHONPATH／PYTHONHOME／LD_PRELOAD／LD_LIBRARY_PATH／OMP_NUM_THREADS／OPENBLAS_NUM_THREADS／MKL_NUM_THREADS／NUMBA_NUM_THREADSは未設定の状態にbindingする。
環境変更時は拒否し、依存versionを自動更新しない。metadata照合は科学backendの動作保証ではない。

確認時は32 CPU、RAM約62 GiB（available約52 GiB）、GPU driver利用不可。既存2×2 sector・vectorized処理・共有cacheを再利用する。
CPU／RAM／worker／phase・total wall-timeへの人工上限を追加しない。実行前に空き資源を再確認し、共有環境・他ユーザー・C1へ干渉しない。

## 承認方式と、その限界

実行承認は**ユーザーが研究Bチャットで行う実際の明示承認**である。ユーザーがその原文を保存し、通常権限で手動発行する。
独立した外部承認API、root所有、特権操作、署名基盤は要求しない。
承認原文hashと参照、SOURCE／closure、recipe／layout、scope／環境、fresh output、packet絶対path、grant digest、one-shot／STOPをreceiptで束縛する。
専用directory0700・file0600は他OSユーザーのアクセスを制限する。**同じOSアカウントのユーザーとCodexを暗号学的に識別しない。**
未標識の偽承認文が本人由来かをプログラムだけで判定できるとは主張しない。実際のチャット承認と別途実行指示は、Codexが会話上でも確認する。
Codexによる実承認文の生成・代行・正式発行、sampleの実承認扱いは禁止する。既知sample markerと空文は発行時に拒否するが、本人性の認証機構ではない。

未承認・policy／SOURCE／契約／scope／環境／出力の不一致・使用済みgrantは拒否する。旧root方式のgrantを新SOURCEの承認に流用しない。
既存`claim_packet`のatomic exclusive-create、fresh output、一回実行、retry／resume禁止、marker削除禁止、完了・失敗ともmandatory STOPを維持する。
発行途中の失敗でも部分packet／receiptを削除・上書きして再試行しない。状態を保全して停止する。

## ユーザーが次に行う操作

**A. 実承認と手動発行。今回は未実施。**

1. 研究Bチャットで本契約の新SOURCE・digest・scope・環境・fresh output・one-shot／STOPを明示承認する。
2. ユーザー自身の端末で次を実行し、`cat`の入力へ**その実承認メッセージ原文**を貼り、Ctrl-Dで保存する。sampleは使わない。参照には実メッセージの日時またはIDを入力する。

```bash
umask 077
mkdir -m 700 /home/abe/.prt-b2d-stage1-approvals
cat > /home/abe/.prt-b2d-stage1-approvals/actual-user-approval.txt
read -r -p '実承認メッセージの日時またはID: ' B2D_APPROVAL_REFERENCE
```

3. 同じ端末で次を実行する。**正式発行のみで、claim・科学計算は始めない。**

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-b2d-stage1-user-science-20261025/scripts/issue_b2d_stage1_approval.py \
  --draft /tmp/prt-b2d-stage1-user-draft-20261025 \
  --packet /home/abe/.prt-b2d-stage1-approvals/stage1-user-20261025 \
  --approval-file /home/abe/.prt-b2d-stage1-approvals/actual-user-approval.txt \
  --approval-reference "$B2D_APPROVAL_REFERENCE" \
  --confirm-contract 155728667abd5a663b44bc06b4e596ea6d64d529da6b02ba1be15ee8c050d4fb \
  --acknowledge-manual-user-review
```

**B. Codexへの別途科学実行指示。Aの完了後だけ行う。**

発行結果のpacket pathとgrant IDを研究Bチャットで伝え、新SOURCE `91f45aeb50275bb6d22050a304d0a9b1ff025d70`・上記digestの固定Stage 1を一回実行するよう明示指示する。
Codexは実承認原文・参照・全binding・未使用grant・資源を確認した後だけ、次を実行する。今回の準備依頼を実行指示に代用しない。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-b2d-stage1-user-science-20261025/scripts/run_b2d_stage1.py \
  --execute --packet /home/abe/.prt-b2d-stage1-approvals/stage1-user-20261025 \
  --output /tmp/prt-b2d-stage1-user-science-20261025/artifacts/research_b_b2d_stage1/20261025/user-approval-once
```

技術準備の独立したblockerは残していない。残るのはユーザーの実承認・手動発行・別途実行指示と、実行時に初めて評価する科学correctnessである。
担当：ユーザーがAとBの指示、Codexがその後の固定科学検証・保存・STOP。C1最優先を維持し、通常のB進行で統括報告は不要。**今回の公開確認後mandatory STOP。**
