# Track B / G1：固定Decision Packetの明示的one-shot実行

## 1. 実行認可

以下の固定sourceに対し、G1 decision packetを一回だけ実行してください。

**固定source S：**

`718cf6c1abb50c0398028ab39c24c994a4de2bd3`

**実行認可文（source-bound statement）：**

source `718cf6c1abb50c0398028ab39c24c994a4de2bd3` のG1固定契約で、decision packetを一回だけ実行し、終了後はmandatory STOPしてください。

この指示は、上記sourceとその固定契約に対する明示的なone-shot実行認可です。

認可範囲は次の3段階のみです。

1. Phase A：固定P₃の6頂点構造の独立数学監査
2. Phase B1：固定8人工LPのrational echo-only verification
3. Phase B2：固定8人工LPのexact solveと独立Fraction verification

Phase AがPASSした場合だけPhase B1へ進み、Phase B1の全8件がPASSした場合だけPhase B2へ進んでください。

**最初の失敗・反例・技術的未判定で直ちにSTOPし、残りの段階へ進まないでください。**

Retryは0回です。

---

## 2. リポジトリ・固定source

Repository：

`HIROMU1015/Partially-Randomized-Trotter`

Branch：

`track-b-g1-source-preparation-20261009`

Source full SHA：

`718cf6c1abb50c0398028ab39c24c994a4de2bd3`

必ず確認する資料：

- `docs/tracks/algorithm_codesign/g1_source_review_20261009.md`
- `artifacts/track_b_g1_source_preparation/2026-10-09/source_manifest_v1.json`
- `artifacts/track_b_g1_result_prior_preparation/2026-10-09/decision_packet_contract_v1.json`
- `artifacts/track_b_g1_result_prior_preparation/2026-10-09/structure_audit_contract_v1.json`
- `artifacts/track_b_g1_result_prior_preparation/2026-10-09/fixture_manifest_v1.json`
- `artifacts/track_b_g1_result_prior_preparation/2026-10-09/runtime_identity_v1.json`

今回のsource review判定：

`G1_SOURCE_REVIEW_PASS_WITH_LIMITATIONS`

ただし、この判定は限定実行に進めるという意味であり、数学監査やbackend実証が成功したという意味ではありません。

---

## 3. 実行前preflight

既存sourceを変更せず、まず以下を確認してください。

- HEADが固定source Sと一致
- remote branchのSHAがSと一致
- worktree clean
- source manifestの全hash一致
- frozen contract・fixture・runtime identity一致
- 旧protected files不変
- 既存SoPlex binaryのSHA256一致
- guardとindependent verifierのSHA256一致
- G1 one-shot markerが未存在
- G1 STOP markerが未存在
- G1 result directoryが未存在

まずread-only source verificationを実施してください。

```bash
/usr/bin/python3 -B scripts/tracks/algorithm_codesign/run_g1_decision_packet.py \
  --source-commit 718cf6c1abb50c0398028ab39c24c994a4de2bd3 \
  --verify-source
```

この段階では、

- structure audit calls = 0
- backend invocations = 0
- LP calls = 0
- one-shot marker作成 = 0

を維持してください。

**Preflightで不一致があれば実行せずSTOPし、GPTへ報告してください。**

不一致を解消するためにsourceや契約を変更してはいけません。

---

# 4. Phase A：6頂点構造の独立数学監査

固定契約：

`structure_audit_contract_v1.json`

に定義されたA01–A10をすべて監査してください。

## 監査対象

固定P₃の7 prototype：

`O0, O2, P2, P3, A0, A1, A2`

に対する係数保存問題です。

\[
V\gamma=t
\]

を一般の \(x>0\) について解析します。

提案されたparameterization：

\[
\begin{aligned}
\gamma_{O0}&=s,\\
\gamma_{O2}&=b,\\
\gamma_{P2}&=\mu+(1-\mu)s-\mu r-b,\\
\gamma_{P3}&=1-r-b,\\
\gamma_{A0}&=\gamma_{A1}=1-s,\\
\gamma_{A2}&=r,
\end{aligned}
\]

ただし、

\[
\mu=\frac{x^2+2}{x^2+6}.
\]

非負性条件：

\[
\begin{aligned}
0&\le s\le1,\\
r,b&\ge0,\\
r+b&\le1,\\
b+\mu r&\le\mu+(1-\mu)s.
\end{aligned}
\]

### 必須証明事項

1. 元sourceの係数式から一般解を独立に導出すること。
2. 提案されたparameterizationとの一致。
3. 非負領域が有界な3次元多面体になること。
4. 6境界から選んだ全20組の三重交点の分類。
5. 6頂点の完全性。
6. ordinary / PTSC-K0 / A / J1 / J2 / J3 の各degree係数保存。
7. B2が `r=1-s, 0<=b<=s` に対応すること。
8. 頂点混合からprecision shareを復元できる条件。
9. Zero-mass groupで不正な除算が発生しないこと。
10. 理想係数classと数値K3・dyadic sampler classの違い。

補助的な有理数代入は固定契約のoff-domain点のみを使用し、一般証明の代わりにしないでください。

**GPTが提案した6頂点を、正しい前提としてコードに与えて結果を生成してはいけません。**

独立導出・完全性証明の結果と照合してください。

### Phase A分類

- `G1_STRUCTURE_PASS_WITH_DECLARED_LIMITS`
- `G1_STRUCTURE_COUNTEREXAMPLE`
- `G1_STRUCTURE_TECHNICAL_INCONCLUSIVE`

PASS以外ではbackendを呼び出さずSTOPしてください。

---

# 5. Phase B1：固定8入力のecho-only検証

Phase AがPASSした場合のみ実施します。

固定8入力について、それぞれ一回だけSoPlexへのrational入力・readbackを検証してください。

**この段階ではLPの最適化を行いません。**

確認事項：

- objective係数
- objective offset
- inequality matrixとrhs
- equality matrixとlhs/rhs
- variable lower/upper bounds
- exact Fraction identity
- canonical rational strings

全8件のecho-only verificationがPASSした場合だけ、Phase B2へ進んでください。

一件でも不一致があればSTOP。

Echo-only call cap：8回。

追加のroundtrip fixtureは実行しないでください。

---

# 6. Phase B2：固定8件の人工LP検証

固定した順序とexpected statusを変更しないでください。

| 順序 | Fixture | 変数数 | Expected |
|---|---|---:|---|
| 1 | `B2_inner` | 28 | OPTIMAL |
| 2 | `B2_outer` | 28 | OPTIMAL |
| 3 | `B2_infeasible` | 28 | INFEASIBLE |
| 4 | `B3_inner` | 22 | OPTIMAL |
| 5 | `B3_outer` | 22 | OPTIMAL |
| 6 | `B3_infeasible` | 22 | INFEASIBLE |
| 7 | `HP100_B2_inner` | 28 | OPTIMAL |
| 8 | `HP100_B3_infeasible` | 22 | INFEASIBLE |

各fixtureは最大一回だけsolveしてください。

### OPTIMALの場合

次を独立Fractionで認証してください。

- exact primal feasibility
- exact objective
- exact dual lower bound
- finite upper-bound correction
- dual multiplierの符号
- reduced cost identity
- primal-dual gap = 0

### INFEASIBLEの場合

次を独立Fractionで認証してください。

- exact Farkas payload取得
- inequality multipliers非負
- equality multipliersの扱い
- finite-box correction
- strict rational separation

solverが`INFEASIBLE`を返しただけではcertificate PASSとしません。

### HP100について

HP100の2件は、既存人工LPに可逆な正の対角変数変換を適用した固定入力です。

今回の目的は、高精度有理数係数・boundsを持つRA-D0型LPの処理可能性確認です。

旧pilotで失敗した`100_digit_infeasible`を再実行してはいけません。

その失敗結果は変更せず保持してください。

---

# 7. Classification規則

元の固定契約に従ってください。

| 事象 | 分類 |
|---|---|
| 全8件の期待certificateを厳密認証 | `G1_BACKEND_CLOSURE_PASS` |
| ERROR・unknown status・証明未取得 | `G1_BACKEND_ACQUISITION_INCONCLUSIVE` |
| 正しいexpected statusと異なる結果 | `G1_FIXTURE_STATUS_INCONCLUSIVE` |
| 取得された整形式certificateが数学的検証に違反 | `G1_BACKEND_INVALID_CERTIFICATE` |
| Guard・resource上限到達 | `G1_RESOURCE_INCONCLUSIVE` |
| 入出力形式・readback・runtime不一致 | `G1_TECHNICAL_INCONCLUSIVE` |

特に、**SoPlexのERRORをINVALID_CERTIFICATEへ誤分類しないでください。**

証明未取得と不正な証明は別です。

妥当なprimal upperとdual lowerが取得できてもgapが0でない場合は、固定契約どおりacquisition inconclusiveにします。

実行後に分類規則を変更しないでください。

---

# 8. Resource caps

次の上限を変更しないでください。

| 項目 | 上限 |
|---|---:|
| G1 packet全体wall | 1,200秒 |
| Phase A wall | 60秒 |
| Phase A RSS | 256 MiB |
| Backend echo calls | 8 |
| Backend LP calls | 8 |
| 各echo/solve wall | 30秒 |
| 各independent verifier wall | 30秒 |
| Backend RSS | 1,536 MiB |
| 新規output | 64 MiB |
| Concurrent solver | 1 |
| Retry | 0 |
| Build / compile | 0 |
| GPU | 0 |

既存guard v2をbyte不変で使用してください。

観測されたRSSはunique PIDごとの保守的合算であり、共有ページの重複とsampling間peakの限界があります。

CPUは既存のsubreaper / wait4 accountingを使用し、測定範囲を明記してください。

---

# 9. One-shot実行方法

PreflightをPASSした後、今回の明示的な実行認可文を含むユーザー指示原文を、privateなinstruction fileへ保存してください。

実行開始前にinstruction fileのSHA256を記録し、その内容に固定source Sが含まれていることを確認してください。

**実行時のsourceは必ずSのままにしてください。実行のために新しいauthorization-only commitを作る必要はありません。**

実行入口：

```bash
/usr/bin/python3 -B scripts/tracks/algorithm_codesign/run_g1_decision_packet.py \
  --source-commit 718cf6c1abb50c0398028ab39c24c994a4de2bd3 \
  --execute-one-shot \
  --instruction-file <EXPLICIT_INSTRUCTION_FILE>
```

`<EXPLICIT_INSTRUCTION_FILE>`は、実際に保存した指示原文ファイルのpathに置き換えてください。

既存controllerの排他的marker・stage ledgerに従い、各呼出しのkeyを実行前に消費してください。

一度markerを作成した後は、technical failureやtimeoutでも同じG1 packetを再実行しません。

---

# 10. 禁止事項

以下はすべて禁止します。

- G1 source変更
- 固定contract変更
- Fixture・expected status変更
- LP順序変更
- SoPlex設定変更
- Backend切替
- Binary再build
- Harness再compile
- 旧失敗fixtureのretry
- 新しい合成角・precision
- J1/J2/J3の登録費用評価
- Registered B2/B3 optimization
- RA-D0 v4 production実装
- RA-D0新authorization
- 科学実験・分子計算
- DF / NPZ / GPU
- Circuit / matrix / trajectory
- 旧研究結果・consumed marker・STOPの変更

構造監査がPASSしても、J1–J3を元のB2 baselineへ追加しないでください。

今回の人工LPの結果から、B3がB2より資源的に有利だと主張しないでください。

---

# 11. Evidence・commit・push

実行結果を既存契約の出力先へ保存してください。

`artifacts/track_b_g1_decision_packet_result/2026-10-09/`

少なくとも以下を保存・照合してください。

- Source identity
- User instruction hash
- One-shot marker / STOP
- Phase Aの証明・分類
- 全20境界三重交点の分類
- 6頂点の完全性・反例の有無
- B2 embeddingとprecision復元の結果
- 8 echo-only結果または未実行理由
- 最大8 LPの実行結果
- Raw solver output
- Exact primal / dual / Farkas verification
- Resource accounting
- 全fixtureのPASS / FAIL / NOT_RUN
- Protected file hash不変
- Evidence manifest

Raw outputと元の分類を結果後に改変しないでください。

実行が成功・失敗・未判定のいずれでも、認可範囲内の証拠整理を行い、別の結果commitとしてpushしてください。

新しい研究条件やsource修正を含めてはいけません。

---

# 12. GPT G1へ戻す報告

完了後、以下を報告してください。

### Source / provenance

- branch
- result full commit SHA
- remote SHA一致
- worktree clean
- source S
- original source/contract/marker不変
- new one-shot marker consumed
- mandatory STOP

### Phase A

- 最終structure classification
- 6頂点の完全性PASS/FAIL
- 導出されたparameterization
- B2 embedding結果
- precision復元結果
- 見つかった反例
- 適用範囲の限界

### Phase B

- 最終backend classification
- Echo-only PASS件数
- 実LP solve件数
- OPTIMAL件数
- INFEASIBLE件数
- Certified primal / dual / Farkas件数
- Unverified / NOT_RUN件数
- HP100結果
- 最初のblocking issue

### Resource

- Wall time
- CPU accounting
- Peak observed RSS
- Output bytes
- 残存process
- Retry数

### 研究判断への引継ぎ

最後に、次の問いへ答えるための証拠を整理してください。

1. 6頂点構造は独立に証明できたか。
2. 現在のB3自由度をどこまで簡素化できるか。
3. SoPlexをv4の限定的backendとして採用できるか。
4. Exact solverの制約・未解決問題は何か。
5. 次のproduction source設計へ進むうえで、何が未確定か。

ただし、**研究方針の最終判断はCodexで行わず、GPT側へ戻してください。**

---

## 13. Mandatory STOP

本packetの処理後は、結果にかかわらずmandatory STOPしてください。

全8件がPASSしても、次の作業へ自動進行しません。

特に、

- v4 production source実装
- Registered B2/B3 optimization
- 新しい科学実験
- 新authorization
- 追加backend pilot
- 新baseline追加

は実施しないでください。

**今回のG1 packetを、次のGPT研究レビューのための固定証拠として完結させてください。**