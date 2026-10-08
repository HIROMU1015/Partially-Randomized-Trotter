# Track B / RA-D0 v4：Exact LP Backend Feasibility Pilot

## 1. 今回の目的

RA-D0 v4の数学監査が完了したため、次の段階として、**exact rational LP backendの独立した実装可能性検証**を実施してください。

今回の検証名：

`RA_D0_V4_EXACT_BACKEND_FEASIBILITY_PILOT`

GPT側の判断：

- 数学的設計：`V4_MATH_AUDIT_PASS` を受理
- Backend：`UNVERIFIED_BACKEND`
- Exact backend pilot：GO
- v4 production source：未認可
- Registered optimization：未認可
- 新しいone-shot：未認可

今回の目的は、次を検証することです。

1. 有理数係数をfloatへ落とさずにLPへ入力できるか。
2. 厳密に検証可能なprimal解を取得できるか。
3. 元のLPに対するcertified dual lower boundを取得できるか。
4. Infeasibilityに対するFarkas certificateを取得できるか。
5. 小規模なB2/B3相当LPを現実的な時間・メモリで処理できるか。
6. production実装へ進むために残る問題を特定できるか。

**今回はbackend単体の検証です。RA-D0 v4のproduction pipelineを実装しないでください。**

---

## 2. 固定入力と既存研究の保護

リポジトリ：

`HIROMU1015/Partially-Randomized-Trotter`

今回の基点：

- branch：`track-b-ra-d0-v4-mathematical-audit-20261009`
- commit：`beb82427d202f479cc2ba954480d73a51941e322`

主要参照資料：

`docs/tracks/algorithm_codesign/ra_d0_v4_gpt_handoff_20261009.md`

`docs/tracks/algorithm_codesign/ra_d0_v4_mathematical_audit_20261009.md`

`artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/`

旧RA-D0 v3のsource・authorization・消費済みmarker・結果、およびT0/T0.1/T0.2、R1/R1.5、Track A、過去STOPは変更しないでください。

今回のpilotは、これらの数値結果を更新・再分類するものではありません。

---

# 3. 第一候補backend

第一候補：

**SoPlex（exact rational LP mode）**

公式資料：

- https://soplex.zib.de/
- https://soplex.zib.de/doc-7.0.0/html/EXACT.php
- https://github.com/scipopt/soplex

ただし、SoPlexの採用は未確定です。

特に、公式APIにrational関連機能が存在することと、現在の環境でそれらを正常利用できることを区別してください。

必要な検証は、

- exact rational input
- exact primal
- exact dual
- Farkas certificate
- independent verification
- runtime/resource measurements

です。

SoPlexの具体的なversion・build・API・CLI optionsは公式資料と実際の導入物で確認し、推測で指定しないでください。

---

# 4. 実行環境と依存関係

## 4.1 隔離環境

既存の研究用Python環境、RA-D0 runtime、production solver環境を変更しないでください。

独立したworktreeと、pilot専用の隔離build/runtimeを使用してください。

推奨branch：

`track-b-ra-d0-v4-exact-backend-pilot-20261009`

許可するのは、pilot専用領域への必要最小限のbackend導入・ビルド・検証です。

禁止：

- system-wide package installation
- root/sudoによる環境変更
- 既存Python環境へのdependency追加
- 旧sourceのsolver置換
- production requirements変更
- 既存環境の共有libraryを上書き

必要なコンパイラ・GMP等が存在しない場合は、まずread-only inventoryを行ってください。

隔離領域内だけでは安全に導入できない場合は、無理に進めず`BACKEND_SETUP_BLOCKED`として報告してください。

## 4.2 Provenance

少なくとも以下を記録してください。

- backend name/version
- source release/tag/full commit
- download URLと取得物SHA256
- build設定
- compiler version
- GMPなど主要dependencyのversion
- 実行binary SHA256
- relevant shared libraries
- license inventory
- exact-mode configuration
- Python/C++ bindingまたはCLIの使用方法

backendと依存関係のidentityを再現可能な形にしてください。

---

# 5. Exact rational inputの検証

ここは最重要です。

LP係数をPython `Fraction`で表したとき、**有理数値を保持したままsolverへ渡せること**を確認してください。

検証する値の例：

\[
\frac13,\quad
\frac27,\quad
2^{-60},\quad
\frac1{10^{18}}
\]

さらに、約100桁の整数を分子・分母に含む有理数も検証してください。

具体的には、

1. Fractionからsolver入力形式への変換
2. solver側での有理数値の解釈
3. rational outputの取得
4. Fractionへの復元
5. 入出力のexact identity

を確認します。

**floatやdoubleへ変換した値を、その後Fraction化しただけではexact rational inputとは認めません。**

CLIが有理数を正確に扱えない場合、公式のrational C++ APIによる最小のpilot harnessを検討して構いません。

ただし、これはpilot専用コードに限定してください。

---

# 6. Primal feasibilityの独立認証

LPを次の標準形として扱います。

\[
\min_x c^\mathsf Tx+c_0
\]

subject to

\[
Ax\le b,
\]

\[
Hx=f,
\]

\[
0\le x\le U.
\]

solverが返したprimal \(x^*\) について、solverのsuccess statusを信用するだけでなく、独立したFraction verifierで、

\[
Ax^*\le b,
\qquad
Hx^*=f,
\qquad
0\le x^*\le U
\]

をすべて厳密に検証してください。

少なくとも、

- inequality violations
- equality residuals
- lower/upper variable bounds
- objective value
- exact rational representation

を保存します。

toleranceを使ったnear-feasibleをcertified feasibleと呼ばないでください。

---

# 7. Dual lower boundの独立認証

RA-D0の元のdual certificate規約との整合性を確認してください。

不等式multipliersを

\[
\nu\ge0
\]

等式multipliersを

\[
u\in\mathbb Q^{m}
\]

としたとき、stationarity residualを、

\[
r_j
=
c_j+
\sum_i\nu_iA_{ij}
+
\sum_\ell u_\ell H_{\ell j}
\]

とする。

有限変数上界 \(U_j\) を用いたcertified lower boundは、

\[
\boxed{
L=
c_0-\nu^\mathsf Tb-u^\mathsf Tf
+\sum_j\min(0,r_j)U_j
}
\]

です。

これを独立に計算してください。

重点確認：

- solverが返すdual multiplierの符号規約
- lower/upper bound multipliersの扱い
- objective offset
- finite-domain stationarity correction
- equality multiplierの自由符号
- exact weak duality
- degenerate LP
- optimal solutionが一意でない場合

少なくともcertified primal upper \(U_{\mathrm{primal}}\) と、

\[
L\le U_{\mathrm{primal}}
\]

が成立することを確認してください。

solverが返すnominal objective値だけからlower boundを構成しないでください。

---

# 8. Farkas certificateの独立認証

Infeasibilityもsolver statusだけでは認証しません。

元LPの不等式・等式・有限変数上界を含め、exact Farkas certificateを検証してください。

例えば、

\[
r_j=
\sum_i\nu_iA_{ij}
+
\sum_\ell u_\ell H_{\ell j},
\qquad \nu\ge0
\]

として、

\[
\nu^\mathsf Tb+u^\mathsf Tf
<
\sum_j\min(0,r_j)U_j
\]

が厳密に成立すれば、元のLPのinfeasibilityを証明できます。

この条件と同等の正しいcertificate形式でも構いませんが、符号と有限boundsの扱いを明示してください。

最低限、

- 明らかなinfeasible LP
- equalityとinequalityの矛盾
- 有限upper boundによるinfeasibility
- feasibleとinfeasibleの境界に近い人工LP

で検証してください。

SoPlexがexact primal/dualを返せても、Farkas certificateを取得できない場合は、その機能をPASSとしないでください。

別のbackendへ無制限に切り替えることは禁止します。問題を報告してGPTへ戻してください。

---

# 9. Synthetic/off-domain LP fixtures

登録RA-D0のLPは解かないでください。

代わりに人工有理数LPを作成し、以下を検証してください。

### A. 基本LP

- feasible / unique optimum
- feasible / multiple optima
- equalityを含むLP
- upper boundsがactive
- 退化したLP
- infeasible LP

### B. 高精度LP

- 係数に \(2^{-60}\) を含む
- \(10^{-18}\) 程度の制約差
- 大きく異なる係数scale
- 100桁程度の有理数
- feasible/infeasibleを微小差だけで区別する

### C. RA-D0型の人工LP

元のRA-D0に近い構造・変数規模を持つsynthetic LPを構成する。

少なくとも、

- B2相当：8 logical groups × 3 precisions
- B3相当：7 logical groups × 3 precisions
- normalization
- degree matching
- mean reserve
- confidence reserve
- resource constraints
- finite variable bounds

を含めてください。

係数は人工的に生成した有理数のみを使用してください。

**保存済みRA-D0 candidate tableの係数を使ってregistered LPを再構成・求解してはいけません。**

数学的な行構造や変数数を参考にすることは認めます。

---

# 10. 取得した証明の独立性

可能なら、backendへの入力と出力をJSON等の中立形式へ保存し、**solverやsolver libraryをimportしない別のFraction verifier**で認証してください。

最低限、以下の2つを分離します。

1. Backend runner：LPを解き、候補とcertificateを出力する。
2. Independent verifier：保存された入力・出力だけを用いて証明を検証する。

verifierはsolver status、nominal feasibility tolerance、内部のfloating-point判定に依存してはいけません。

certificate取得失敗時には、無理にrational reconstructionしてPASS扱いしないでください。

---

# 11. Resource benchmark

数学的に正しくても、計算負荷が大きすぎればRA-D0へ利用できません。

そのため、人工LPについて、

- LP dimensions
- exact input bit-length
- primal solve time
- dual acquisition time
- Farkas acquisition time
- independent verification time
- peak RSS
- output size

を計測してください。

小規模な複数条件を使用し、実行時間のばらつきも保存してください。

今回は限定pilotとして、暫定的に次のhard capを設定します。

| 項目 | 上限 |
|---|---:|
| 同時solver実行数 | 1 |
| Solver threads | 1 |
| Pilot全体wall | 3,600秒 |
| LP solve calls | 30件 |
| 1件のLP wall | 30秒 |
| Pilot peak RSS | 1,536 MiB |
| Pilot output | 64 MiB |
| Retry | 0 |
| GPU | 0 |

これらは**pilotの安全上限**であり、v4 production capではありません。

可能ならsolverを別processとして起動し、独立したsupervisorでwall capを強制してください。

cap到達時は結果を保存してSTOPし、設定変更やretryを行わないでください。

ビルドが完了しない場合も、無制限に時間を延長しないでください。

---

# 12. Production計算量の予備評価

人工LPの計測結果から、旧RA-D0のquery規模との関係を整理してください。

旧v3の理論最大：

- main LP：55,275
- auxiliary込み：110,550

ただし、**この件数を新v4へ自動適用しないでください。**

exact solverでは、

- continuous inner LP
- B2 outer LP
- dual/Farkas取得
- independent verification

のコストが異なる可能性があります。

また、v4の保守的inner infeasibilityと元クラスのinfeasibilityは異なります。

したがって、単純な「平均時間 × 55,275」だけで実行可能と結論せず、まず小規模pilotの計測範囲と限界を示してください。

必要なら、今後のproduction source設計において、

- anchor-firstの維持
- query数の削減
- certificate取得の効率化
- 出力サイズの抑制

が必要か提案してください。

ただし、研究条件やquery recipeは今回変更しません。

---

# 13. Infeasibility semantics

以下を混同しないでください。

| 状態 | 意味 |
|---|---|
| Exact primal certified feasible | 対象LPに実行可能解がある |
| Exact dual lower certified | 対象LPの目的値下界が認証された |
| Exact Farkas certified infeasible | **対象LP**が実行不能 |
| Inner LP infeasible | 保守的な生成器の実行不能。元クラスの実行不能とは限らない |
| Outer LP certified infeasible | 包含関係の証明を前提に、元クラスの実行不能を示せる |
| Backend failed / timeout | Technical inconclusive |

人工fixtureについても、inner infeasibilityを元のclassのinfeasibilityへ読み替えないでください。

---

# 14. 今回の最終判定

以下のいずれかを選択してください。

### `V4_EXACT_BACKEND_PILOT_PASS`

- exact rational input PASS
- exact primal certificate PASS
- exact dual certificate PASS
- exact Farkas certificate PASS
- independent Fraction verification PASS
- 必要な人工LPをresource cap内で処理
- 重大なblocking issueなし

ただし、production実装や全queryの実行可能性はまだ保証しない。

### `V4_EXACT_BACKEND_PARTIAL`

一部の機能はPASSしたが、dualまたはFarkasなど重要機能が未実証。

### `V4_EXACT_BACKEND_UNAVAILABLE`

隔離環境でbackendを利用できず、実証へ進めない。

### `V4_EXACT_BACKEND_CERTIFICATE_FAIL`

取得した証明が独立Fraction検証を通らない、あるいはrational I/Oの正確性を確認できない。

### `V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE`

時間・メモリ・call数などの上限に到達して未完了。

なお、環境構築失敗と証明失敗は区別してください。

---

# 15. Branchと成果物

基点：

`beb82427d202f479cc2ba954480d73a51941e322`

推奨branch：

`track-b-ra-d0-v4-exact-backend-pilot-20261009`

Docs：

- `docs/tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_20261009.md`
- `docs/tracks/algorithm_codesign/ra_d0_v4_exact_backend_gpt_handoff_20261009.md`

Artifacts：

`artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/`

最低限：

- `input_identity_v1.json`
- `backend_inventory_v1.json`
- `build_runtime_identity_v1.json`
- `rational_io_roundtrip_v1.json`
- `synthetic_fixture_manifest_v1.json`
- `primal_certificate_audit_v1.json`
- `dual_certificate_audit_v1.json`
- `farkas_certificate_audit_v1.json`
- `resource_benchmark_v1.json`
- `failure_semantics_v1.json`
- `verification_v1.json`
- `evidence_manifest_v1.json`

必要なpilot専用script・harness・testsは、新規pathへ追加して構いません。

外部backendのソース一式や大容量binaryをGitHubリポジトリへcommitする必要はありません。backend identityと再現手順を保存してください。

---

# 16. 変更・実行禁止事項

今回も以下は禁止します。

- 旧RA-D0 v3の再実行
- 旧markerの削除・変更
- 新authorization
- RA-D0 v4 production source実装
- registered B2/B3 solve
- 実際のbudget freeze
- 新しいcandidate table
- 新synthesis
- 新angle / precision
- IS / CTS
- science
- molecule / DF / NPZ
- circuit / matrix / trajectory
- GPU
- 旧R1/R1.5・Track A変更
- 旧研究結果の再分類

今回許可されるsolver呼出しは、**隔離環境におけるsynthetic/off-domain LPのみ**です。

既存135 protected filesのidentityを監査し、不変を確認してください。

---

# 17. 完了報告

commit・push後、次を報告してください。

- branch
- full SHA
- remote SHA一致
- worktree clean
- final pilot classification
- backend name/version/build
- isolated environment identity
- rational input roundtrip結果
- exact primal PASS/FAIL
- exact dual PASS/FAIL
- exact Farkas PASS/FAIL
- independent verification PASS/FAIL
- synthetic LPの実行件数
- solve time / verification time / peak RSS
- 実行可能と判断できるLP規模
- production利用時の懸念事項
- 新dependencyとライセンスの記録
- 元v4数学監査・旧v3/T0/T0.1/T0.2 protected hashes不変
- registered LP calls = 0
- new synthesis/science = 0
- production source modifications = 0
- blocking issue

最後に、

**「このbackendをRA-D0 v4の数値実装へ採用できるか」**

について、証拠に基づくGO/STOP提案を提示してください。

## 18. Mandatory STOP

監査・pilotを完了したら、結果をcommit・pushしてmandatory STOPしてください。

`V4_EXACT_BACKEND_PILOT_PASS`でも、production source実装、authorization、新one-shotへ自動移行しないでください。

次の研究判断と実装方針の決定はGPT側で行います。