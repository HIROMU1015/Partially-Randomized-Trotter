# Track B / RA-D0 v4：Resource Guard修正・Exact Backend Pilot v2

## 1. 目的と今回の認可範囲

RA-D0 v4のexact rational LP backendについて、前回のpilotで発生したresource guardの不具合を修正し、独立したPilot v2としてbackendの実証を行ってください。

検証名：

`RA_D0_V4_EXACT_BACKEND_PILOT_V2`

今回の作業は、以下の2段階とします。

**Phase A：Resource Guardの修正・独立検証**

- RSSの二重計上を修正
- CPU accountingを確認
- 出力容量・wall・process terminationを確認
- 人工プロセスで監視が正常に機能することを検証

**Phase B：SoPlex Exact Backend Pilot**

Phase AがPASSした場合に限り、SoPlexのharnessをビルドし、synthetic/off-domain LPでexact rational I/O、primal、dual、Farkasを検証する。

**Phase AがFAILした場合は、Phase Bを実行せずmandatory STOPしてください。**

今回は前回の実行のretryではありません。

別branch・別pilot contract・別execution ledgerを持つ新規の限定的な技術検証です。

ただし、同じ新pilot内での失敗後retry、resource cap緩和、別backendへの切替は認可しません。

---

## 2. 固定リポジトリ・既存結果

Repository：

`HIROMU1015/Partially-Randomized-Trotter`

今回の基点：

- branch：`track-b-ra-d0-v4-exact-backend-pilot-20261009`
- full SHA：`9e6d37fe6e345b402ba507db75f77dbec0855199`

必ず確認する資料：

1. `docs/tracks/algorithm_codesign/ra_d0_v4_exact_backend_gpt_handoff_20261009.md`
2. `docs/tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_20261009.md`
3. `docs/tracks/algorithm_codesign/ra_d0_v4_mathematical_audit_20261009.md`
4. `artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/`
5. `scripts/tracks/algorithm_codesign/exact_backend_pilot/`

既存の重要な分類：

- v4数学監査：`V4_MATH_AUDIT_PASS`
- v4 backend pilot：`V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE`

これらは変更しません。

旧v3、T0、T0.1、T0.2、数学監査、前回backend pilot、R1/R1.5、Track Aのsource・結果・分類・STOPを保護してください。

前回までの153 protected pathsに加えて、前回backend pilotで新規追加されたtracked filesもすべて保護対象にしてください。

---

# 3. 前回の停止原因

前回の監視コード：

`scripts/tracks/algorithm_codesign/exact_backend_pilot/supervise.py`

に、次のRSS集計がありました。

```python
peak = max(
    peak,
    tree_rss(process.pid) + tree_rss(os.getpid())
)
```

`tree_rss(os.getpid())`にはsupervisor配下のcompiler process treeが既に含まれるため、compiler側が二重計上されます。

前回は、

- RSS guard indicator：約1,537.664 MiB
- 設定上限：1,536 MiB
- harness compile wall：6.260秒
- compiler return code：SIGKILL
- synthetic LP calls：0

でした。

**このindicatorは真のunique RSSを示しません。**

したがって、前回の停止からSoPlexが実際に1,536 MiBを超えたとは結論できません。

ただし、実際に上限以下だったことも未確定です。

今回、この点を独立に検証してください。

---

# 4. 新branch・独立pilot

推奨branch：

`track-b-ra-d0-v4-exact-backend-pilot-v2-20261009`

前回のcommit

`9e6d37fe6e345b402ba507db75f77dbec0855199`

を基点としてください。

前回の失敗したsupervisorやexecution recordsは上書きしないでください。

新規pathにv2の監視実装・fixture・execution evidenceを作成してください。

推奨：

`scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/`

新規private runtime例：

`/tmp/ra-d0-v4-exact-backend-pilot-v2-20261009`

前回のprivate環境が残っている場合は、SoPlex静的libraryや依存物のidentityをSHA256で確認したうえで再利用して構いません。

ただし、旧pilotのログ・状態ファイル・実行記録を変更しないこと。

前回のprivate領域が存在しない場合は、保存されたversion・commit・取得物SHA256・build設定を使い、新pilotの範囲で一度だけ再構築して構いません。

前回のSTOPを解除したことにはしないでください。

---

# 5. Phase A：Resource Guard修正

## 5.1 RSS accounting

RSSの二重計上を修正してください。

最優先は、**明確に定義した監視対象のメモリ使用量を、一度だけ集計すること**です。

可能であれば、権限昇格なしに利用可能な専用cgroupのmemory accountingを優先してください。

cgroupを利用できない場合は、監視対象のprocess treeについて、unique PIDの集合を構築し、それぞれ一度だけRSSを加算する保守的な監視を使用して構いません。

ただし、unique PIDごとのRSS合算でも共有メモリが重複計上され得ます。

その場合は、

`UNIQUE_PID_RSS_SUM_CONSERVATIVE`

などの名称を使用し、物理メモリの正確なunique usageと同一視しないでください。

サンプリング監視では、観測間の瞬間的なpeakを見逃す可能性もあるため、測定精度とenforcementの限界を明記してください。

**前回の1,536 MiBというcapは変更しません。**

capに合わせて測定値を補正したり、閾値を事後的に緩めたりしないでください。

## 5.2 CPU accounting

前回は、compilerのgrandchildが停止した際、`RUSAGE_CHILDREN`から総CPUを正しく取得できませんでした。

以下を確認してください。

- 直接のchild
- compilerなどのgrandchild
- 正常終了したprocess
- guardにkillされたprocess

CPU accountingについて、可能ならcgroupまたはprocess accountingに基づき集計してください。

正確な総CPUの取得が難しい場合は、測定可能な範囲と未回収部分を明記し、推定値をexact measurementと呼ばないでください。

CPU accounting自体を理由に、resource capを緩めてはいけません。

## 5.3 Wall time

以下のwall capを保持してください。

- pilot全体：3,600秒
- LP単位：30秒
- build工程：前回と同じ個別cap以下

pilot全体の開始時刻を、新pilotの実際の開始時に固定してください。

各subprocessの実行では、全体残り時間と個別capの小さい方を適用してください。

timeout発生時はprocess groupを停止し、残存processがないことを確認してください。

## 5.4 Output cap

前回はbuild logの監視が中心で、生成物全体の容量を継続的に監視できていませんでした。

今回は、監視対象に少なくとも以下を含めてください。

- build logs
- compiler生成物
- harness binary
- solver stdout/stderr
- 中間certificate
- pilot生成JSON
- その他pilotが新規生成する成果物

取得済みの外部source・依存packageをoutput capから除外する場合は、対象と除外範囲を実行前に固定してください。

**64 MiBの上限を変更しないこと。**

同じファイルを複数pathから二重計上しないようにしてください。

## 5.5 Process termination

guardが発火した場合、

1. 新規subprocessの生成を停止
2. 対象process groupを終了
3. 残存child/grandchildを確認
4. termination evidenceを記録
5. 追加実行を禁止

してください。

cap到達後の自動retryを実装しないでください。

---

# 6. Phase Aの独立検証

SoPlex harnessをコンパイルする前に、監視機構だけの独立したsynthetic testsを行ってください。

最低限：

1. childが1つのケース
2. childとgrandchildが存在するケース
3. 複数childが存在するケース
4. child終了後にgrandchildが残るケース
5. 通常終了
6. wall timeout
7. RSS guard発火
8. output cap発火
9. SIGKILL後のprocess残存確認
10. CPU accountingの検証
11. process tree二重計上が発生しないこと
12. 監視対象外processを誤って計上しないこと
13. total pilot wallの引継ぎ
14. guard発火後のretry禁止

を検証してください。

このテストでは小さな人工capを使用して構いません。実際に大量のメモリやCPUを消費させる必要はありません。

本番pilotの1,536 MiB capは変更しないでください。

テスト結果を確認する前に、guard implementationとtest specificationを固定し、SHA256を保存してください。

### Phase A判定

`GUARD_V2_PASS`

全監視・停止・accountingテストが、明示した測定方式の保証範囲内でPASS。

`GUARD_V2_REVISION_REQUIRED`

guardの実装が不正確、または実行前に合意した安全条件を満たさない。

`GUARD_V2_TECHNICAL_INCONCLUSIVE`

必要なprocess accountingや安全な停止を確認できない。

**PASS以外ではPhase Bへ進まないでください。**

---

# 7. Phase Bの実行前freeze

Phase AがPASSした場合、Phase Bの実行前に以下を固定してください。

- backend version/build identity
- guard v2 SHA256
- harness source SHA256
- independent verifier SHA256
- synthetic fixture manifest SHA256
- LP実行順序
- solver call cap
- wall/RSS/output caps
- classification rules

前回定義した23件のsynthetic fixtureを使用してください。

ただし、前回のfixtureはguard停止後に保存されたものであり、実行前登録済みの結果ではありません。

今回、**新pilotの開始前にfixture内容と実行順序を固定する**ことで、検証入力を明確にしてください。

fixtureの誤りを事前の静的検査で見つけた場合は、Phase B開始前に記録してください。

Phase B開始後に結果を見てfixtureや判定基準を変更してはいけません。

---

# 8. Phase B：SoPlex Harness Build

対象backend：

- SoPlex 7.0.0
- commit `6657fb3b27044bad7bf2bb58b16de2461de82109`
- GMP 6.2.1
- Boost 1.74.0

前回の静的library identityを照合してください。

Pilot harness：

`scripts/tracks/algorithm_codesign/exact_backend_pilot/harness.cpp`

を参照してください。

APIやcompiler optionsは固定SoPlex 7.0.0 sourceと照合してください。

特に、

- rational input API
- exact solve mode
- exact check mode
- rational primal API
- rational dual API
- rational Farkas API

を確認してください。

SoPlexの内部でfloating-point refinementが使われても、入力がexactに保存され、最終的な証明を独立Fraction verifierで認証できれば検証対象として構いません。

ただし、floatで丸めた入力を元のFraction入力と同一視しないでください。

### Buildの制限

- isolated environmentのみ
- compiler build concurrencyは1
- 既存system environment変更なし
- 新しいproduction dependency変更なし
- fixed resource capsを維持
- harness compileは新pilot内で1回

コンパイルに失敗した場合は、原因を記録してSTOPしてください。

**失敗後にharnessを修正して再コンパイルすることは認可しません。**

ソース/API上の問題が判明した場合はGPTへ返してください。

---

# 9. Rational I/Oの検証

最初に以下をexact roundtripしてください。

\[
\frac13,\quad
\frac27,\quad
2^{-60},\quad
10^{-18}.
\]

さらに約100桁の整数を含む有理数を検証してください。

確認事項：

- Fractionからbackendへのexact input
- SoPlex内部rational coefficient
- rational output
- Fractionへの復元
- 全係数・bounds・rhs・objectiveの一致

solver側から読み戻した値と、元の入力のexact equalityを検証してください。

**この段階でFAILした場合、LP最適化へ進まないでください。**

---

# 10. Exact Primal / Dual / Farkas

前回の数学監査で固定されたLP証明規約を維持してください。

LP：

\[
\min c^\mathsf Tx+c_0
\]

subject to

\[
Ax\le b,\qquad Hx=f,\qquad 0\le x\le U.
\]

## Primal

solverが返したrational solutionについて、

\[
Ax\le b,\qquad Hx=f,\qquad 0\le x\le U
\]

を、独立Fraction verifierで確認してください。

## Dual

非負不等式multiplier \(\nu\) と自由等式multiplier \(u\) を用い、

\[
r=c+A^\mathsf T\nu+H^\mathsf Tu
\]

\[
L=
c_0-\nu^\mathsf Tb-u^\mathsf Tf+
\sum_j\min(0,r_j)U_j
\]

をexactに検証してください。

solver APIのdual符号と上記規約の対応を明示してください。

Primal upperとのweak dualityも確認してください。

## Farkas

\[
r=A^\mathsf T\nu+H^\mathsf Tu,\qquad \nu\ge0
\]

について、

\[
\nu^\mathsf Tb+u^\mathsf Tf
<
\sum_j\min(0,r_j)U_j
\]

を厳密に検証してください。

solver statusが`INFEASIBLE`でも、この証明が成立しなければcertified infeasibleとしないこと。

Dual/Farkasの符号対応は、API規約を調べ、検証前に固定してください。

結果に合わせた恣意的な符号変更やrational reconstructionをしないでください。

---

# 11. Synthetic LPの実行順序

登録済みRA-D0のLPは一切解かないでください。

今回の実行順序は以下とします。

### Gate B1：最小機能確認

- Rational I/O
- 小型feasible LPのexact primal
- 小型LPのexact dual
- 明らかなinfeasible LPのexact Farkas

すべて独立verificationを行う。

この段階で重大なcertificate failureがあれば、それ以上進めずSTOPしてください。

### Gate B2：精度境界

- dyadic coefficient
- 極小のfeasibility gap
- 大きな有理数
- degeneracy
- active finite bounds

を検証。

### Gate B3：RA-D0型人工LP

- B2-shaped：8 groups × 3 precisions
- B3-shaped：7 groups × 3 precisions
- normalization
- degree matching
- mean reserve
- confidence reserve
- resource caps

を含む人工LPを検証。

入力係数は人工的な有理数だけを用いてください。

**保存済みRA-D0 tableの係数を利用したregistered LPの求解は禁止です。**

---

# 12. Resource benchmark

以下を固定してください。

| 項目 | Cap |
|---|---:|
| 同時solver実行 | 1 |
| Solver threads | 1 |
| Pilot全体wall | 3,600秒 |
| Solver calls | 最大30 |
| 1 LP wall | 30秒 |
| RSS guard | 1,536 MiB |
| Output | 64 MiB |
| Retry | 0 |
| GPU | 0 |

各fixtureについて記録：

- LP dimensions
- rational coefficient bit-length
- solve wall time
- primal取得時間
- dual取得時間
- Farkas取得時間
- independent verification時間
- memory usageおよび測定方式
- output bytes
- solver status
- exact certificate status

同じfixtureを繰り返し解くbenchmarkは行わないでください。

今回の目的は、再現性のある詳細な性能分布を確立することではなく、**少なくともRA-D0型のLPを厳密に求解・検証できるか判断すること**です。

少数の人工LPだけから、旧RA-D0の全queryを一定時間で処理できると結論してはいけません。

旧v3の55,275 main LP / 110,550 total callsは参考情報であり、v4へ自動継承しないこと。

---

# 13. Independent Verificationの独立性

可能な限り、

1. SoPlex harness
2. Saved neutral certificate
3. Independent Fraction verifier

を分離してください。

VerifierはSoPlex libraryをimport・linkせず、保存されたexact input/outputだけを読み込みます。

最低限以下を保存：

- solver input exact representation
- solver-echoed LP coefficients
- primal solution
- dual multipliers
- Farkas ray
- resource accounting
- independent verifier verdict

ソルバー自身の成功statusをcertificateの代わりに使用しないでください。

認証不能なstatusは明確に`UNVERIFIED`としてください。

---

# 14. Failure Semantics

次を厳密に区別してください。

**A. Guard failure**

監視・停止機構そのものが不正確。

**B. Build failure**

Guardは正常だがharnessのコンパイルが失敗。

**C. Resource limit**

正常なGuardがcap到達を検出。

**D. Solver failure**

Backendが解・証明を取得できない。

**E. Certificate failure**

Solver出力を独立Fraction verifierが認証できない。

**F. Exact backend success**

Rational I/O、primal、dual、Farkas、独立verificationがPASS。

特に、backend setup failureやtime limitを元のB2/B3のinfeasibilityと呼ばないでください。

また、inner LPのFarkasはinnerのinfeasibilityしか意味しません。

元のB2 classを包含するouter LPについてのみ、適切な証明を条件としてoriginal-class infeasibilityを主張できます。

---

# 15. 最終Classification

次のいずれかを選んでください。

### `V4_EXACT_BACKEND_PILOT_V2_PASS`

- Resource guard PASS
- Harness build PASS
- Rational I/O PASS
- Exact primal PASS
- Exact dual PASS
- Exact Farkas PASS
- Independent Fraction verification PASS
- 必須synthetic LPをcap内で実行
- 重大なblocking issueなし

### `V4_EXACT_BACKEND_GUARD_UNVERIFIED`

Guardが独立検証を通らないため、backend executionへ進めない。

### `V4_EXACT_BACKEND_BUILD_BLOCKED`

GuardはPASSしたが、harnessのcompile/linkが失敗した。

### `V4_EXACT_BACKEND_PARTIAL`

Primal等は認証できたが、dual/Farkasなど必須機能の一部が取得できない。

### `V4_EXACT_BACKEND_CERTIFICATE_FAIL`

取得したrational値・証明が独立verificationを通らない。

### `V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE`

正常なGuardがresource capを検出し、検証が未完了。

### `V4_EXACT_BACKEND_TECHNICAL_INCONCLUSIVE`

その他の技術的障害により判定不能。

上記の分類を今回の実行前に固定してください。

---

# 16. 成果物

推奨Docs：

- `docs/tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_v2_20261009.md`
- `docs/tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_v2_gpt_handoff_20261009.md`

Artifacts：

`artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/`

最低限：

- `input_identity_v1.json`
- `guard_v2_contract_v1.json`
- `guard_v2_test_results_v1.json`
- `backend_identity_v1.json`
- `execution_contract_v1.json`
- `rational_io_v1.json`
- `primal_audit_v1.json`
- `dual_audit_v1.json`
- `farkas_audit_v1.json`
- `synthetic_results_v1.json`
- `resource_benchmark_v1.json`
- `verification_v1.json`
- `evidence_manifest_v1.json`

未実行の検証は`NOT_RUN`と記録してください。

PASSしたことにして空の出力を埋めないでください。

---

# 17. 禁止事項

- 旧backend pilotのretry・上書き
- 既存source/result/marker変更
- v4 production source実装
- registered B2/B3 optimization
- 新authorization
- RA-D0 one-shot
- 実際のbudget freeze
- science / molecule / DF / NPZ
- synthesis / new angles / precision
- IS / CTS
- circuit / matrix / trajectory
- GPU
- 元の研究結果の再分類
- 別backendへの無許可切替
- cap変更
- 失敗後の再compile・solver retry

Guardの修正と検証はPhase Aとして許可します。

ただし、Phase B開始後の失敗を受けてGuard・harness・fixture・certificate ruleを修正し、同じpilotを続けることは禁止します。

---

# 18. 最終報告

commit・push後、以下を報告してください。

- branch
- full SHA
- remote SHA一致
- worktree clean
- Phase A classification
- Phase B classification
- Guard実装の変更点
- Guard独立testsのPASS/FAIL
- RSS監視方式・実際のpeak・測定上の限界
- CPU accounting方式
- output cap enforcement
- SoPlex build status
- Rational I/O PASS/FAIL
- Exact primal PASS/FAIL
- Exact dual PASS/FAIL
- Exact Farkas PASS/FAIL
- Independent Fraction verification PASS/FAIL
- Synthetic LP実行件数
- LP dimensions
- solve time・verification time・RSS
- blocking issue
- 旧研究・前pilotのprotected hashes不変
- registered LP calls = 0
- production source changes = 0
- science/synthesis = 0
- retries = 0

最後に、

**「SoPlex 7.0.0をRA-D0 v4のexact rational LP backendとして採用できるか」**

について、証拠に基づくGO/STOP提案を提示してください。

ただし、最終的なbackend採用と研究方針の判断はGPT側で行います。

---

# 19. Mandatory STOP

Phase AがPASSしない場合は即STOP。

Phase AがPASSした場合に限り、Phase Bを実行します。

Phase Bの検証完了後、結果をcommit・pushしてmandatory STOPしてください。

`V4_EXACT_BACKEND_PILOT_V2_PASS`であっても、production source実装、authorization、registered optimizationへ自動移行しないでください。

このpilotをもってexact backendの技術的採否を判断できるだけの証拠を得ることを目的とします。

同種の技術的な修復pilotを無期限に繰り返す方針にはしません。