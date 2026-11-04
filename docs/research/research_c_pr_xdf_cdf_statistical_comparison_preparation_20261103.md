# 研究C：新PR-aware対保存XDF8/CDF8の直接統計比較・技術準備

2026-11-03。状態：**技術的GO / UNAPPROVED_EXECUTABLE_TECHNICAL_PROPOSAL**。
実H4の新CI・p値・直接統計判定は未取得。科学実行0、新sampling0。
Branchは `codex/research-c-pr-xdf-cdf-statistical-comparison-preparation-20261103`、
独立worktreeは `/tmp/prt-c-pr-xdf-cdf-statistical-comparison-preparation-20261103`。
原本の正式共通指示 `research_project_instructions_unified_2026-10-25.md` の§8、§8.4、§10を実際に読み、適用した。
この準備は科学実行・重要科学レビュー・採択・新規性判断・論文化GOの許可ではない。

## 固定した出典と対象

| 対象 | 科学SOURCE | 保存結果commit |
|---|---|---|
| 新PR-aware native32 | `4df8eca29c59ffd55e2dd898f05da16bb246ddbe` | `beb8de5cd8b71f80bb69556f92299435ff431c3e` |
| XDF8/CDF8共通ideal | `2d20ad196a406752de8e4c0ce9516b101e49d927` | `39ee9705b7a459c40c205c542d660b5d2ab90d33` |
| 元の強決定論評価 | `1aeedd96178a3f50a15bc2701c3c332f4f391b64` | `54ceff9981d9bc53210db97c743a7d1cf694f362` |

- [PR下流評価・raw入口](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md)
- [XDF/CDF共通ideal評価](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ee9705b7a459c40c205c542d660b5d2ab90d33/docs/research/research_c_xdf_cdf_common_ideal_lane_execution_20261103.md)
- [最新Evidence Map：C25/C26/C28/C29/Q02](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/169e327ae50960173bb2690ab98ea92c85e7a5ed/docs/research/research_c_near_objective_xdf_evidence_integration_20261103.md)

PR endpoint SHA256は `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
ENTRY、旧update064、旧256/640件は比較標本に入れない。XDFは `B3_XDF_spin_relative`、CDFは `B2_CDF_relative`。
XDFは元OpenFermion1.6.1 eigh順prefix8のspin recipe、CDFは保存number CDF8 general signed J/real U。
同じ分子でもDF表現・PF誤差・DF誤差・own-DF ground・native回路は候補固有であり、同一J/U比較ではない。
XDFの元snapshot SHA256は `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、
CDF fit exportは `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`。
候補別PF/native/DFの科学SOURCEはコピーした旧ideal契約の `candidate_identities` に保持している。

固定条件は直鎖H4・隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36、rank8、S2/s1、h=.1、m0..12、
CA=.00159 Ha。PRは元順LD2/r32/K0のpartial ordered S2、XDF/CDFは全frame ordered S2（r/K非適用）。
各DF自身のground-state oracle、状態準備費用除外、保存P0、head/X/Y shell込みfenced serial。
P0はQiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、
既存Gaussian精度およびMPFR160の宣言phaseを保持し、global phase fitなし。

## 保存値のbinding（新しい集計ではない）

| 保存指標 | PR | XDF8 | CDF8 |
|---|---:|---:|---:|
| 条件付き総RZ点値 | 94,288,043,542 | 100,267,421,030 | 102,056,444,569 |
| native価格 | 確率的complete kernel | 固定1回路 | 固定1回路 |
| 価格校正標本数 | 32 | 非適用 | 非適用 |
| RPE shots | 14,738 | 13,600 | 13,606 |
| kernel反復重みW | 12,539,808 | 9,211,518 | 9,225,854 |

PRの保存complete kernel平均RZは7,519.09375、保存平均SEは107.32188319090913、保存shell RZは51,583。
これらは保存resultのフィールドとの一致だけを確認した。32価格から平均・分散・SE・比・CI・p値を再計算していない。
PRの保存 `bindings/ideal_exact_native` を直接利用し、shots solverやserial cost集計を再実行しない。
会計は `W * E[complete controlled r32 kernel RZ] + shell`。Wにr32や校正件数32を掛けず、
13roundを独立した価格標本として扱わず、shellも1回だけ加算する。
固定RPE政策は価格校正値でshotsを選ぶ政策ではない（旧SOURCEのB→C→Eの順と保存bindingを確認）。

XDF/CDFは元packetと共通ideal結果のid、kernel/shell metrics、DF誤差、保存PF各roundの対応を照合した。
旧contract/SOURCE/STOPと保存61-call完了を保持。PRの旧1231-call完了も照合した。
idealのnative synthesis error=0は反実仮想の仮定であり、測定された0ではない。
PR evidence laneの無資格totalはnullのまま。Assumed laneも今回のprimaryへ混ぜない。

## Native32の監査と限界

`artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/C/samples/sample00..31_*.json`
のmetrics/events/raw/compiled/adjoint/XY、seed manifest、journal、STOP、Aのidentity/ground/law/normalization、
head/X/Y shell記録を読取り確認した。NPZ/QPYをdecodeせず、新しい演算子・信号・価格分布を評価していない。

- cohort `near_practical_PR_r32_fixed32_v1`、master `2026110304`、axis `NEAR_PRACTICAL_PR_PRICE_ONLY`、
  PCG64、非adaptive、非paired、旧cohort poolingなし。元manifest SHA256 `103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739` に結合。
- 既存 `c_near_practical_r32.seed_manifest` と `c1_stage3_connection.trajectory_seed` を使い、
  candidate/SOURCE/cohort/axis/sample番号に結合した64-bit hash seedを照合した。PRの32キーは衝突なし。
  保存manifestはENTRY側との分離記録も含むが、ENTRY価格は読まず、poolingしない。
- 各sample idは00..31の1回ずつ。各complete kernelにK0の32 microeventを含み、
  価格行は六つの有限非負整数metric。Journalのsampling/native correctness/metricsは各32 attemptと32 COMPLETE、再試行・除外・補充なし。
- 実現event列の完全一致は0組。等しい価格だけで重複標本とは判定せず、実現event列が一致しても除外しない。
  入力packetそのものの重複はsample id検査で拒否する。標本のtrim/winsorize/reweightは行わない。
- raw/compiled/adjointの保存絶対gateとXY gate、own-DF ground/normalization/lawのPASSを照合。
  これは保存correctness証拠の再確認であり、新しい回路correctness計算ではない。
- 固定SOURCEの `c_pr_integrated_performance.native` では価格を見る前にeventsを生成し、
  `c_pr_aware_df_stage_b.independent_occurrence` → `sample_occurrence` → `trotterlib.rte.iter_sample_rte_events` の
  PCG64/binary64 categorical `choice` を使う。価格依存の再採取・受理選別はない。
  complete kernel内では既存の合法relative head融合を使い、head/tailをまたぐ未観測の融合節約を仮定しない。

独立hash seedと上記実装構造は、PCG64の数学的独立性・IID・未観測support/tail・有限bit分布とideal lawの距離・
interval coverageを保証しない。固定candidateの保存finite-bit価格モデルを母集団の対象とし、
ideal条件付き資源への移送はproxyのまま未資格。元の標本が今回の仮説のprospective holdoutへ変わることもない。

## 仮説・方法・判定政策

主比較は低い固定費用のXDF8：`H0: μ_PR >= C_XDF8`、`H1: μ_PR < C_XDF8`。
推定対象は固定candidate/r32/RPE政策と保存finite-bit sampling実装に対応する `μ_PR=W E[c_kernel]+S`。
XDFは固定価格定数であり、sampling分散を創作したWelch検定は使わない。

既存 `c_r32_price_statistics.price_statistics` を使い、固定N32の一標本Student-t近似、
PR平均・SE、総RZの両側95%区間、固定XDFとの差・比の区間を評価する予定。
一側検定はSciPy `ttest_1samp(..., alternative='less')` をkernel価格と `(C_XDF-S)/W` に適用する。
一側α=.025で、両側95%区間の上端と97.5%一側上側信頼限界が一致する。
両側95%と一側α=.05を同じ判定基準として混同しない。CI上端<XDFとp<.025が数値的に食い違えば未解決とする。

現在のnear-practical native32政策の95%、halfwidth/固定比較費用≤5%、skew絶対値≤2、
delete-one平均変化≤halfwidthの.25を再利用する。主精度の分母は固定XDF。
旧update064の1%/256件、update005の0.5%/640件の精度資格を32件へ移さない。
精度不足・診断警告は全標本と推定値を保持した未解決label、分散0はCI/p値nullの未解決、
N不足・非有限値・identity/correctness不一致はFAIL/partial evidence/STOP。
診断PASSだけでIIDやtailを認証しない。

副比較CDF8は同じPR総RZ区間を固定CDF費用でシフト/除算するだけで、追加t検定・p値・cohort・校正は0。
XDF<CDFなのでXDFに対する上側boundで低費用を支持できればCDFにも含意する。
主比較の精度・診断・CI/test整合が通れば、XDFの符号未解決でも同じboundからCDFだけの支持を記述できる。
これは同じ一変量区間の像であり、独立な再現や第二の検定ではない。比は固定分母でPR側だけにsampling不確かさがある。

全labelを `POSTHOC_EXPLORATORY_...` とする。新契約を作っても事後的・探索的という資格は変わらない。
近似モデル内の低費用支持が出てもconfirmatory validation、保証付き資源優位、actual CA、全law誤差、
implemented-law signal誤差、理想law母平均への移送、同族方式全体への一般化、再最適化後の順位、
PR目的関数単独の因果効果を認定しない。Equivalence/noninferiorityもnull。

## 再利用と追加実装

- Update005/640件の `c_r32_price_statistics.price_statistics/validate_rows/distribution_diagnostics`：
  affine区間・校正・診断をそのまま再利用。
- [Update064/256件adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/src/trottertracks/c_update064_r32_price_statistics.py)：
  同じ共有一標本engineの再利用を確認。新sampling部分と旧candidate/1%政策は使わない。
- `c_near_r32_statistics.arm_summary/affine`：current native32の95%/5%/診断とW/S定義を確認。
  PR/ENTRYのWelch `primary` とshots/serial再集計は今回の実処理に使わない。
- 共通ideal評価の固定packet/result/contractを利用し、deterministic価格評価も再実行しない。
- `c1_stage1_execution.read/save/sha/journal`、`c1_stage3_integrated.Calls` を再利用。

追加は [最小adapter](../../src/trottertracks/c_pr_xdf_statistics.py)、
[passive runner](../../scripts/run_research_c_pr_xdf_statistics.py)、
[人工・管理テスト](../../tests/management/test_research_c_pr_xdf_statistics.py)。
新しい統計library、bootstrap engine、native builder、RPE solver、承認証明・grant基盤は作っていない。

## 固定契約・SOURCE・実行境界

[未承認契約](research_c_pr_xdf_cdf_statistical_comparison_contract_unapproved_20261103.json)。
実装固定commit `c761acaa5430100aaa0d605f523a13295d225133`。
[技術SOURCE](../../artifacts/research_c_pr_xdf_statistics/2026-11-03/technical/SOURCE.json) は
既存139-file closureと今回の4file、計143fileを固定。元139件は保存manifestからのdriftなし。
保存inputは213file：192 native JSON、関連metadata/journal/packet、候補と入力のbyte identity。
候補/inputのNPZはhashのみ、decodeしない。旧idealのresult/contract/SOURCE/STOPの公開済み4 JSONだけを
同一bytesで `inputs/ideal_*.json` へコピーし、元commit/pathも契約へ残した。元rawをコピー・改変していない。

- 技術SOURCE SHA256：`0f35c863224a197641bef433d3034d73f26756035c787069b3aecd44e031be2a`
- 契約SHA256：`b6ec9f13247c989187c09d066c70b1ec0f99aa40a37cc18b7ca5376100eec460`
- future fresh output：`artifacts/research_c_pr_xdf_statistics/2026-11-03/run01`（現在は未作成）

| logical call | 最大回数 |
|---|---:|
| 保存metadata読取り | 1 |
| 固定scope/会計gate | 1 |
| 既存native32 JSON読取り | 1 |
| 行・seed/cohort/correctness gate | 1 |
| 一標本主比較（校正・CI・診断・一側p・整合） | 1 |
| 同一区間からの副比較 | 1 |
| 結果summary | 1 |
| 合計 | **7** |

旧1231/61callや価格32件を新call上限に転用していない。保存値読取りと新統計callを分ける。
正常完了・統計的未解決でも7。Correctness FAILではそれ以降を開始せず、partial evidenceとSTOPを保存する。
Retry=resume=0、fresh outputのみ。旧契約や消費済みgrantを新実行へ流用しない。

既定のpassive確認コマンド（科学実行0）：

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B scripts/run_research_c_pr_xdf_statistics.py
```

別の明示実行承認を受けた場合に限り、同じコマンドへ `--execute` を付けて固定7callを実行する。
フラグは誤起動防止であり、承認の証明ではない。ユーザーにID・承認ファイル・grantの発行を求めない。
承認後もSOURCE/input driftまたは既存fresh出力があれば自動修正・再実行せずSTOPする。

## 技術確認・保全・STOP

人工52件＋共有engine回帰14件、**66 PASS / 対象外8 deselected**。
人工の既知平均/SEとt区間、affine差/比、一側pとCI、CDFのengine再呼出なし、分散0、N不足、非有限値、
精度/診断/数値境界未解決、wrong candidate/cohort/SOURCE/W/shell/lane、重複入力、correctness FAIL、
source/input drift/欠損、未opt-in拒否、fresh/retry/resume、call cap、partial evidence/mandatory STOPを確認した。
実H4 native価格はテストの数値入力にしていない。Scopeテストの保存JSONはidentity/scalar gate専用。

[Passive preflight](../../artifacts/research_c_pr_xdf_statistics/2026-11-03/technical/passive_preflight.json) PASS。
NumPy/SciPy/Qiskit/ffsim/JAX/PySCF/OpenFermionのimportを例外化したプロセスでもPASSし、新統計0を確認した。
143 SOURCEと213input、保存scope/native監査、fresh不在が一致。CLIの既定動作もpassive。

空き資源を確認（論理CPU32、available RAM約53GiB）。今回の支配処理は小さなJSONと人工32-row統計で、
重い科学kernel・GPU・parallel compileはない。逐次依存する7callを直列で扱い、FAIL後の依存起動を防ぐ。
CPU/RAM/GPU/worker/wallの人工上限は設けず、テストのみprocess-local BLASを1として不要な多重threadを避けた。
共有venv・OS・cache・他job・旧SOURCE/raw/契約/grant/ledger/marker/STOPを変更していない。
元dirty checkoutは触らず独立worktree/branchで作業。科学runner/test・新PF/RTE・shots・ground・fit・compile実行0。

旧Evidence Map C01–C29/N01–N10/G01/Q01–Q02、研究概要、validation manifestは変更していない。
無関係な既知manifest旧形式FAILの修復は今回へ含めず、global validation PASSとは主張しない。
Commit/push後はbranch SHA/tree、公開本文と主要blob、リンクを照合する。

**Mandatory STOP。** 次の担当はユーザー：この固定7callによる実H4保存標本の科学数値再解析を許可するか判断する。
許可後のCodex：同じSOURCE/input/契約に限り一度実行し、探索的なCI/p/判定と資格を報告して再びSTOPする。
新sampling/prospective cohort・移送・重要科学レビューはその許可範囲に含めない。
