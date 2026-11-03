# 研究C：保存CDF8/XDF8・共通ideal laneの61-call実行結果

2026-11-03 JST。HIROMU1015/Partially-Randomized-Trotter。

**固定61/61 logical callsを一度実行し、両候補が `CONDITIONAL_IDEAL_CA_PASS`。結果は `COMMON_IDEAL_LANE_COMPLETE`、失敗0、retry/resume 0、mandatory STOP。** Shell込み条件付き総RZはCDF8 **102056444569**、XDF8 **100267421030**。今回の二候補間では、既存比較engineの判定は `XDF_LOWER_CONDITIONAL_INTERROGATION_COST` だった。

これはnative synthesis signal errorを0と仮定した**反実的ideal lane**である。実装誤差0の測定、actual CA達成、厳密誤差包囲、PRに対する統計的優位・同等性・非劣性の認定ではない。候補固有DF/PF/ground/native証拠と旧numerical synthesis laneを保持した。

[実行結果JSON](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/result.json) ／ [STOP](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/STOP.json) ／ [call journal](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/science_calls.jsonl) ／ [事前照合](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/execution_preflight.json) ／ [事後照合](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/execution_postflight.json) ／ [準備報告](research_c_xdf_cdf_common_ideal_lane_preparation_20261103.md)。

## 1. 実行範囲・SOURCE・来歴

技術準備完了後の明示的な科学実行許可に基づき、準備済みの固定二候補・保存JSON30件・最大61-callだけを実行した。契約の歴史的な `UNAPPROVED_SCIENCE_CONTRACT` metadataは変更していない。チャット許可に承認証明・新grant・原文保存・認証機構は追加していない。

| 固定物 | Identity |
|---|---|
| 今回の科学実行SOURCE | `2d20ad196a406752de8e4c0ce9516b101e49d927` |
| Adapter実装commit | `cc051d9e7b80a7818730104d45ff9a9664d8c95b` |
| 技術起点・固定監査A | `8375433869999c2e10501b1983c11b2b22c5d423` |
| 旧数値合成laneの科学SOURCE | `1aeedd96178a3f50a15bc2701c3c332f4f391b64` |
| 旧数値合成laneの結果commit | `54ceff9981d9bc53210db97c743a7d1cf694f362` |
| [実行contract](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/contract.json) | SHA256 `6c4ef36ab1b7fd57fc8d516138515012be30909ab8ddfc99806f7df60526b7eb`、準備時から同一bytes |
| [技術SOURCE manifest](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/technical/SOURCE.json) | 84 files、SHA256 `8761deb2a35e904252b6091ae96f872aa218166286bccf0584089703dc6143e5` |
| [実行SOURCE record](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/SOURCE.json) | 同じ84 filesと `execution_commit=2d20ad196a406752de8e4c0ce9516b101e49d927` |
| 実行branch | `codex/research-c-xdf-cdf-common-ideal-lane-execution-20261103` |
| 専用worktree | `/tmp/prt-c-xdf-cdf-common-ideal-lane-execution-20261103` |
| Fresh出力 | `artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01` |

今回の結果を収録するcommitは閲覧・公開用であり、実行SOURCEへ付け替えない。元H4 inputのSHA256は `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。本実行はNPZを読まず、固定済みJSON evidenceのみを使用した。

CDFは `B2_CDF_relative`、fit SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11`、export SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`。XDFは `B3_XDF_spin_relative`、元OpenFermion1.6.1 eighのsigned λ/G・rank10 snapshotの元順prefix8。両者を同じJ/U・同じ近似Hamiltonianとして扱わない。

| 候補 | DF SOURCE | PF/ground SOURCE | Native SOURCE |
|---|---|---|---|
| CDF8 | `bec41ac7dc7e034bd664817acfdcfed858ddadd0` | `6ea5c6a9c10cc4b5411039a0d61a9f19acc4f34f` | `880006396326a4d46f4a4030564279e92fadb9cf` |
| XDF8 | `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5` | `328f31007ee8069d1f7ba24f91029a3748b59109` | `b40fce5c62c9fff97f68073cbc02182e4bbf2b9b` |

旧packetを既存loader/attachmentで再接続した結果は、保存公開packetと完全一致した。元κ・scalar・frame順・route固有の対角化/融合、旧SOURCE/契約/STOPを変更していない。より詳しい出典と再利用APIは[準備報告](research_c_xdf_cdf_common_ideal_lane_preparation_20261103.md)の固定記録を参照する。

## 2. 科学条件と評価資格

| 条件 | 固定値 |
|---|---|
| 物理問題 | Linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36、8 spin orbitals |
| DF・PF | 保存CDF8/XDF8、全frame ordered symmetric S2、s=1、元frame順0..7。決定論のためsplit L_D・r/Kは非該当 |
| 時間・round | δ=h=0.1、m=0..12、各roundのkernel反復2^m |
| 精度 | CA=0.00159 Ha、DF reserve=0.00019875 Ha、βtotal=π/6、βstat=π/12 |
| Failure | αaxis=.01/26、各候補αtotal=.01。二候補の選択後・同時保証へ拡張しない |
| 状態 | 各DF自身のground-state oracle、保存alias gate、状態準備費用除外 |
| Signal | native synthesis誤差0を反実仮定、attenuation=1、RTE=0、finite-law=0、deterministic law |
| 残す誤差 | 候補固有の保存PF13値、保存DF sector norm＋guardをenergy側に一回 |
| Native価格 | 保存P0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false |
| 位相・会計 | 保存MPFR160 declared phase、合法なGaussian/relative fusion、head/X/Y shell込みfenced serial、scheduled serial depth |

既存 `c1_stage3_accuracy` のround/energy/aggregate/compare、既存RPE Hoeffding solverとfixed-N margin検査、`c1_h6_ca_resources.serial_round_cost` を再利用した。新しいaccuracy式・shots solver・費用engineは作っていない。新実装は準備済みadapter/CLIだけで、実行時のSOURCE変更は0件。

Ideal synthesisはexact Hamiltonian evolutionを仮定するものではない。PF誤差はsignal側に残り、DF誤差はenergy側に残る。価格を反実的に最適化せず、旧CDFの8モードκ直接対角化とB3 XDFのspin対応routeの差も保持した。

## 3. 保存された評価結果

| 指標 | CDF8・B2 | XDF8・B3 |
|---|---:|---:|
| 候補status | CONDITIONAL_IDEAL_CA_PASS | CONDITIONAL_IDEAL_CA_PASS |
| Energy十分条件・Ha | 0.0013549794824010532 | 0.001373656337788953 |
| 保存DF norm＋guard・Ha | 0.00007666215916301914 | 0.00009533901455091894 |
| 予定shots / executions（X/Y合計） | 13606 | 13600 |
| W（全予定shotsのkernel反復重み） | 9225854 | 9211518 |
| 総RZ | 102056444569 | 100267421030 |
| 総RZ depth | 51452635379 | 50884473032 |
| 総CX | 64654784832 | 63485782056 |
| 総CX depth | 57938363120 | 57037719456 |
| 総depth | 111882019897 | 110289593414 |
| 総circuit size | 193844506833 | 189812628308 |

数値は今回の `result.json` に保存されたfieldの転記。Energy fieldは宣言した数値pilot証拠に条件付けた十分条件で、厳密包囲・実機CA達成を意味しない。全26 roundでfixed-N statistical marginは正だった。測定標本を生成した実験ではなく、`quantum_shots_drawn=0`、表のshotsは必要予定数である。

各roundのNはX/Yそれぞれ共通で、次の保存結果だった。

| m | CDF N/axis | XDF N/axis |
|---:|---:|---:|
| 0 | 511 | 511 |
| 1 | 512 | 512 |
| 2 | 512 | 512 |
| 3 | 512 | 512 |
| 4 | 512 | 512 |
| 5 | 512 | 512 |
| 6 | 513 | 513 |
| 7 | 514 | 514 |
| 8 | 516 | 516 |
| 9 | 520 | 520 |
| 10 | 530 | 529 |
| 11 | 549 | 548 |
| 12 | 590 | 589 |

**新ideal laneと旧numerical synthesis laneの各round整数N・shots合計・六指標総量は一致した。** 事後処理は保存fieldの一致だけを確認し、旧評価を再実行していない。誤差仮定が違っても整数Nが変わらなかった結果であり、旧合成誤差が実測0だったとは解釈しない。旧numerical結果を新laneで上書きしていない。

二候補間の条件付き総費用比較はXDF lower。各価格区間は保存決定論gate価格に基づく点区間で、統計CIではない。PRの価格値は本runの入力でも比較対象でもなく、新しいPR/XDFの差・比・CI・有意性検定・同等性・非劣性は計算していない。PRの有限bit価格からideal-law価格への移送資格も追加していない。

## 4. Call内訳・実行・資源

| 処理 | 完了logical calls |
|---|---:|
| Saved bundle load | 1 |
| Candidate gate | 2 |
| Ideal round accuracy | 26 |
| Ideal energy | 2 |
| Round serial cost（X/Y pair） | 26 |
| Ideal aggregate | 2 |
| CDF/XDF comparison | 1 |
| Result summary | 1 |
| **合計** | **61 / 61** |

Journalはattempt61、COMPLETE61、FAIL0。STOPは `failed=false`、`first_failure=null`、retry/resume 0。新fit・U最適化・DF/PF再計算・ground eigensolve・array decode・build/compile・synthesis再計算・PR再計算・新統計推定は各0。旧86-call runnerを再実行していない。

専用worktree内で次の一回だけを起動した。Fresh run01は既に存在し、retry/resumeは禁止されているため、再実行手順としての提示ではない。

```bash
PYTHONDONTWRITEBYTECODE=1 \
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
scripts/run_research_c_xdf_cdf_common_ideal_lane.py \
--contract docs/research/research_c_xdf_cdf_common_ideal_lane_contract_unapproved_20261103.json \
--execute
```

既存evaluation venvのPython3.11.1と固定9 package pinsを使用。事前にaffinity32 CPU、MemAvailable約54 GBを確認し、resource_planの `quota_cpus` はnullだった。主処理は小さいscalarの独立二候補chainで、既存resource_planの2 workersを使用。Dense行列/GPU workloadはなく、人工的CPU/RAM/worker/wall上限や共有設定変更は追加していない。

[run resource record](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/resources.json)はwall 1.5176597088575363秒、CPU 9.799995550999999秒、peak RSS 346292 KiB。Import等を含むCLI全体の `/usr/bin/time -v` はwall 1.82秒、peak RSS 347728 KiB、exit 0だった。[CLI log](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/execution.log)。比較測定による高速化率の主張はない。

## 5. 技術確認・保全・停止後の担当

事前のpassive照合と実行後のbyte照合で、契約・SOURCE84 files・保存JSON30件は一致。保存candidate packetも元公開packetと一致した。実行SOURCEの既存tracked file変更は0、NPZ/QPY/NPYの新規出力0。実行後に科学関数を再度呼び出していない。準備時の人工・mock・回帰69 unique PASSを同じSOURCEの技術証拠として再利用し、今回はtest再実行0件。

旧科学SOURCE・raw結果・元dirty checkout・旧数値合成lane・既存契約/grant/ledger/marker/STOPは変更していない。新規の実行出力・照合記録・本報告だけを独立branchへ収録する。Codex-Aのlive worktree/results/process/cache/lockへアクセスせず、途中結果による条件変更、job停止・再起動、共有venv/cache/設定変更も行っていない。

共有PROJECT_MAP・研究概要・Evidence Map・VALIDATION_STATUS・validation manifestは、指定どおり後続統合担当の更新対象として残した。今回の結果は条件付きideal laneの計算完了であり、最終総費用評価・保証・論文化GOの認定ではない。Actual CA、厳密DF/PF enclosure、全law signal/synthesis、hardware実時間・FT費用、分子/rank/r一般性は未資格のまま。

**次の担当は統合Codex**：公開固定された本CDF/XDF結果と、別途固定公開されるAのendpoint結果について、SOURCE・lane・候補identityを区別してEvidence Map等へ接続する。Aの途中結果や未固定identityは使わない。**ユーザー／研究C**：PR対XDF/CDFの統計比較等が必要なら、科学的な問いと対象を別途決め、必要な新科学実行・重要レビューをそれぞれ明示許可する。

今回の担当は公開commit/push後にremote SHA/tree・公開本文・主要リンクを照合し、mandatory STOPする。新しい科学計算・統計比較・別条件探索へ自動進行しない。
