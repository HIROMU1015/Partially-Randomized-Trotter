# H4 tol-only DF10 A50 最終技術seal（2026-10-19）

**A50の最終技術seal完了、実科学計算の明示承認待ち、mandatory STOP。** ユーザーはD完了commit `406f70570323366da185355550a3900388717fbb` を正式に受け入れ、今回のA契約準備と公開を承認した。A sampling・実H4 compile・保存値の新統計解析・S_STATS/P・新しい科学レビューは実行していない。有効grantは発行していない。

## 固定した条件と再利用

科学SOURCEは **`84db3909639b34ec521004c3d8f7ef9b5e6ad851`** のまま。Phase 0の50候補、既存のlazy adapter、保存tol-only DF状態、forward/native/finite-RTE kernelを再利用し、科学kernelや候補集合を変更していない。metadata専用の[seal準備・byte/seed auditor](../../scripts/seal_h4_tol_a.py)を追加した。

linear H4・隣接1 Å・STO-3G、tol-only threshold1e-8のDF rank10、system8 qubits＋ancilla、Nα=Nβ=2のsector36、T=.8、P0。通常signed bias、allowance倍率1、ε=.05/.01/.005/.001/.0005。q1/2/4/8/16（δ=.8/.4/.2/.1/.05）。既存B2二次、prefix1/2/3、r4、K2/4の30候補と、中央prefix2のr2/8・K2/4の20候補を固定した。H4 dense eighとH6/H8 eigsh、正規化・残差判定の差は[Phase 0記録](track_a_h4_tol_main_phase0_20261019.md)のまま保持し、保存状態を再生成しない。

Qiskit1.3.0、rz/sx/x/cx、optimization1、seed_transpiler17、backend/coupling/layout/routing=None。symmetric_directional control、元constant＋tail identityは一度、reference無回転、元の無条件finite-RTE lawを維持する。

| cohort | 候補 | 標本数 | 今後の扱い |
|---|---|---:|---|
| 保存FP anchor | `B2_p3_q4_2nd_r4_K4` 1候補 | n32 | 保存64軸・signal・mean/SE等をそのまま利用。raw費用の再集計なし |
| 新規EXPLORATION | 残る49候補 | 各n8 | signed/倍率1で少なくとも1精度が適格な候補だけ取得。最大392 trajectory/784軸 |

両cohortをpoolしない。新規候補の不足・不明な適格性・欠測費用を除外に使わない。固定集合内の選択規則だけを将来の承認実行で適用し、今回の結果依存再選択は0。

## 受入済みD

D結果SHA256は `106469f5a604af01ff2cb77012a736646df3291abe60a7c870e77785058ec8a3`。commitのGit blob・保存結果・終了監査のSHA一致を確認し、5精度の保存winnerを[そのまま転記](../../artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19/accepted_D_saved_winners.json)した。

| ε | 保存済み最良PF | 保存G_RZ |
|---|---|---:|
| .05 | `B0I_p5_q1_2nd` | 70,108,214 |
| .01 | `B0_p7_q2_2nd` | 5,203,661,250 |
| .005 | `B0_p7_q4_2nd` | 33,735,246,876 |
| .001 | `B0I_p7_q2_4th` | 1,446,021,078,216 |
| .0005 | `B0_p7_q8_2nd` | 7,856,558,002,614 |

費用・shot・bias・順位を再計算していない。上表はD190の登録内engineering比較であり、global optimumや厳密bias保証ではない。新PR/PF比較はまだ存在しない。

## seed契約と衝突監査

namespace `H4_TOL10_MAIN_V1_20261019_EXPLORATION`、SOURCE・plan digest・input binding・cell_id・replicaのcanonical SHA256先頭64 bitからmasterを導出し、元の`direct_master_to_step_seeds_v1`を維持した。392 masterと2,368 step、計2,760整数を私有registryに固定した。cosine/sineは同じtrajectory、候補間CRNなし。

既知67,510明示整数と衝突0、内部重複0。さらに6 geometry・登録1,308候補中のstochastic1,164候補について、n32のmasterと全order/component予約slotを含む保守的superset **7,737,216 slot** を6自然単位で並列照合し、衝突0だった。Taylor次数・component・RTE eventは抽選していない。geometry streamの件数と順序SHAは過去の固定監査とも一致した。

旧H4 legacy・探索A・固定n32・conditional B・T=.4/1.6・FP、rank12 Pの予約/実使用4,185本を照合した。並行研究はH6 legacy/A128・固定δ予約・conditional・高次PF探索/確認の予約と使用、最終確認152本、H8 legacy/A128・新PR確認・conditionalを保存registryから読み取り専用で追補した。停止済み/未使用の予約seedも除外した。[seed監査projection](../../artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19/seed_inventory_projection.json)に各scopeのSHAとcutoffを保存し、seed整数自体は公開していない。

対象はcutoff `2026-10-19T04:26:49Z` 時点の既知登録campaignであり、未知・未来のjobまで網羅した主張ではない。後続campaignはこのregistryを除外する。科学起動前は、追加登録されたcampaignとの衝突だけを再照合し、このseedを引き直さない。衝突ならSTOPし新契約として扱う。seed整数の相違は数学的なPRNG独立性の証明ではない。他worktreeのHEAD/status/資料更新を実行拒否条件にしない。

## 固定call予算・並列化・終了監査

| 新規科学call | 最大値 |
|---|---:|
| base signal / signed shot endpoint | 49 / 245 |
| independent forward signal / tail matrix | 49 / 49 |
| trajectory / native evolution | 392 / 392 |
| occurrence / RTE event | 2,448 / 10,784 |
| cosine/sine wrapper compile | 784軸 |
| control probe / independent event oracle | 196 / 49 |
| preparation reconstruction / primitive probe | 3 / 120 |
| 新cohort endpoint会計 | 1 pass |
| SCF・integral・DF・state solver・full-H reference新生成 | すべて0 |

これは上限契約であり、実科学callの実績はすべて0。FPの保存64軸は新規784軸に含めない。実行時は49新signalを並列化し、適格な候補のreplica0のcontrolとoracleを全てPASSした後、残りreplicaをtrajectory間で並列化する。cosine/sineは既存kernelの順序を維持する。3 preparationのcache、fork/COW、paired evolution共有、fingerprint再利用を使う。

CPU ID・pool sizeは科学承認後のfresh affinity/物理CPU・既存quota・実効RAM・既知稼働割当で決める。今回はaffinity64を観測しただけで、割当は空、admissionはfalse。CPU/RAM/worker/wallの恣意的な上限なし。worker内thread1と外側poolで重複並列を避ける。共有設定・他job・他worktreeを変更せず、PSIは診断のみとする。

終了にはnative exit0・同一親wait、全worker exit0/回収、固定registryの適格subset各n8・両軸・event/seed/law/preparation/compiler/fingerprint一致、control/oracle、非払い戻し予約と実測ledger、250 endpoint行、FP保存統計の不変、SOURCE/input/provider/runtimeと保全SHAの再照合が必要。終了auditorは科学統計を再dispatchしない。不利な科学結果は技術失敗扱いにせず保存する。中断/失敗時の自動retry、追加sampling、次stage自動開始は禁止する。[並列・終了契約](../../artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19/parallel_and_completion_plan.json)を参照。

## 固定証拠・検査・次の承認

SOURCE17 file・入力826行・provider251 file・runtime34,273 file・Python binaryをbyte SHAで照合した。入力は元774行にD受入結果・seed registry・除外在庫・scope来歴・承認境界等を追補したもの。旧774保全レコード、完成D2,033レコード、公開D11 fileもSHA不変。

私有契約digest：`4d84da109de9e2f4d1a10ee79e65955bb64bdb9d67010295bb656ec22b4490f2`。[実行契約projection](../../artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19/A_contract_SEALED_PENDING_APPROVAL.json)にSOURCE/input/provider/runtime/seedのbinding、[INVALID grant](../../artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19/A_grant_INVALID.json)にone-shot・retry禁止・契約digestを固定した。私有原本は `.h4_tol_main_private/A_seal01_20261019/` に保持する。承認後の有効化は、新しい承認receipt・fresh資源receipt・execution契約・grantを作り、今回のpending契約とINVALID grantを上書きしない。

人工/回帰は **96 PASS**（A seal14＋D seal8＋D/A adapter32＋既存DP29＋followup13）。followupのcompileは人工小Hamiltonianのみ。実H4を使っていない。凍結A kernelのlazy import、seed欠落/改変/衝突・不完全receiptの拒否、無効grantのdirect/witness実CLI2経路の科学import前拒否もPASS。Aのoutput/ONE_SHOT/witness/科学workerは未作成。[テスト証拠](../../artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19/test_report.json)、[終了監査](../../artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19/completion_audit.json)、[artifact SHA](../../artifacts/resource_applicability/track_a_h4_tol_a_seal/2026-10-19/artifact_manifest.json)を保存した。作者自身の技術検査であり、別担当の独立科学レビューではない。

未解決の技術異常はない。残るのは別のA科学実行承認、起動時のfresh資源割当、新登録seedの追補照合。Aの適格性・新費用・順位は実行前なので未確定である。

次の作業：**ユーザー**が上記SOURCE・seed binding・各n8・最大392 trajectory/784軸・one-shot/retry禁止を確認し、A50科学実行を別途明示承認する。承認後に**Codex**がfresh gateと新grantを固定して一回実行し、終了監査・保存・公開後に再びSTOPする。S_STATS/Pと新しい科学レビューは今回開始しない。現在はmandatory STOP。
