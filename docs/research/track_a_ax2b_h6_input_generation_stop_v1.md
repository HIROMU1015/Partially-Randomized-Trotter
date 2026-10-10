# Track A AX-2B：H6入力生成一回実行・DF検査STOP v1

2026-10-10 JST。**認可済みの入力生成を一回起動したが、DF adapterのHermitization検査で停止した。**
計算待ちではない。親/workerの原status `H6_INPUT_GENERATION_STOP`を保存し、再実行・修正は行わない。
H6技術pilot・本検証・H8は未認可。`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
これは科学的GO/STOPの代行や、PRの有効性に関する結論ではない。

## 実行条件と一次記録

linear H6 / 1.00 Å / STO-3G、charge0/multiplicity1、12 spin orbitals、予定Nα=Nβ=3 / sector400。
DF tol-only1e-8・後段cutoff0・Hermitization許容1e-10。actual rank・DF target/stateは未固定。
入力生成にはPF prefix / delta windowがなく、signal・PF・shot・回路資源比較は実施していない。
[seal・専用grant](track_a_ax2b_h6_input_generation_execution_seal_v1.md)でCPU2/worker1/BLAS1、900/300/900秒・total2100秒、AS8GiB/output128MiBを固定。
source `67312f3195aede26e8ba4f5727d89c236772f82e`、seal/grant `affa3f328950a12ac08c11b09085bbed1bfbdb53`。
[起動receipt](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/launch_receipt_v1.json)と[起動一回claim](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/launch_attempt_claim_v1.json)、
[remote seal照合](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/seal_remote_verification_v1.json)でsource183/validation3とgrant bytesを結ぶ。

| 保存事実 | 結果 |
|---|---|
| [親terminal](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/terminal_status.json) | STOP、wall 1.696076954秒、worker exit1 |
| [worker terminal](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/worker_terminal.json) | `ValueError:HERMITIZATION_POLICY:fragment_15` |
| integral build | attempted1 / completed1 |
| DF adapter | attempted1 / completed0 |
| solver matvec / trajectory / occurrence / compile | 0 / 0 / 0 / 0 |
| [SCF/integral receipt](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/integral_receipt.json) | RHF収束記録、conv_tol1e-9、max_cycle50、cycles6 |
| [integrals.npz](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/integrals.npz) | 6 arrays、保存receiptとfile/data SHA・header一致 |
| DF receipt / state snapshot / state receipt | 未生成 |

`df_decomposition` counterはdecomposer内部完了数ではなく、adapterが正常returnした完了数。
保存labelとsourceからdecomposer return後のfragment検査でraiseしたと分かるが、返された分解rawは保存されていない。
最後の[progress](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/progress_0005.json)はDF開始境界までのcounter/RSS記録で、全run peak RSSとは呼ばない。
[worker log](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/worker.log)は0 bytes、[supervisor console](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/supervisor_console.log)はSTOPを記録。
scratchのSCF checkpoint/font cacheを含む全20 raw files・526546 bytesを保存する。

## 保存監査と静的原因経路

[保存run監査](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/saved_input_audit_v1.json)は`SAVED_INPUT_STOP_RECORDED`。
停止記録・grant/manifest/terminalの結合と全raw file hashesを監査し、分子検証PASSへ昇格しない。
[追加integral bytes監査](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/integral_saved_bytes_audit_v1.json)はstdlibで保存NPZ header/raw/receiptを確認。
この監査は新しいSCF/DF/state/residual/signalの計算ではない。

[静的source監査](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/stop_static_source_audit_v1.json)が示す経路は、
[凍結adapter](../../src/trottertracks/resource_applicability/ax2b_h6_input.py) L45のdecomposer呼出し→L56–65の各fragment検査→L60の例外である。
sourceの検査条件は`norm((g+g†)/2-g) > 1e-10`、labelの15はzero-based index。
**実際の差の値・失敗fragment bytes/lambda・actual rank・切断値は保存されていない。**
metadata/df receiptの作成前に例外が上がるため、保存integralsだけからそれらを推定・補完しない。
rank政策・閾値の緩和、係数discard、再Hermitization、decomposer再実行は未実施。
精度不足、分解表現、退化、provider/adapter規約のいずれが根本原因かは、この保存証拠だけでは決められない。

## GPTへ戻す論点と停止境界

[採用済みGPTレビュー](track_a_ax2b_h4_limited_stop_independent_review_2026-10-10.md) §§16,19の早期差戻し条件に従い、
実DF出力と登録Hermitization条件の不整合を、H6技術pilot前にレビューへ戻す。
GPTには、tol-only・係数order/cutoff・DF target・sector/state意味論を保つ診断方針と、
次の一回診断に必要な保存項目を判断してもらう。元integralsを再利用できるが、DF再実行には新source/plan/別認可が必要。
診断候補は失敗前のraw分解/lambda/rank/切断値/補正・Hermitization差の保存であり、今回実装・実行していない。
このSTOPはH6本検証の科学的No-GoやPR優位の否定ではない。
[結果inventory](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/stop_evidence_inventory_v1.json)に欠測と全raw SHAを記録。
N/Gnull・UNDETERMINED・総u/ground-state未認定を維持。sampling/回路build/transpile/compile/H6 pilot0。
grant/outputは一回消費済み。retry/resumeせず、旧H4/source/freezes/dirty・Track Bを保全して停止する。
