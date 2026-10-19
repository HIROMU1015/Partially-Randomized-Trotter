# H4 tol-only DF10 決定論PF検証D：完了結果（2026-10-19）

**COMPLETE_MANDATORY_STOP。** ユーザーの「次の作業に進んで大丈夫」を、直前に提示したD190候補・最大372新規軸・one-shot・自動retryなしの承認として固定し、Dだけを一回実行した。native exit0、全worker回収、終了監査PASS。A50のseed割当・最終seal・科学実行は開始していない。

## 固定条件・SOURCE

Phase 0 commit `80652ab6b42cf0c11ee97485214027f507b1bf0a`、承認待ちseal commit `470ee15bd9e1871c1e2bab81ea703a2b029d1c3e` を継承し、科学SOURCEは **`84db3909639b34ec521004c3d8f7ef9b5e6ad851`** のまま実行した。SOURCE17・入力774・provider251・runtime34,273のbyte SHAを開始前と終了後に照合した。CPU/RAMの観測と承認receiptを追加した新実行契約を作り、旧seal・INVALID grant・過去のgrant/ledger/marker/STOPは保持した。

H4 linear・隣接1.00 Å・STO-3G、保存済みtol-only threshold1e-8のDF rank10、system8 qubits＋ancilla、Nα=Nβ=2のsector36、T=.8、P=0。通常signed bias、allowance倍率1、ε=.05/.01/.005/.001/.0005。q=1/2/4/8/16、δ=.8/.4/.2/.1/.05。B0/B0I prefix1〜9、full B1 prefix10、二次/四次で190候補。

Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing=None、symmetric_directional control、元constant＋tail identityを一度だけ、reference無回転を維持。Hamiltonian/state/snapshot/compiler hashは[Phase 0報告](track_a_h4_tol_main_phase0_20261019.md)と同一。H4 dense eighとH6/H8 eigsh、正規化・残差判定の差も記録したまま保存状態を再利用した。新しいSCF・integral・DF・保存state生成・full-H reference計算は0。

## 登録内の最小PF

190候補×5精度＝950 endpoint行を一回会計した。129候補が少なくとも1精度で適格となり、同一条件のFP4候補8軸を再利用し、残る125候補250軸を直接compileした。全適格候補の費用が揃った後に、exact費用・同値ならcell_id辞書順で選んだ。欠測を不適格として除外していない。

|ε|登録内の最小PF|G_RZ（必要shot×full wrapper compiled RZ）|
|---|---|---:|
|0.05|`B0I_p5_q1_2nd`|70,108,214|
|0.01|`B0_p7_q2_2nd`|5,203,661,250|
|0.005|`B0_p7_q4_2nd`|33,735,246,876|
|0.001|`B0I_p7_q2_4th`|1,446,021,078,216|
|0.0005|`B0_p7_q8_2nd`|7,856,558,002,614|

上表は実行済み[result.json](../../artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/result.json)の保存winnerを転記した。costは固定compilerの各cosine/sine full wrapper実測RZに、そのendpointの必要shotを掛けた和である。実際にその回数の量子shotを測定した結果ではない。各回路の費用取得は決定論的n1で、PRのn32や新Aのn8を意味しない。

これは固定した190候補、δ窓、保存状態、通常signed/倍率1のengineering適格性の下での最小費用であり、global optimum・厳密bias certificate・PRに対する優位性・分子サイズだけの効果を認定する結果ではない。H6/H8とのsolver・compiler profile来歴・探索領域の差は残る。A50が未実行なので、新しいPR/PF順位や費用比を追加していない。

## 実行範囲・並列化・終了監査

|項目|実行結果|
|---|---|
|新規base signal / 解析overlay|96 / 88。保存6 signal recordを再利用|
|新規compile|125候補・250軸（承認上限186候補・372軸以内）|
|保存FP費用再利用|決定論4候補・8軸。PR n32 cohortはDに混ぜない|
|control|125候補・500 probe、全PASS|
|primitive|365 time pair×3＝1095 probe。負のYoshida時間を含む全登録coverage|
|preparation|10 reconstruction。保存入力は再生成しない|
|会計|190×5＝950行、1 pass|
|並列化|fresh許可物理CPU64基、最大64 worker、1 worker 1 thread。計221 native worker processを起動・回収|
|資源|開始時実効空きメモリ約469 GiB。既存affinity/quotaを尊重し、恣意的CPU/RAM/worker/wall上限なし|
|壁時計時間|1017.587秒（約16分58秒、witness記録）|
|終了|native exit0、全worker同一親回収、STOP・中断・retryなし|
|保全|旧774レコードと、新しい実行2033レコードをSHA保存|

構成間並列、fork/COW、preparation/native cache、paired evolution共有、fingerprint再利用を使用した。q16四次PFの最後の軸まで待ち、PSIや他jobとの一時的競合を理由とする停止・再起動は行わなかった。共有設定・package・他process・他worktreeを変更していない。

終了監査は[metadata専用auditor](../../scripts/audit_h4_tol_d_completed.py)で行った。候補集合・適格性・軸ごとの予約と実測・compiler/fingerprint、control、worker exit/reap/SHA、保存shotとRZの費用恒等式、記録済みwinnerの最小性と同値規則を検査した。科学accounting・signal・sampling・compileを再dispatchしていない。監査は著者Codexによる確認で、別reviewerによる独立レビューではない。実装の人工40件PASSは[seal時の報告](track_a_h4_tol_d_seal_20261019.md)に保持した。

結果SHA256：`106469f5a604af01ff2cb77012a736646df3291abe60a7c870e77785058ec8a3`。

## 保存証拠と次の段階

[終了監査](../../artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/completion_audit.json)、[native exit](../../artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/native_exit.json)、[SOURCE/契約/承認/資源binding](../../artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/execution_binding.json)、[承認receipt](../../artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/approval_receipt.json)、[control](../../artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/control_validation.json)、[worker回収](../../artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/worker_receipts.json)、[artifact SHA](../../artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19/artifact_manifest.json)。signalsとpaired測定値は同directoryのgzipに保存し、展開bytesが私有原本と同一であることを照合した。実行契約・grant・ONE_SHOT・原本は `.h4_tol_main_private/D_seal01_20261019/` に保持する。

結果commitは既存branch `codex/h4-tol-only-main-phase0-20261019` にpushし、remote branch SHAとの一致を確認する。確認したSHAは最終チャットと私有publication receiptに記録する。このDの公開は、H4/H6/H8の最終evidence freezeを意味しない。

**A50はPhase 0のまま**で、FP1候補n32と新規49候補n8の区別を維持している。新seed割当・旧seedとの最終衝突検査・最終SOURCE/入力/call/契約・A科学実行は未実施。S_STATS/Pも開始していない。Dについて未解決の技術異常はない。

次の作業：**ユーザー**がD結果を正式に受け入れ、A50の最終sealを指示する。条件が成立した後に限り、**Codex**がAのseed・衝突検査・契約・人工/拒否検査を完了し、commit・push・remote SHA確認後にSTOPする。**A50の実科学計算は別の明示承認が必要**。現在はmandatory STOPで、A/Pを自動開始しない。
