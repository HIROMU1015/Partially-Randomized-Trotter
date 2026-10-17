# H8 経験的非相殺bias・必要shot診断：run01完了（NONCANCELLATION_SHOTS のみ）

**承認された `NONCANCELLATION_SHOTS` stageを一回実行し、正常終了した（native exit0、修正・再実行0）。全184決定論候補とPR確認union6の1,900行・380軸を評価し、未解決0。保存通常会計1,900行は凍結D0 ruleで完全再現した。**
非相殺条件では、ε=.05で16候補（旧32の3候補、追加152の13候補）が通常biasでは適格から不適格に変わり、ε≤.01の適格性は通常biasと同一だった。逆向き（不適格→適格）は0。費用比較・ランキング・PR削減率は計算していない。

- SOURCE：`cbfd7cacad8a5828c0dbfd284a99ef0871860e66`（GitHub branch tip `7a6227f5…` の親として照合）
- Plan v2 digest：`002f9edbf35506fca256431df641552d7c319581f306f13a316b736e07897e6d`（未承認plan・無効grant v2は不変のまま保持）
- 実行ID：`h8-noncancellation-shots-20261017-attempt01`、新しい一回限りgrant（`NONCANCELLATION_SHOTS` だけ、費用stageは不許可）
- [結果bundle](../../artifacts/resource_applicability/track_a_h8_noncancellation_shots_run01/2026-10-17/README.md)／[準備と修正](track_a_h8_noncancellation_readiness_20261017.md)

H8直線鎖1.00 Å/STO-3G、保存tol-only DF28、16 spin modes/sector4900、T=.8、q1/2/4/8。決定論184候補368軸（旧B0+I復元12候補を含むsigned run01 `8e54303b…` の成分）とPR確認union6・12軸（各n32）。ε=.05/.01/.005/.001/.0005、allowance倍率1と10、α_axis=.025、凍結D0 shot算術・normalization・headroom規則。

## bias量の区別

- 通常signed bias `abs(total)`：保存会計の再現gateと比較列にだけ使い、新しい費用比較はしない。
- 経験的非相殺bias：B0/B0+I `abs(discard)+abs(PF)`、full PF `abs(PF)`、PR `abs(outer_PF)+abs(finite_RTE)`。分解粒度に依存する感度診断である。
- 厳密bias保証：全出力でnull。保存combined u×倍率は軸ごとに一度だけ使い、成分へ重複加算・二乗和縮小をしていない。real/imagは別々に判定した。

## gateと照合

|gate（shot stage）|結果|
|---|---|
|入力22 roleのbytes/SHA、Phase A gzip原bytes、来歴join、確認seed再導出|PASS（終了時に入力・SOURCEを再照合）|
|保存通常会計1,900行の凍結D0 rule完全再現|1,900/1,900一致|
|PR normalization `log_B == R*log(b)`|6/6一致|
|PR成分・端点・closure・forward・非回転reference（rel/abs 1e-12）|6候補×14検査＝84 PASS。最大差：forward corrected 2.65e-14、reference 2.11e-15、他0|
|`PR_saved_paired_metric_reproduction`・`PR_confirmation_n32_pairing`|費用比較stageのgate。今回 `executed=false`（`NOT_EXECUTED_IN_NONCANCELLATION_SHOTS_STAGE`）|
|S4 model|`UNVALIDATED_B2_FULL_PASS_NOT_BOUND_S4_MODEL_NOT_USED`|

## 適格性（倍率1と10で件数は同一）

|ε|旧32 通常→非相殺（適格/不適格）|追加152 通常→非相殺|PR6 通常→非相殺|
|---:|---|---|---|
|.05|24/8 → **21/11**|132/20 → **119/33**|6/0 → 6/0|
|.01|13/19 → 13/19|92/60 → 92/60|6/0 → 6/0|
|.005|12/20 → 12/20|78/74 → 78/74|6/0 → 6/0|
|.001|6/26 → 6/26|54/98 → 54/98|2/4 → 2/4|
|.0005|6/26 → 6/26|51/101 → 51/101|2/4 → 2/4|

全1,900行の遷移は、適格→適格948、適格→不適格32、不適格→不適格920で、不適格→適格・未解決（UNDETERMINED/guard）・LOG_DOMAIN_ONLYは0。
ε=.05で不適格に変わったのは次の16候補（両倍率）：旧32の `B0I_p14_q1_2nd`、`B0_p14_q1_2nd`、`B0_p14_q2_2nd`、追加の `B0I_p8/9/10/11/12/13_q1_2nd`、`B0I_p9_q2_2nd`、`B0_p10/11/12/13_q1_2nd`、`B0_p13_q2_2nd`、`B0_p13_q4_2nd`。

## 必要shot

両条件で適格な948行の非相殺/通常の総shot比は、最小1、中央値1、最大46.2（旧32の最大24.9、追加152の最大46.2、PR6は最大1.00004）。
380軸の非相殺bias/通常biasは、最小1、中央値1.004、最大33.0。245軸で非相殺量が大きく、135軸で等しい（full PFの16軸は定義上等しく、B0 56・B0+I 56・PR 7軸は成分が同符号または一方が0）。
候補別のshot・headroom・u/headroomは[1,900行CSV](../../artifacts/resource_applicability/track_a_h8_noncancellation_shots_run01/2026-10-17/accounting_rows_v1.csv)、軸別の量は[380軸CSV](../../artifacts/resource_applicability/track_a_h8_noncancellation_shots_run01/2026-10-17/axis_envelopes_v1.csv)に保存した。
これらは保存DFモデルのfinite-time signalに対する経験的な感度診断であり、真の物理biasの上界や必要shotの保証ではない。

PRの保存Taylor次数histogramは確認0次3,580/2次4、**4次は未観測**。未観測層の費用はnull、母平均上界は未証明のままで、費用0とみなさない。探索n8は使っていない。
来歴限定（prefix19の元runtime SHA欠測16候補、共有reference/scalar、旧native projection8/12、full PF旧profile unknown、経験的u非証明）は結果にそのまま保持した。

## 実行・資源・非干渉

固定Python3.12.3（D0と同じ実行環境hash）、1 process・child0。native wall 0.508秒（runner本体0.281秒）、CPU user 0.460秒/system 0.042秒、peak RSS 57.2 MiB。launcherがwait4で直接回収し、native exit0。PID不在を確認し、one-shot lockを保持した。試行1、技術修正0、再実行0。
実行前のfresh gateで、SOURCE（Git object一致）・環境・入力SHA・空きメモリ（約464 GB available）・PSI0・disk/inodeを確認した。並行して動いていたB2（sine-psi-observation attempt01、CPU1〜4）と重ならないよう、自分のjobはCPU8〜63に置いた。
実行中のPSIサンプルでは、host・user.slice・自user・sessionとも最大avg10が0、累積stallの増分も0だった。B2 processは前後で不変、resource_stopは発生していない。
保護snapshot8件の保護file変更は0。元準備worktree（HEAD `d567e6e1…`）とB2のSOURCE・結果・STOP・grant・ledger・markerは変更していない。検索は対象fileに限定し、広域の再帰走査はしていない。

## 公開と次の担当

結果は独立cloneの専用branch `track-a-h8-noncancellation-shots-result-20261017`（親 `7a6227f5…`）へcommitした。有効grant・消費済みlock・private入力・原結果JSONは非公開領域に保全し、公開物はdigestとprojectionだけとする。

**ユーザー**：結果を確認し、費用比較stage（B2全4軸PASSの結合が前提）へ進むかを判断する。
**Claude Code**：別途承認があった場合に限り、その範囲で対応する。今回はmandatory STOP。
