# HANDOFF：効率的controlled ZZとP1（run01 STOP）

参照：[REPORT](REPORT.md)、[PLAN](PLAN.json)、[run01/STOP](run01/STOP.json)、[RUN02_PLAN（未実行）](RUN02_PLAN.json)。

## ユーザー
- P1 run01は、compileの前にrunnerの不具合（planの`source`欄の欠落）で、全4 pathがSTOPした。compileは0件、retryもしていない。
- 技術修正とtestを公開済み。科学範囲は同一（同じPLAN・seed・4 path・最大8 compile）である。
- **run02を実行するかどうかの指示が必要**。指示を受けたら、Dは`RUN02_PLAN.json`に従って実行する。

## Codex C
- 新しいlowering（`CONTROLLED_ZZ_CX_CRZ_CX_V1`）は、構造S（`Accounting`の`'crzz'`＝CX·CRZ·CX）と同じ分解である。
- 既存のK0/K1・E[S]は、そのまま利用できる。今回、C側の作業は不要。

## Codex B
- 作業は不要。
- 将来の価格recordでは、旧lowering（`CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424`）と新lowering（`CONTROLLED_ZZ_CX_CRZ_CX_V1`）を別のSOURCE/identityとして扱い、混同しない。
- 旧価格（C1、長時間研究、Full-H）は旧loweringのままで、上書きしない。
