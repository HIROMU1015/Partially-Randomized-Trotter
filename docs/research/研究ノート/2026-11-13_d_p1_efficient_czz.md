# 2026-11-13：効率的controlled ZZへの移行とP1 run01（Codex D）

## 確認済み（技術）

- PR・Full-H RTEのcontrolled ZZを`CX·CRZ·CX`で生成する新しいSOURCEを作った（lowering ID `CONTROLLED_ZZ_CX_CRZ_CX_V1`）。
  - provider builderのsubclassとして実装し、明示的に切り替える。旧providerのfileは変更していない。
- 技術test 44件がPASSした。確認した内容：
  - unitaryの一致（global phase・制御phaseを含む）
  - 対象外のcaseを元の実装へ委譲すること
  - 切り替えと復元
  - tailのparity
  - PR・Full-Hの経路が呼出し時にbuilderをimportすること
  - 旧providerのbytesが変わっていないこと
  - 固定PLAN

## P1 run01（STOP）

- 固定PLAN（`bb475af8`）で実行した。
- phase A（4 pathのseed照合とtail parity）は通過した。
- phase Bで、固定C1 `Backend.price_requests`がplan["source"]を要求したが、planにその欄がなく、全4 pathがKeyErrorになった。
- compileは0件。sampling済みのevent・seed・lawはprivateに保全した。retryはしていない。

## 判断

- 技術修正（実行planへのsource付与、plan keyの事前検査、PHASE_A/STOPの先行保存）を公開した。
- RUN02_PLANは未実行。再実行にはユーザーの指示が必要。
- [REPORT](../d_p1_efficient_czz_20261113/REPORT.md)
