# 2026-11-13：効率的controlled ZZへの移行とP1（Codex D）

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

- 固定PLAN（`bb475af8`）で実行した。phase Aは通過した。
- phase Bで、固定C1 `Backend.price_requests`がplan["source"]を要求したが、planにその欄がなく、全4 pathがKeyErrorになった。
- compileは0件。sampling済みのevent・seed・lawはprivateに保全した。retryはしていない。
- 技術修正を`20fe1194`で公開した。

## P1 run02（完了、ユーザーの明示指示による）

- SOURCE `20fe1194`で、同じPLAN・seed・loweringで実行した。
- run01のeventとのSHA-256一致（実行前後とも）と、phase A parityの全件PASSを確認した。compileは8/8件。
- 結果（X＝Y、D＝C_optimized−S）：

  | path | C_optimized | S | D |
  |---|---:|---:|---:|
  | m0/0 | 1,906 | 1,845 | +61 |
  | m0/1 | 1,674 | 1,529 | +145 |
  | m6/0 | 1,826,433 | 1,747,585 | +78,848 |
  | m6/1 | 1,815,119 | 1,750,316 | +64,803 |

- 観察（4 pathの数値。modelではない）：
  - Dはすべて正。m6のDはCの3.6〜4.3%で、eventあたり8〜10 RZ。
  - P0の局所差を参照すると、旧loweringより約12.7万〜14.3万RZ少ない（旧回路はcompileしていない）。
  - DはN_ZZだけでは説明できない。
- 計算負荷：m6の1 pathに約53分かかった。compileは各約4分で、大半は固定Backendのcompose・fingerprint計算だった。

## 判断

- P1は完了。P2/P3、fit、全lawのE[C]、N/RMSE/Gには進まない（mandatory STOP）。
- [REPORT](../d_p1_efficient_czz_20261113/REPORT.md)
