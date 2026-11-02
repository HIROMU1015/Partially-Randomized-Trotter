# GPT review：RA-D0数値baselineと実行契約

利用者formal design §16の準備を実施した。
[source review](ra_d0_source_preparation_review_v1.md)、[原文](inputs/ra_d0_formal_design_v1_user_input.md)、
[manifest](../../../artifacts/track_b_ra_d0_preparation/2026-10-06/evidence_manifest_v1.json)を確認する。

**`REVISE_RA_D0_NUMERICAL_BASELINE_AND_EXECUTION_CONTRACT`。**
21 columns/x、18 sign pairs一致、保存126 sequencesのidentity照合、35 focused tests PASS。
registered optimization=0、新synthesis/science/circuit/trajectory/GPU=0。
R1/R1.5/旧STOP・Track Aは不変。資料公開は実行認可ではない。

GPTへ戻す決定は次の三点。

1. **数値baselineの入れ子規約。** ideal B0→B1→B2→B3は成立するが、
   midpoint lawのB0_savedはB0_idealと同じではない。
   B1のirrational group ratioと全qのdyadic規則はexact membershipでは両立しない。
   B0_savedをreproduction anchorとして保持し、B1/B2の固定丸めに由来するmembership許容幅と
   それを含むB2 outer certificateを採用するか。
2. **結果から逆流しないbudget/certificate規約。** B2 minimaをcertified feasible upperでbudget化し、
   B3前にfreezeするか。solver infeasible flagは証明書にせず、
   B3-only feasibleはsecondary保存、uncertified failureはtechnical inconclusiveとするか。
3. **実行scope／計算資源上限。** 原文全gridは737 points、最大444,411 primary/minimum LP呼出し
   （各infeasibilityに一Farkas補助LPを加える保守上限888,822）。
   これを採用するか、結果前にquery scopeを狭めるか。wall/CPU/RSS/output等の上限も決める。

integer-shot gridの1.005保証は確認できた。今回全cost minimaがpositiveでupper boundも構成できる。
grid／anchors／query recipeは保存したが、B2 minimum値、実budget値、最適解、witnessは未取得。
実装のB2 outer LPはlower-bound用までで、dyadic B2 primal membershipは未確定規約により未完了。
登録domainへのsolver呼出しは拒否する。authorization/marker/optimization runnerは作成していない。

このレビューは全面的な研究方針採択やRA-D0 positive/negative判定を求めるものではない。
三点を閉じたsource review後にだけ、別のsaved-table development one-shot authorizationへ進む。
全outcomeでSTOPして研究判断をGPTへ戻すという原文の運用は維持する。
