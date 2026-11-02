# H4/H6標準Yoshida4の条件付き独立費用方向：正式採用

2026-10-24、今回のユーザー指示に基づき、固定科学条件・固定shot政策に条件付けた独立PR回路費用方向の再現を正式採用した。
[採用記録・原典・全保存値・限界・保全](../../artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/REPORT.md)。

linear H4/H6・隣接1Å・STO-3G・tol-only DF1e-8/rank10/19・T=.8・ε=.001・α_axis=.025・P=0・usual signed。
保存signal/reference・empirical allowance・analytic shots、状態準備を除くsymmetric-directional full cos/sin Hadamard wrapperのcompiled RZ総費用。
PR prefix3/5・K4・ABS_TIME_PER_OUTER32のS2/q8対標準Y4/q4で、独立S2 n25/23と新独立Y4 n45/32を比較した保存結果を採用した。
保存G比Y4/S2はH4 `0.8649598948712324`、H6 `0.8627624094690314`。
固定2比較family・α=.05のBonferroni正規近似区間は差の正側。元判定 `REPRODUCED_APPROXIMATE` を保持する。

同じεの保存FullはS2/q8低費用、独立PRはY4/q4低費用。
異なるqの固定構成の方向逆転として扱う。同一q対照は両系ともq4でFull/PR双方Y4、q8で双方S2が低費用で、逆転なし。
探索、既存独立S2、新独立Y4をpoolせず、旧H6 K2/S2やm5_best別候補を今回の標準Y4へ転用しない。

H2/H4/H6の精度依存の観測とC/N/G会計の保存値整合性は既存batchを参照する。
q、signed piece、RTE配分、partition、bias、normalizationが関与し、次数単独の因果効果は未確定。
母平均・希少tail・厳密coverage/bias・全候補最適性・一般化・独立新規性・論文中心主張の採択は未認定。

採用版の入口は[paper manifest](../../artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/paper_manifest.json)。
f75版CEM21行/台帳391行を保持して2主張と14参照を追加し、3環境索引へは同一IDの差分を接続した。
元結果 `4d117041469fb8eebf2ad55405c0f7d1b70dea7d`、科学SOURCE `a2a53698a72da7280d217adc7931938121ed732d` と旧失敗履歴を保全。
主表・Figure2・旧科学契約/STOPは不変。今回の科学計算・数値再解析・grant0。

mandatory STOP。旧strict COMPAREはBLOCKED。詳細科学レビュー・最終freeze・大幅改稿を開始しない。
次担当：ユーザーが必要時に論文中心主張・新規性の採否を判断する。Codexは停止を維持する。
