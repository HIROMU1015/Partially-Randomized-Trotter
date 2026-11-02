# E/T v6：数値guardの観測と会計不整合の特定

S6 `90ce36e3fc383fcb4988c4df86328a08b0fef789` → A6 `3462c2b436f43d495bf52d999794b3dcdb701cf1`から一回実行。TECHNICAL_INCONCLUSIVE、retry0、marker消費済み。
原292file/37,120,020bytes、native10/T0を保存。source/contract/manifest963対象一致。

r6 controlled Aのcombined upperは約0.0981748145。
うちnearest W（π/4）phase distanceは約0.0981747704、他の四項の合計は約4.41e-8。
raw Operator residual upperは約2.99e-12、forward guard1.63e-8、reference radius2.79e-8。
この回路のscalar phaseはunsupportedで、W exponent=None。native IRはraw phaseを保持している。
旧guardは、実施しないnearest-W置換の距離まで加算していた。

技術修正は、raw phaseを保持するnative回路と、対応する場合だけ行うW loweringを区別する。
strict full Operator比較・実行位相・numeric2e-7・誤差保証・compiler/capsは変更しない。
unsupported scalar phaseのT費用は既存T_INCONCLUSIVEのまま、known lower/actual totalを捏造しない。

これは旧結果の再分類ではない。v6 raw失敗値/markerを保持した新versionへ進む。
利用者は技術修正・新source実行を計算完了まで継続認可している。
既存contractのnativeとT会計を整合させる修正として、off-domainの位相保存/誤位相検知/unsupported Tテスト後に固定する。
source末尾のmetadataだけの修正とは区別し、全差分と新guard会計を明示する。
科学条件、threshold、実行位相を変更する修正は行わない。

数値のexact rational、IR hash、raw graphとouter statusは新audit/result namespace v3を参照。
科学的A/B勝敗は未判定。G10/RA-D0/分子/GPU等へ進まない。
