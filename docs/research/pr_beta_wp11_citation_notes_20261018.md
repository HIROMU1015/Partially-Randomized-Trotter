# B02/B06：旧H4 WP11費用表を引用する際の注意事項

対象は科学snapshot `f29eafbf4412193e8e8660d9f051889c2cba684b`の`wp11_all_r_opt2_coherent_analysis_20260925_065827.json`。前段監査commit `3fd8fca966e6b5a6a972aa71de68b31a0129f5ab`の保存所見を引用し、結果の数値・保証欄を更新するものではない。

## 保証状態の読み方

|SOURCEの状態|意味と引用上の制限|
|---|---|
|`feasible`|宣言した会計の数値条件を満たす。許容差を含み、単独では証明書ではない|
|`empirical_screening`|経験的PF係数等に条件付く評価。厳密上界が検証されたとの意味ではない|
|`certified`|SOURCEが要求する追加条件を、宣言した入力・モデル・仮定の下で満たす。PF係数の厳密性そのものをSOURCEが外部証明したという意味ではない|
|`not_certified`|保証条件が不成立または未充足。数値会計に使えるfeasible候補にもなり得る|

旧最終round projectionは`feasible`を中心に保持し、`guarantee_status`／`certification_reasons`／`assumptions`を全て保存していない。したがって、`all_rounds_feasible=True`から、全roundが`certified`だったと引用してはいけない。

WP11経路は`same_snapshot_paper_d6_empirical_coefficient`と`is_rigorous_bound=False`を使う。このため、SOURCEから経験的な条件付きscreeningとして解釈できることと、各保存rowに`empirical_screening`というliteralな状態欄があることは別である。今回、古いrowの状態を再実行・再判定したり、欠けた保証欄を補完したりしていない。

引用時には次を併記する。

- 固定科学snapshot、最終JSON名・content fingerprint、および解釈に使う旧SOURCE。B03新SOURCEで旧結果を生成したように記さない。
- linear H4、1 Å、STO-3G、保存DF rank12、L_D3／12、δ=.02、18 round、q_max131072という対象。
- 総β=.4 rad・総α=.05は設計上の公称予算であり、厳密なbinary64予算不等式の認証ではない。
- effective partial-S2の厳密な固有状態、減衰前signal半径1、alias-free energy branch、fresh-IID bounded Hadamard outcomeの仮定。
- PF係数は経験的。finite-RTE式・測定shot式が接続しても、これだけで全systematic errorの厳密上界にはならない。
- 費用範囲は状態準備P=0のfull Hadamard wrapper compiled RZ、同じopt2 compiler。後半qの単価に外挿が残る。PR/PFの総予算は共通でも、tail-free決定論のβ_RTE=0は許容する。
- 点費用のPR優位と、保存区間の重なりを同時に保持する。RPE最終成功率、energy総資源、実backend／noise、H6/H8移送の保証へ広げない。
- [B08のpreparation差](pr_beta_preparation_lineage_20261018.md)が残るため、旧物理gridを同一WP11 preparationの検証として扱わない。

引用文の例：

> 固定旧H4 snapshotについて、経験的PF係数、effective partial-S2固有状態と信号半径・位相枝の仮定の下で、β・α・整数shot・round scheduleを共同選択した状態準備なしcompiled-RZ点費用を比較した。保存された数値的実行可能性は厳密なRPE成功保証を意味せず、費用の不確かさ区間は重なる。

この例は説明文であり、古いJSONへ新しい保証分類を付与するものではない。

## binary64予算境界

前段監査の固定保存記録では、表現されたbinary64値の厳密有理数和が保存総予算を上回るものは、**β6/50、α20/50 scenario**だった。[引用metadataと出典commit](../../artifacts/resource_applicability/pr_beta_contract_followup/2026-10-18/b02_b06_citation_metadata.json)。今回50 scenarioを再計算していない。

許容差付きの数値的実行可能性と、許容差なしの予算不等式は別である。`1e-15`等の比較許容差や境界refinementの微小余裕は数値処理のためのもので、PFモデル誤差に対するheadroomではない。

binary64のわずかな超過を、無条件の保証PASSにしない。一方、今回費用順位が変わったとは判定しておらず、この所見だけから保存shot・費用・α/β配分の再計算が必要とは結論しない。元の丸めや数値を調整して一致させない。

将来rigorous certificateを要求する場合は、外向き丸め、厳密な予算配分、正の半径・systematic error・branch条件を含む別の契約が必要になる。今回その契約を採択したり科学計算を開始したりしない。B03は例外処理の修正であり、この境界問題やB08の科学的証拠不足を解消するものではない。
