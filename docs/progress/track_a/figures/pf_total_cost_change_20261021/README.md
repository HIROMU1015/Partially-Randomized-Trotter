# PF総費用変化率図：保存100 scope

[構成案](../../PR資源評価_詳細スライド構成案_14枚基本・12枚圧縮_2026-10-21_PF総費用変化量改訂版.md)の本編14枚7／12枚6用。横軸は保存順のscope index、縦軸は最良PF総費用の政策間変化率%。

- [主図PNG](PF_total_cost_change100_clean.png)・[SVG](PF_total_cost_change100_clean.svg)・[PDF](PF_total_cost_change100_clean.pdf)
- [100点と原fieldの対応](display_points.json)
- [H8候補変更2scopeの保存rows](H8_candidate_change_saved_rows.json)
- [表示専用描画プログラム](render_saved_cost_change.py)・[描画検証／出力SHA](render_receipt.json)
- [原成果物コピーのSHA](source_provenance.json)

白抜き丸は不変、色付き点は非ゼロ変化。青丸は旧70のH6 signed、橙四角は旧70のH8、緑三角は新30のH6 noncancellation。星はH8の候補変更2scopeへの重ね表示。100scopeは固有40endpoint条件で、独立100実験ではない。正はPF費用増、負はPF費用減。

保存済みfigure_data.jsonのtotal_change_percentを全100点へ直接転記した。平均・分位点・bin・費用・変化率・候補順位を新たに計算していない。保存順の番号以外は新しい数値を作らない。H8の＋0.283003%／＋0.282968%は政策間の最小費用変化で、候補切替だけの改善量ではない。

完全な日本語caption・条件・限界は構成案B.19と[Codex参照メモ](../../2026-10-21_PR資源評価_PF総費用変化量_Codex参照メモ.md)へ集約する。

|原図（元bytesを保全）|PNG|SVG|PDF|
|---|---|---|---|
|最良PF費用と変化率|[PNG](source/PF_cost_change100.png)|[SVG](source/PF_cost_change100.svg)|[PDF](source/PF_cost_change100.pdf)|
|保存PR/PF比|[PNG](source/PR_PF_ratios100.png)|[SVG](source/PR_PF_ratios100.svg)|[PDF](source/PR_PF_ratios100.pdf)|

source/は結果commit3983fe4fの固定コピー。旧件数図Bは既存ファイルを構成案B.19から参照し、変更していない。画像・描画対応の技術確認は科学的再評価・独立レビューではない。
