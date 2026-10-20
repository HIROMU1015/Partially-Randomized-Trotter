# 保存450軸：最終表示用4図

| 図 | PNG | SVG | PDF |
| --- | --- | --- | --- |
| 予測対実測、対角線と±2%帯 | [PNG](predicted_vs_measured.png) | [SVG](predicted_vs_measured.svg) | [PDF](predicted_vs_measured.pdf) |
| 相対誤差分布、±2%境界 | [PNG](relative_error_distribution.png) | [SVG](relative_error_distribution.svg) | [PDF](relative_error_distribution.pdf) |
| q別の全点と記述平均 | [PNG](error_vs_q.png) | [SVG](error_vs_q.svg) | [PDF](error_vs_q.pdf) |
| prefix別の全点と記述平均 | [PNG](error_vs_prefix.png) | [SVG](error_vs_prefix.svg) | [PDF](error_vs_prefix.pdf) |

linear H6 DF19/H8 DF28、1 Å/STO-3G、T=.8、P=0、状態準備なしfull Hadamard-wrapper RZ。符号付き相対誤差=100×(保存予測−実測)/実測。S4は合成仮説。450軸は225 cosine/sine pairであり、独立450標本を主張しない。

H6 profile差、runtime同一性未証明、初回終了gate欠測を残す条件付き診断。全点2%未満は保存比較集合での観測であり、全適用域の保証ではない。線は保存集合をpoolした単純平均でfitではない。

`figure_data.json`の全点・分布bin・captionは[初回版](../figure_data.json)とSHAまで同一。変更は目盛のfontsizeのみ。初回の4図×3形式は親directoryに保全した。`display_provenance.json`と`rendering_completion.json`に出典/図SHAを記録する。
