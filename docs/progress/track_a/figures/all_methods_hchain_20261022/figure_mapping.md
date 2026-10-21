# 図・描画行・入力の対応

全図は同じ[figure_data.json](figure_data.json)のrowsから作図する。
絶対図はcost_exact、相対図はrelative_cost_exact。float表示への変換前の有理数tokenを保持した。
各行のresult_sourceで元commit・path・SHA・record locatorへ戻れる。

| 図base名（各PNG/SVG/PDF） | 役割 | ε | data row範囲（1起点） |
|---|---|---:|---|
| `absolute_cost_eps_0p05` | 主図 | 0.05 | 1–28 |
| `relative_cost_eps_0p05` | 相対費用補助図 | 0.05 | 1–28 |
| `absolute_cost_eps_0p005` | 主図 | 0.005 | 29–56 |
| `relative_cost_eps_0p005` | 相対費用補助図 | 0.005 | 29–56 |
| `absolute_cost_eps_0p0005` | 主図 | 0.0005 | 57–84 |
| `relative_cost_eps_0p0005` | 相対費用補助図 | 0.0005 | 57–84 |
| `absolute_cost_eps_0p01` | 追加精度 | 0.01 | 85–112 |
| `relative_cost_eps_0p01` | 相対費用補助図 | 0.01 | 85–112 |
| `absolute_cost_eps_0p001` | 追加精度 | 0.001 | 113–140 |
| `relative_cost_eps_0p001` | 相対費用補助図 | 0.001 | 113–140 |

各28枠はH2/H4/H6/H8の順で、各系の7枠はPR、full PF S2、full PF S4、B0 S2、B0 S4、B0+I S2、B0+I S4。
欠測はnullの枠をそのまま保持する。source全体の別bias・倍率・cohortを描画していない。
H6/H8のresult_sourceは既保存PF総費用表のreplacement列、saved_rank_sourceは同じ政策の保存ordinal。
provenanceの科学SOURCEは計算実装識別子、commitは保存result blobの所在を示すもので別の役割を持つ。
