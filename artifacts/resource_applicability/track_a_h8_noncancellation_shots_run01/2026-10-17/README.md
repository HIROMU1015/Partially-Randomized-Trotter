# H8 経験的非相殺bias・shot診断 run01（NONCANCELLATION_SHOTS のみ）

[結果報告](../../../../docs/research/track_a_h8_noncancellation_shots_result_20261017.md)。SOURCE `cbfd7cacad8a5828c0dbfd284a99ef0871860e66`、plan v2 digest `002f9edb…`。

- `summary_v1.json`：状態、gate、精度×倍率×集合ごとの適格性、通常→非相殺の遷移、shot比・軸別比の分布、新たに不適格となった候補、PR成分照合の最大差、希少層、来歴限定。
- `accounting_rows_v1.csv`：184決定論候補＋PR6の5精度×倍率1/10＝1,900行。通常（保存再現）と非相殺の適格性・shot・headroom。
- `axis_envelopes_v1.csv`：380軸の `abs(total)` と経験的非相殺bias。厳密保証欄は空（null）。
- `PR_component_checks_v1.csv`：PR6候補×14検査の合否と差。
- `execution_binding_v1.json`：SOURCE・plan・grant digest（grant本体は非公開）・fresh gate・環境・入力22 receipt・原結果hash。
- `native_exit_receipt_v1.json`／`resource_receipt_v1.json`／`preservation_receipt_v1.json`：wait4のnative exit・CPU・RSS、PSI、B2非干渉、保全。
- `publication_inventory_v1.json`：公開filesのbytes/SHA（自身を除く）。

経験的感度診断であり厳密bias保証ではない。費用比較・ランキング・PR削減率は計算していない。有効grant・lock・private入力・原結果は非公開領域に保全。mandatory STOP。
