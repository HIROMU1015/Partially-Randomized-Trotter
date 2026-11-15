# A → B（統合担当にも）：PR-Pauli LD5 の13 round信号packet

根拠は [REPORT](REPORT.md)。LD5の固有価格は、別branchの `a_pauli_ld5_cal_price_20261116/HANDOFF_BC.md`（`codex/a-pauli-ld2-expected-price-plan-20261115`）にある。

## packetの固定locator

| 対象 | commit | path |
|---|---|---|
| **MEAN_HANDOFF（Bの入力）** | **`ac46d100e74f3f56b4a3ff4d95c15fdc7fe8ca98`** | `docs/research/df_head_pauli_tail_ld5_signals_20261116/MEAN_HANDOFF.json` |
| round別の生record（各rowの `mean_source`） | `1e3985e7eb29894934877a74e2a96c3088dd435a` | `docs/research/df_head_pauli_tail_ld5_signals_20261116/run01/LD5_m{0..12}.json`（selector `record`） |
| 実行の記録・normalizer | `1e3985e7e…` | `.../run01/result.json`（`normalizer_cache` に3組） |
| round別の要約 | `ac46d100e…` | `docs/research/df_head_pauli_tail_ld5_signals_20261116/ROUND_SUMMARY.csv` |
| configuration | `39a1fc7dfe9b093f0e246be595e2f7938191368b` | `docs/research/df_head_pauli_tail_ld5_stage1_20261116/config_run01/CONFIGURATION_LD5.json`（ID `4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb`） |
| 辞書 | `6255fbc422121fd777cad52763ef5a684abd46a3` | `docs/research/df_head_pauli_tail_ld5_stage1_20261116/p0_run01/LD5_DICTIONARY.json`（sha256 `2947784c…`） |
| 数値SOURCE（各recordの `source.commit`） | `c67469ad49fd92a3054cb6da9337a33f276494d0` | provider（kernel `e8641b84f`）、adapter `src/trottertracks/df_head_pauli_tail_split_signals.py` |
| B intake helper | `8c8ce542e` | `src/trottertracks/df_head_pauli_tail_ld3_signals.py` の `b_intake_dictionary_identity`（LD3と同じものをそのまま使える） |

**packetの形式**
- `packets` は1件（prefix 5、`A_PAULI_ALL_ROUND_MEANS_ONLY_NOT_N_PACKET`）。
- `evaluated_rounds` は0〜12、`failed_rounds` は空、`signal_coverage_complete` はtrue、`per_round_source_is_authoritative` はtrue。
- 各rowには次が入っている：canonicalの `candidate_id`・`identity`・`law_binding`、`mean_record`（raw・corrected・exact_tail・reference・log B・B・attenuation、数値gate、allowance）、`mean_source`。
- N・C・G・RMSEはnull。
- LD3の公開packetと同じ並び・欄である（LD3のspecでこの経路から公開packetが完全に再現することを確認済み）。

**13個のphysical ID**：段階1の [HANDOFF_B](../df_head_pauli_tail_ld5_stage1_20261116/HANDOFF_B.md) の表と同じ（m12＝`RMSE:795c5535…`）。

## 次の段階（未実行・要許可）

1. **B：LD5のtotal view**：LD3と同じ手順で、helperは上記のものを使う。
2. **B：Q3**
   - VP `[48,…,13,9]` と FULL_PF__B_PRICE `[48,…,16,8,2]` で行う。
   - LD5のQ3適否は未評価である。
   - 信号の上では、偏りがLD3の1.0020倍、後半roundのattenuationは0.962だった。
3. **B：配分探索**：LD5の価格（`a_pauli_ld5_cal_price_20261116/PRICE_26AXIS.json`）を使う。
   - LD5は、m0〜m10ではLD3より1.11〜1.56倍高い。
   - m11では0.86倍、m12では0.60倍で、LD3より安い。
   - そのため、配分の形によって有利な分割が変わる。

## ロードマップへの影響（本体は編集していない）

| 箇所 | 更新の提案 |
|---|---|
| PR-Pauliの行 | 「LD5の13 round信号を取得（`ac46d100e`、13/13 gate PASS、偏りはLD3の1.0020倍）。LD5のQ3は未評価」 |
| 次の判断 | BのLD5 total viewとQ3（2列） |

**mandatory STOP。**
