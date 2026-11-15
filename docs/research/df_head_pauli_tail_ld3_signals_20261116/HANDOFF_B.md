# A → B（統合担当にも）：PR-Pauli LD3 の13 round信号packet（段階2）

根拠は [REPORT](REPORT.md)。段階1の引き渡し（configuration・physical ID・辞書・κ）は [`e77c37989` のHANDOFF_B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e77c37989489bc562651b5f1be11411f6b71f92c/docs/research/df_head_pauli_tail_ld3_stage1_20261116/HANDOFF_B.md) のまま変わらない。

## packetの固定locator

| 対象 | commit | path |
|---|---|---|
| **MEAN_HANDOFF（Bの入力）** | **`cdbce383f9944b7f9da1cbbd7851406dbbaae88a`** | `docs/research/df_head_pauli_tail_ld3_signals_20261116/MEAN_HANDOFF.json` |
| round別の生record（各rowの `mean_source`） | `26bac4f320abb345b90ba5cb56a461ff22e1229d` | `docs/research/df_head_pauli_tail_ld3_signals_20261116/run01/LD3_m{0..12}.json`（selector `record`） |
| 実行の記録 | `26bac4f32…` | `.../run01/result.json`（status `ALL_13_PAULI_LD3_MEANS_AVAILABLE_NOT_RPE_QUALIFIED`） |
| round別の要約 | `cdbce383f…` | `docs/research/df_head_pauli_tail_ld3_signals_20261116/ROUND_SUMMARY.csv` |
| configuration | `421258d400a86fffd7c4955d5cf1bd24ac5729cb` | `docs/research/df_head_pauli_tail_ld3_stage1_20261116/config_run01/CONFIGURATION_LD3.json`（ID `dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847`） |
| 辞書 | `50640b74c65c36d381d6058824745f6e38ab3c2d` | `docs/research/df_head_pauli_tail_ld3_stage1_20261116/p0_run01/LD3_DICTIONARY.json`（sha256 `be4ee9cd…`） |
| 数値SOURCE（各recordの `source.commit`） | `8c8ce542e9fc25ead709fe6b0eb71b87ba7434ca` | provider `src/trottertracks/pr_rpe_signal_provider.py`（kernel `e8641b84f`）、adapter `src/trottertracks/df_head_pauli_tail_ld3_signals.py` |

**packetの形式**
- `packets` は1件（prefix 3）。kindは `A_PAULI_ALL_ROUND_MEANS_ONLY_NOT_N_PACKET`。
- 各フラグ：`evaluated_rounds`＝0〜12、`failed_rounds`＝[]、`signal_coverage_complete`＝true、`per_round_source_is_authoritative`＝true。
- 各rowには、`candidate_id`・`identity`・`law_binding`（canonical）、`mean_record`、`mean_source` がある。N・C・G・RMSEの欄はすべてnull。
- 並びと欄は、LD1・LD2の全round packet（`714247619` の `docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json`）と同じ。

## 13個のphysical ID（段階1と同じ）

| m | physical ID | m | physical ID |
|---:|---|---:|---|
| 0 | `RMSE:40cabfa674ff533123d315334f69e4088b2834d1e9a649915627c0471482e86b` | 7 | `RMSE:7912ac19dafcb219164c4760b3fd63c7c3f85d9e8f9f48aef6a4fc1928409dfd` |
| 1 | `RMSE:cec7d69ebcc5e36064d6dcd59e3ecf208bceb697d61eec4a0c6c8bd605bab8bc` | 8 | `RMSE:2382d4cf9a87f0eeb485bf60b07f019038b60dc6c9a0b3931b6b0ae96ca03244` |
| 2 | `RMSE:549d1869e0cec2d2e269723b0454671fa440af039684e873e893229f9bd87606` | 9 | `RMSE:e1e08825d02ab6cb6fb0725ca3c4641a6ac24f6d25a2c9082fd07b0e74c5765c` |
| 3 | `RMSE:919e2c86defb2ef05f101aaa7e0b37bf11f2b2560755eaaaf86d82370256eebf` | 10 | `RMSE:11ee2297ac9b3d3415275c7c61b895933d04ee44d2d1facb9f5a2f78e7627e74` |
| 4 | `RMSE:6290c7ac4b32ece22e8e40258311820445c0c4e7423ba25060123819a9d063c9` | 11 | `RMSE:43d46a2c24aac5f5d54619aeacb89d0083ee42c4a135bad1155793eeb7453c26` |
| 5 | `RMSE:3a1b37dd5311e034fffc409682551f7d66c833894ae87943788ef8aefb803225` | 12 | `RMSE:95014c1c1c105ae09155753b7dfa3edd40ba8900bfc2efe72152e670b1337a28` |
| 6 | `RMSE:d3d7d92e24fec588be5d8febc49c001ce67df335ae0416e113225bc0fb0bc9c1` | | |

## Bの取り込みで変える必要がある1点

既存のB intake（`rpe_pauli_signal_intake.bind_packet`、b2 `284a11c7d`）について。

**何が合わないか**
- 辞書identityの照合に、Aの固定P1 `dictionary_identity`（`714247619`）を使っている。
- この関数は、`P0_operator_residual_source` を `4220f3006` の `run01/result.json` に固定している。
- LD3の辞書identityでは、この欄が `50640b74c` の `p0_run01/result.json` である。そのため、そのままでは「dictionary/hash/bit order」の照合で止まる。

**どう変えるか**
- `helpers.dictionary_identity` として、次の関数を渡してほしい。
  - commit `8c8ce542e`、`src/trottertracks/df_head_pauli_tail_ld3_signals.py` の `b_intake_dictionary_identity`
  - 規則：residual locatorを、辞書と同じcommitにある同じdirectoryの `result.json` にする。
- この関数の引数は、Bと同じ形の `(payload, dictionary_sha256, source)` である。
- LD1・LD2・LD3の辞書identityを、それぞれbit単位で再現することをテストで確認した。
- `job` helperは、固定P1のもの（変更なし）でよい。

**ほかの条件**：packetの組み立て時に、次のB intakeの条件を事前に確認済みである。
- canonical identity・law・candidateが一致する
- record内の `physical_context` が、P1 `job()` で作り直したものと一致する
- signed_weights・allocation・event_countが一致する
- 各rowの `mean_source.scientific_source` が、recordの `source.commit` と一致する
- `scalar_once` がtrueである
- N・C・Gがnullである

## 次の段階（未実行・要許可）

1. **B：LD3のtotal view**
   - LD1・LD2と同じ手順で作る（LD2のtotal viewは `285bf481…`）。
   - 上記の1点以外は、そのまま取り込める想定。
2. **B：LD2と同じVPでのQ3**
   - Pauli Q3の固定経路（`f54b41a2e`、`scripts/run_pauli_q3_three_quarter.py`）を、VP `[48,45,42,39,36,33,29,26,23,19,16,13,9]`（各軸）で1回実行する。
   - LD2のVPの結果（`50be68b80`、MSE上界6.2075×10⁻⁷）と並べる。
   - 信号の上では、偏りがLD2の0.9986倍、log Bがほぼ同じだった。ただし、適合するかどうかは未評価である。
3. **A：段階3の価格**
   - LD3の非configuration cell（q＝8、R＝660、T＝0.8）について、2 path × X/Y、最大4 compileで行う。
   - その後、Bが各分割で配分を探索する。

## ロードマップへの影響（本体は編集していない）

| 箇所 | 更新の提案 |
|---|---|
| PR-Pauliの行 | 「LD3の13 round信号を取得（`cdbce383f` MEAN_HANDOFF、13/13のgateがPASS）。補正後信号の偏りはLD2の0.9986倍、log Bはほぼ同じ。tail event数は0.157倍。Q3・headの費用・event単価は未評価」 |
| 次の判断 | BのLD3 total viewと、同じVPでのQ3（1回） |
| 依存関係 | B：total view → B：Q3 → A：段階3の価格 → B：配分の探索 |

**mandatory STOP。**
