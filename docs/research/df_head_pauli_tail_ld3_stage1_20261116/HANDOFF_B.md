# A → B（統合担当にも）：PR-Pauli LD3 のconfigurationとschedule（段階1）

根拠は [REPORT](REPORT.md)。今回の計算は、P0辞書1件と、metadataだけのconfiguration生成である。信号・価格・Q3・配分はまだない。

## LD3のconfiguration

| 項目 | 値・locator |
|---|---|
| 候補 | PR-Pauli **L_D＝3**（head：one-body＋DF fragment 0〜2、tail：fragment 3〜9 のJW・global集約Pauli辞書） |
| **configuration ID** | **`dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847`** |
| configuration本体 | commit `421258d400a86fffd7c4955d5cf1bd24ac5729cb`、`docs/research/df_head_pauli_tail_ld3_stage1_20261116/config_run01/CONFIGURATION_LD3.json` |
| 比較する基準 | LD2 `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3`（保存P1 PLAN `e8641b84f`、`docs/research/df_head_pauli_tail_p1_results_20261114/run01/PLAN.json`、selector `common_kappa_configurations/2`）。total viewは `285bf481` |
| 固定条件（LD2と同じ） | H4 linear 1.00 Å／STO-3G／tol-only DF10、H `c1a6c428…`、state `9bcc0b0e…`、snapshot `12219b7d…`、元のfragment順、S2、ΔPF＝0.1、K＝2、h＝0.1、M＝12、P＝0 |
| LD2 specとの差 | `L_D`・`split`・`law` だけ。κ・mapper・binding・profile・rmse_policyは同じ |

## 13個のphysical IDとschedule

- R_m：1 stepあたりのtail event数。event数：round全体のq_m×R_m。
- 機械可読表：同じcommitの `config_run01/SCHEDULE_LD2_LD3.csv`・`.json`。

| m | q | R_m LD3 | event LD3 | physical ID（LD3） | physical ID（LD2、参照） |
|---:|---:|---:|---:|---|---|
| 0 | 1 | 1 | 1 | `RMSE:40cabfa674ff533123d315334f69e4088b2834d1e9a649915627c0471482e86b` | `RMSE:babca160daaf91bf567324e3e3ce75d7f2b6c6b9658634d5042e970d4ddc444d` |
| 1 | 2 | 1 | 2 | `RMSE:cec7d69ebcc5e36064d6dcd59e3ecf208bceb697d61eec4a0c6c8bd605bab8bc` | `RMSE:2d543fd5aa57a38f7bd42c4a6efda369590564a201caeaac711019b95d62d081` |
| 2 | 4 | 1 | 4 | `RMSE:549d1869e0cec2d2e269723b0454671fa440af039684e873e893229f9bd87606` | `RMSE:83936d128d760ac8698abecee967a33f3300a90df8810d50851c1b0fb74eae65` |
| 3 | 8 | 2 | 16 | `RMSE:919e2c86defb2ef05f101aaa7e0b37bf11f2b2560755eaaaf86d82370256eebf` | `RMSE:935e3f859b4a6fe2a86890652339f3a52ca295150a005cf8d4b78a3ad64e098e` |
| 4 | 16 | 3 | 48 | `RMSE:6290c7ac4b32ece22e8e40258311820445c0c4e7423ba25060123819a9d063c9` | `RMSE:9b520ce436a0ef95dcf34928e9396f33d7fe8298d1779f374693ea2d54549698` |
| 5 | 32 | 6 | 192 | `RMSE:3a1b37dd5311e034fffc409682551f7d66c833894ae87943788ef8aefb803225` | `RMSE:01489f6e391e9917e96eaa56a0006a91b6099e60956cb5fcea3a516b24df5802` |
| 6 | 64 | 11 | 704 | `RMSE:d3d7d92e24fec588be5d8febc49c001ce67df335ae0416e113225bc0fb0bc9c1` | `RMSE:8826b5c0b36fdb91115109a85c6f2eca40cf1989bb3b3e67d8cbcd1598043529` |
| 7 | 128 | 21 | 2,688 | `RMSE:7912ac19dafcb219164c4760b3fd63c7c3f85d9e8f9f48aef6a4fc1928409dfd` | `RMSE:2ff08a3dcdfb9b52d4a3e005035a8e32dda4f9416fa37029c98ac27687697648` |
| 8 | 256 | 42 | 10,752 | `RMSE:2382d4cf9a87f0eeb485bf60b07f019038b60dc6c9a0b3931b6b0ae96ca03244` | `RMSE:f19730508816a513b8a86ecb49f40bf1d683ed42c82367aaae7d616dc714dba1` |
| 9 | 512 | 83 | 42,496 | `RMSE:e1e08825d02ab6cb6fb0725ca3c4641a6ac24f6d25a2c9082fd07b0e74c5765c` | `RMSE:978923356c4e83c897d03442fb488997d32047237444f95757658f61321b2729` |
| 10 | 1,024 | 165 | 168,960 | `RMSE:11ee2297ac9b3d3415275c7c61b895933d04ee44d2d1facb9f5a2f78e7627e74` | `RMSE:ef4c53c8eb6349379a938d07bcb13518352c20c0aae9059be85a12a802f4cedf` |
| 11 | 2,048 | 330 | 675,840 | `RMSE:43d46a2c24aac5f5d54619aeacb89d0083ee42c4a135bad1155793eeb7453c26` | `RMSE:a7b48a2459439b72bdcfa05e3e7a3baf197b6e13761180ea676027fdaed0eae4` |
| 12 | 4,096 | 660 | 2,703,360 | `RMSE:95014c1c1c105ae09155753b7dfa3edd40ba8900bfc2efe72152e670b1337a28` | `RMSE:e76061c8b32676397b386c5deb2948e95b75c50ed84b3261175e58986581c9db` |

- 13 round合計：LD2 22,967,060 → LD3 3,605,063（0.157倍）。m12では、R_mが4,205 → 660。
- T＝binary64の0.1×2^m（m0の T＝3602879701896397/36028797018963968）。

## 固定locator

| 対象 | locator |
|---|---|
| LD3辞書 | commit `50640b74c65c36d381d6058824745f6e38ab3c2d`、`docs/research/df_head_pauli_tail_ld3_stage1_20261116/p0_run01/LD3_DICTIONARY.json`。sha256（canonical JSON）`be4ee9cda676a4eb3a885912f41f340ee29a3dac09ea8d9402fc26c6fda8da79`。項数184 |
| λ_P | 0.8137790393725127（同じ辞書の `lambda_P`。exact値 7329869956959937/9007199254740992）。configurationのsplit `lambda_R` 欄に入る |
| P0の結果・operator | 同じcommitの `p0_run01/result.json`（81 gate PASS、LD2はbit単位で一致）、`p0_run01/LD3_OPERATORS.npz`（head／tail／Pauli_nonidentity） |
| common operator | 再保存していない（bit単位で一致を確認済み）。`4220f3006` の `docs/research/df_head_pauli_tail_p0_results_20261114/run01/COMMON_OPERATORS.npz` を使う |
| scalar | configuration：保存DFのs_R＝0.09261516589219046（`KEEP_SAVED_sR_AND_c_ONCE`）。Pauli identity＝0.09261516589219046（差0.0） |
| 保存DFのλ_R（参考のみ） | 0.5832704246339966。context `/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/H4_context.json`（sha256 `8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f`、selector `prefixes/3`、索引 `7a8bc7f13` SOURCE_INDEX） |
| κ | exact値 13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201（≈24.3053）。LD2のspecと同じ |
| 整数化 | B `ce0517de` の `paper_event_mapper`。R_m＝max(piece数, ⌈⌈κ(λ_P T)²⌉/q_m⌉) |
| 入力 | snapshot `/home/AbeHiromu/projects/h4-followup-validation/20261018/private/inputs/tol_only_snapshot.npz`（sha256 `12219b7d…`）、receipt `.../tol_only_receipt.json`（`7a161c1a…`）。SOURCE_INDEX `7a8bc7f13` の saved_inputs.H4 |
| code | 凍結 `2e1949a0b66dd8fe086ac7cd3104eb89b4553e38`（`src/trottertracks/df_head_pauli_tail_ld3.py`、`scripts/run_df_head_pauli_tail_ld3.py`）。固定P0 `fd111f989`・P1 `05447d1d4`・B `ce0517de`（DF元specはB `437ba4ac`） |
| DF元configuration（adapter入力のみ） | `6493f4fd5237b8a227d64425f55dda29581ae7016844b4ab7a88eb2d715c46c1`、同じcommitの `config_run01/ORIGINAL_DF_CONFIGURATION_LD3.json`。PR-DFのLD3候補ではない |

## 次の段階へのつなぎ方（未実行・要許可）

1. **段階2：13 roundの信号**（A）
   - 既存のP1・全round経路で、configuration `dd5b881d…` の13 jobを実行する。
   - 必要なもの：
     - common operator（`4220f3006`）
     - `LD3_OPERATORS.npz` の `Pauli_nonidentity`（`50640b74c`）
     - 保存H4 reference
   - 形式は、LD1・LD2と同じMEAN_HANDOFFのpacket。
   - その後、Bが同じ手順でtotal viewを作る。
2. **LD2との同じVPでの比較**（B）
   - Pauli Q3の固定経路（`f54b41a2e`、`scripts/run_pauli_q3_three_quarter.py`）を、VP `[48,45,42,39,36,33,29,26,23,19,16,13,9]` で1回実行する。
   - LD2のVPの結果（`50be68b80`、MSE上界6.2075×10⁻⁷）と並べる。
3. **価格と配分の探索**
   - 計画の段階3の価格規則（LD3の非configuration cellを最大4 compile）が入った後に、各分割で行う。
   - 段階3の監査cellのRは、R_12(LD3)＝660 になる。

**未評価のもの**
- headが1 block増えることの費用
- LD3のevent 1件あたりの費用
- Q3で適合するのに必要なshot数

event数の減少（0.157倍）は、このうちtail側の負荷だけを示す。

## ロードマップへの影響（本体は編集していない）

| 箇所 | 更新の提案 |
|---|---|
| PR-Pauliの行 | 「LD3の段階1が完了：λ_P＝0.8138（184項）、configuration `dd5b881d`。同じκで、tail event数はLD2の0.157倍（m12でR 4205→660）。headの費用、event単価、Q3は未評価」 |
| 次の判断 | LD3の段階2（13 roundの信号、compile 0）を許可するか |
| 依存関係 | A：段階2の信号 → B：total viewと同じVPでのQ3 → A：段階3の価格 → B：配分の探索 |

**mandatory STOP。**
