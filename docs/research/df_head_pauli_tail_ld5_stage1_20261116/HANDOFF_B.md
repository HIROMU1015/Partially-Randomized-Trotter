# A → B（統合担当にも）：PR-Pauli LD5 のconfigurationとschedule（段階1）

根拠は [REPORT](REPORT.md)。今回の計算は、P0辞書1件と、metadataだけのconfiguration生成である。信号・価格・Q3はまだない。

## LD5のconfiguration

| 項目 | 値・locator |
|---|---|
| 候補 | PR-Pauli **L_D＝5**（head：one-body＋DF fragment 0〜4、tail：fragment 5〜9 のJW・global集約Pauli辞書。元の保存順） |
| **configuration ID** | **`4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb`** |
| 本体 | commit `39a1fc7dfe9b093f0e246be595e2f7938191368b`、`docs/research/df_head_pauli_tail_ld5_stage1_20261116/config_run01/CONFIGURATION_LD5.json` |
| 比較する基準 | LD3 `dd5b881d…`（`421258d40`）。total view `6757998a…`（B `2de33a4f4`） |
| 固定条件（LD3と同じ） | H4 linear 1.00 Å／STO-3G／tol-only DF10、H `c1a6c428…`、state `9bcc0b0e…`、snapshot `12219b7d…`、S2、ΔPF＝0.1、K＝2、h＝0.1、M＝12、P＝0、exactな共通κ、B `ce0517de` の `paper_event_mapper` |
| LD3 specとの差 | `L_D`・`split`・`law` だけ |

## 13個のphysical IDとschedule

R_m：1 stepあたりのtail event数（「下限」はR＝1）。event：round全体のq×R_m。

| m | q | R_m | event | physical ID |
|---:|---:|---:|---:|---|
| 0 | 1 | 1（下限） | 1 | `RMSE:a5351598308c1e1dac90c56bff4b772c7af14446fefa98badaf106550d9883ec` |
| 1 | 2 | 1（下限） | 2 | `RMSE:4038c439cc0544c35e541b99a1fafce746d610220a7215fb072a60805cb68294` |
| 2 | 4 | 1（下限） | 4 | `RMSE:5bf6695b6177cba6eb18c5793db0663ec29cc6edac0d6e3793080946f7105f72` |
| 3 | 8 | 1（下限） | 8 | `RMSE:c943efea89f5909a224ddd5ec2199743e70b2efd4afc2f7a8ae98f5185a0404b` |
| 4 | 16 | 1（下限） | 16 | `RMSE:5c7ee2b9b248f15bc938dd25cbaa2da354b01940bbbb7eec69029f9166cc6a2a` |
| 5 | 32 | 1（下限） | 32 | `RMSE:98572f0f16ed0e403aaae132547be9c3d18f9d5a81bb468a38a0fbb80f586afd` |
| 6 | 64 | 1（下限） | 64 | `RMSE:5d165e00f731ee50029e0d92c503ea2580801413be28161d5198187e9d182f6e` |
| 7 | 128 | 1（下限） | 128 | `RMSE:a6a4dcc93eb26259b227031623f0de5aca6200456242268f54f5f252fa7d594d` |
| 8 | 256 | 1（下限） | 256 | `RMSE:da51a62286735b5ee4392148a2ae2ff291fadbd2f4d1de8f59f9eddcae0d8c8c` |
| 9 | 512 | 1（下限） | 512 | `RMSE:f9ce88668a2edf3470b26a6aa0cdada4027328cf9e92266193b800a22b40cb1f` |
| 10 | 1,024 | 1（下限） | 1,024 | `RMSE:2b3478a03fca0840b1dff9e60f9636500d700111d45358a9354f0a5f70f5970f` |
| 11 | 2,048 | 2 | 4,096 | `RMSE:3023dfdd6865bca0bd01243d85113693300b5aaab8be35ac6ed9664a1e3ac3fa` |
| 12 | 4,096 | 4 | 16,384 | `RMSE:795c55355aec18b321e90769af58f2b4a8d7a3dcd793b488cbc87fa3d2a1a267` |

- 13 round合計：LD3 3,605,063 → LD5 22,527。
- shotを掛けた総event数（X＋Y）
  - 保存VP：73,947,076 → 480,276
  - LD3のFULL_PF__B_PRICE（`fb4d78781`）：29,341,636 → 209,940
- 機械可読表：同じcommitの `config_run01/SCHEDULE_LD3_LD5.csv`・`.json`（LD3とLD5の両方のID、目標値、下限の印、配分別の総event数）。

## 固定locator

| 対象 | locator |
|---|---|
| LD5辞書 | commit `6255fbc422121fd777cad52763ef5a684abd46a3`、`docs/research/df_head_pauli_tail_ld5_stage1_20261116/p0_run01/LD5_DICTIONARY.json`。sha256（canonical JSON）`2947784cb614c097cb5f1e77e5cde99a39a62f89c4f2aa63af8ce3c739428648`。項数184 |
| λ_P | 0.061570372477616604（辞書の `lambda_P`）。configurationのsplit `lambda_R` 欄 |
| P0の結果・operator | 同じcommitの `p0_run01/result.json`（81 gate PASS、LD2はbit単位で一致）、`p0_run01/LD5_OPERATORS.npz`（head／tail／Pauli_nonidentity）。common operatorは `4220f3006` の `COMMON_OPERATORS.npz` |
| scalar | configuration：保存s_R＝0.008913337788299013（`KEEP_SAVED_sR_AND_c_ONCE`）。Pauli identity＝0.008913337788299017（差3.5×10⁻¹⁸） |
| 保存DFのλ_R（参考） | 0.05440471795776202。context `/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/H4_context.json`（sha256 `8396605f…`、`prefixes/5`、索引 `7a8bc7f13`） |
| κ | LD3と同じexact値（13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201） |
| code | 凍結 `d6385593e`（`scripts/run_df_head_pauli_tail_split_stage1.py`）。再利用した `2e1949a0b`（`df_head_pauli_tail_ld3.py`）。固定P0 `fd111f989`・P1 `05447d1d4`・B `ce0517de`（DF元specはB `437ba4ac`） |
| B intake helper | `8c8ce542e` の `b_intake_dictionary_identity` が、そのまま使える（residual locator＝辞書と同じcommitの `p0_run01/result.json`） |

## 次の段階の候補（未実行・要許可）

1. **段階3を先に：LD5の較正cell（最大4 compile）でH5を測る**
   - 判断の目安：FULL_PF__B_PRICEで、headの増分が1 stepあたり約1,535 RZを下回ればLD5が有利、保存VPなら約2,408 RZが目安。
   - この目安は、LD5のeventあたり費用がLD3のE3と同じと仮定した場合の値である。
   - 参考：LD2→LD3の1 block増分は785 RZ/stepだった。
2. **段階2：LD5の13 round信号**（compile 0）
   - 既存のLD3段階2の経路（`8c8ce542e`）を、prefix 5に広げて使う。
   - その後、BがQ3を行う（VPとFULL_PF__B_PRICE）。
3. LD5の単価・総費用・Q3適否は未評価である。

## ロードマップへの影響（本体は編集していない）

| 箇所 | 更新の提案 |
|---|---|
| PR-Pauliの行 | 「LD5の段階1：λ_P＝0.0616、configuration `4d6e2c9c`。tail eventはLD3の0.62%（m0〜m10はR＝1の下限）。headが2 block増える費用は未測定。FULL_PF__B_PRICEでの損益分岐はhead増分約1,535 RZ/step」 |
| 次の判断 | LD5の較正cell（H5）を先に測るか。または信号を先に取るか |
| 依存関係 | A：H5の測定 → （分岐点を下回れば）A：信号 → B：Q3・配分 |

**mandatory STOP。**
