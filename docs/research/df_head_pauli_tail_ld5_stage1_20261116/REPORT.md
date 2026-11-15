# PR-Pauli LD5 段階1：P0辞書・λ_P・configuration・13 roundのschedule

LD3からfragment 3・4をheadへ移した分割（LD5：head＝one-body＋DF fragment 0〜4、tail＝fragment 5〜9、元の保存順）について、LD3と同じ条件・同じ整数化でtail event数を確定した。

## 冒頭の答え

| 問い | 答え |
|---|---|
| LD5のλ_P | **0.061570372477616604**（非identity項184）。LD3は0.8137790393725127 |
| LD3に対するevent数 | 13 round合計で 3,605,063 → **22,527**（0.62%）。m12は1 stepあたりR 660 → **4**、round全体で 2,703,360 → **16,384** |
| R_m＝1の下限 | **m0〜m10の11 roundがR＝1（1 stepあたり1 event）**。m11はR＝2、m12はR＝4。LD3ではm0〜m2だけが下限だった |
| shotを掛けた総event数（X＋Y） | 保存VP：73,947,076 → **480,276**（0.65%）。LD3のFULL_PF__B_PRICE：29,341,636 → **209,940**（0.72%） |
| LD5の13 round信号へ進むか | **信号の前に、headの費用を測ることを勧める。** 理由は下記 |

**勧める理由**
- tail側のeventは、ほぼ消える。ただし、11 roundは「1 stepに1 event」の下限で止まっている。
- 費用の判断を左右するのは、fragment 3・4の2 blockがheadに加わることによる決定論部分の増分で、まだ測っていない。
- 損益分岐の目安（LD5の単価・総額ではなく、判断のための条件）
  - LD3の価格規則（`f25a05f8a`）では、FULL_PF__B_PRICEでのLD3のevent部分は約1.98億RZ。LD5で減らせるのは最大でこの量（約1.97億RZ）である。
  - これがheadの増分と釣り合う点は、1 stepあたり**約1,535 RZ**（2 blockで、1 blockあたり約768 RZ）。
  - 保存VPでは、分岐点は1 stepあたり約2,408 RZ（1 blockあたり約1,204 RZ）。
  - 参考：LD2→LD3では、1 block（fragment 2）で785 RZ/stepだった。
  - つまり、現在の最安適合配分（FULL_PF__B_PRICE、LD3で541,936,295.95 RZ、`fb4d78781`）の付近では、LD5の利得はほぼゼロになりうる。
- 13 roundの信号（compile 0、数秒）は安いが、それだけでは費用の判断ができない。
- そこで、LD5の較正cell（段階3、最大4 compile）でH5を先に測り、FULL_PF__B_PRICEでの分岐点を下回る場合に信号へ進む順序が効率的である。
  - Q3の適否は、信号がないと判断できない。LD3では信号の偏りがLD2とほぼ同じ（0.9986倍）だったので、LD5でも大きく変わらない可能性はある。ただし未評価である。

**まだ評価していないこと**：LD5の単価・総費用・Q3適否。上の分岐点は、LD5のeventあたり費用をLD3のE3（6.747 RZ/event）と同じ、切片も同じと仮定したものである。

## 1. 13 roundのschedule（整数化後の実際の値）

- R_m：1 stepあたりのtail event数。round全体のevent数：q_m × R_m。
- LD3は保存configuration `dd5b881d…` のscheduleを再利用した。
- 「下限」は、整数化でR＝1（1 stepに1 event）になったroundを示す。そのとき、整数化前の目標値はq以下である。

| m | T | q | R_m LD3 | R_m LD5 | event LD3 | event LD5 | LD5/LD3 | LD5の目標値 | 切上げによる超過 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 1 | 1（下限） | 1（下限） | 1 | 1 | 1.000 | 1 | 0 |
| 1 | 0.2 | 2 | 1（下限） | 1（下限） | 2 | 2 | 1.000 | 1 | 1 |
| 2 | 0.4 | 4 | 1（下限） | 1（下限） | 4 | 4 | 1.000 | 1 | 3 |
| 3 | 0.8 | 8 | 2 | 1（下限） | 16 | 8 | 0.500 | 1 | 7 |
| 4 | 1.6 | 16 | 3 | 1（下限） | 48 | 16 | 0.333 | 1 | 15 |
| 5 | 3.2 | 32 | 6 | 1（下限） | 192 | 32 | 0.167 | 1 | 31 |
| 6 | 6.4 | 64 | 11 | 1（下限） | 704 | 64 | 0.091 | 4 | 60 |
| 7 | 12.8 | 128 | 21 | 1（下限） | 2,688 | 128 | 0.048 | 16 | 112 |
| 8 | 25.6 | 256 | 42 | 1（下限） | 10,752 | 256 | 0.024 | 61 | 195 |
| 9 | 51.2 | 512 | 83 | 1（下限） | 42,496 | 512 | 0.012 | 242 | 270 |
| 10 | 102.4 | 1,024 | 165 | 1（下限） | 168,960 | 1,024 | 0.0061 | 967 | 57 |
| 11 | 204.8 | 2,048 | 330 | 2 | 675,840 | 4,096 | 0.0061 | 3,865 | 231 |
| 12 | 409.6 | 4,096 | 660 | 4 | 2,703,360 | 16,384 | 0.0061 | 15,459 | 925 |
| 計 | | | | | 3,605,063 | 22,527 | 0.0062 | | |

- (λ_P5/λ_P3)²＝0.005724。後半3 roundの比0.0061は、整数化のため少し大きい。
- m3〜m9では、下限のため、整数化前の目標値の2〜32倍のeventが入る（例：m6は目標4に対して64、m9は目標242に対して512）。

**shotを掛けた総event数**（`Σ_m (N_X+N_Y)·q_m·R_m`。event数であり、価格ではない）

| 配分（X/Y各軸のN） | 出所 | LD3 | LD5 | LD5/LD3 |
|---|---|---:|---:|---:|
| 保存VP `[48,45,42,39,36,33,29,26,23,19,16,13,9]` | `2de33a4f4`（`PAULI_LD3_USER_VP_MATCH_LD2_01`） | 73,947,076 | 480,276 | 0.0065 |
| LD3のFULL_PF__B_PRICE `[48,45,42,39,36,33,29,26,23,19,16,8,2]` | `fb4d78781`（`PAULI_LD3_FULL_PF__B_PRICE_VP_CAP_01`、LD3で適合） | 29,341,636 | 209,940 | 0.0072 |

機械可読表：[SCHEDULE_LD3_LD5.csv](config_run01/SCHEDULE_LD3_LD5.csv)・[.json](config_run01/SCHEDULE_LD3_LD5.json)。両LDの13個のphysical ID、目標値、下限の印、配分別の総event数を含む。

## 2. LD5のP0辞書（[p0_run01](p0_run01/)、commit `6255fbc42`）

| 項目 | 値 |
|---|---|
| head / tail | one-body＋二体DF fragment 0〜4 / fragment 5〜9 |
| 非identity Pauli項数 | 184（LD1〜LD3と同数） |
| **λ_P**（辞書の係数の絶対値和。configurationで使用） | **0.061570372477616604** |
| identity由来のscalar | 0.008913337788299017 |
| 保存DF descriptor（別欄・参考） | λ_R＝0.05440471795776202、s_R＝0.008913337788299013。出典は保存context（sha256 `8396605f…`、selector `prefixes/5`、索引 `7a8bc7f13`） |
| identityとs_Rの差 | 3.5×10⁻¹⁸。configurationは固定規則 `KEEP_SAVED_sR_AND_c_ONCE` に従い、保存s_Rを使う |
| λ_P/λ_R | 1.132（LD3 1.395、LD2 1.282） |
| 辞書sha256 | `2947784cb614c097cb5f1e77e5cde99a39a62f89c4f2aa63af8ce3c739428648` |
| 数値gate | 81/81 PASS。同じrunで通したLD2は、保存P0（`4220f3006`）とbit単位で一致した |

## 3. configuration（[config_run01](config_run01/)、commit `39a1fc7df`）

- **LD5 configuration ID：`4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb`**（B validate PASS）
- κ・mapper（B `paper_event_mapper`、目標＝⌈κ(λ_P T)²⌉、R＝max(1, ⌈目標/q⌉)）・binding・profileは、LD3と同じ。specの差は `L_D`・`split`・`law` だけ。
- 元configuration（adapterの入力）：DF `b484d51d…`。PR-DFのLD5候補ではない。
- 同じrun内の再現の対照（すべて全体一致）
  - LD1・LD2の元spec（MENU `a9abf8de`）
  - Pauli LD1・LD2（P1 PLAN `e8641b84f`）
  - LD3の元specと、Pauli LD3 configuration `dd5b881d`（`421258d40`）

## 4. 実装と照合

- LD3段階1の実装（`df_head_pauli_tail_ld3.py`、`2e1949a0b`）とLD3 runnerの補助関数を、無変更で再利用した。
- 新しいfileは2つ（凍結 `d6385593e`）。
  - `scripts/run_df_head_pauli_tail_split_stage1.py`：prefixを引数にした段階1の入口
  - `tests/check_df_head_pauli_tail_split_stage1.py`
- テスト（作成者自身の技術検査で、独立レビューではない）
  - 新テスト2件PASS：人工7 fragmentでのP0の入口、保存metadataでのconfigureの入口（LD3の再現を含む）。
  - 既存テストもPASS：LD3段階1の9件、P0の5件。
- §10.5の照合
  - 範囲：2026-11-15以降に更新されたremote branchと、commit件名。
  - 同じsnapshot・分割・辞書規則・κのLD5辞書やconfigurationはなく、関連する稼働中のjobもなかった。
  - 見つかったLD5関連の資料は、旧条件のものだけだった：H9のLD5 CGS、Truncated PFのprefix 5、D2のPR-DF（LD7・LD9）。
- 実行環境：プロジェクト専用venv（Python 3.11.0rc1、openfermion 1.6.1）。P0は3.3秒、configureは8.9秒。

## 5. 行っていないこと

- 新しい信号、path sampling、回路のbuild・compile、価格のfit、Q3：いずれも0件。
- 固定SOURCE、保存結果、旧STOPは変更していない。
- ロードマップ本体は編集していない（影響は [HANDOFF_B](HANDOFF_B.md)）。

**mandatory STOP。**
