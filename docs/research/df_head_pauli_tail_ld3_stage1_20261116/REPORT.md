# PR-Pauli LD3 段階1：P0辞書・λ_P・configuration・13 roundのschedule

固定計画 [`a50dce8f1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a50dce8f1cbb38b2914bb2fb4ff79bab35e6e8f6/docs/research/a_pauli_ld_split_plan_20261116/REPORT.md) の段階1だけを実行した。LD2からfragment 2をheadへ移したときのtail event数を、同じ条件・同じ整数化で確定した。

## 冒頭の答え

| 問い | 答え |
|---|---|
| LD3のλ_P | **0.8137790393725127**（非identity項184、Pauli係数の絶対値和）。LD2は2.0550998769244275 |
| LD2に対するevent数 | 同じκで**約0.157倍**（(λ_P3/λ_P2)²＝0.1568）。13 round合計は22,967,060 → **3,605,063** |
| 後半round | m10〜m12の合計が22,607,872 → **3,548,160**（−84.3%）。m12は1 stepあたりR＝4,205 → **660**、round全体で17,223,680 → **2,703,360** |
| 次にLD3の13 round信号を取得するか | **勧める。** 理由は下記 |

**勧める理由**
- LD2の負荷はほぼ後半roundにある。既存VP配分で重み付けすると、m10〜m12がevent数の96.9%を占める。その後半roundで、tail event数が一様に約16%へ下がった。
- 信号の取得はcompile 0で、既存のproviderとP1経路をそのまま使える（configuration `dd5b881d…` から13 job）。
- 一方、今回は判断できないことがある。fragment 2をheadで決定論的に扱うと、headのPF誤差（Δ＝0.1のS2）が変わり、信号のbiasとQ3で必要なshot数が変わり得る。これは13 roundの信号がないと評価できない。

**まだ評価していないこと**
- event数はtail側の負荷の指標である。次の3点は未評価である。
  - headが1 block（fragment 2）増えることの費用
  - LD3辞書のevent 1件あたりの費用
  - Q3で適合するのに必要なshot数
- 参考（採否には使わない）：event 1件あたりの費用がLD2と同じ（S0-r1で約6.3 RZ）と仮定すると、m12の1 stepのtail本体は約26,700 → 約4,200 RZになる。headの増分がこの差（約22,000 RZ/step）を下回れば、1 stepの費用は下がる方向である。LD1とLD2のheadの差は、m0の単一pathで856 RZだった。

## 1. 13 roundのschedule（整数化後の実際の値）

- R_m：1 stepあたりのtail RTE event数（BのR_outer、step時間は0.1）。
- round全体のevent数：q_m × R_m（q_m stepの合計。Bのcandidate event_countと一致することを確認済み）。
- LD2は保存configuration `d7dd4f52…`（`e8641b84f` P1 PLAN.json）のscheduleをそのまま再利用した。

| m | T | q | R_m LD2 | R_m LD3 | R比 | event LD2 | event LD3 | event比 | 減少分 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 1 | 2 | 1 | 0.500 | 2 | 1 | 0.500 | 1 |
| 1 | 0.2 | 2 | 3 | 1 | 0.333 | 6 | 2 | 0.333 | 4 |
| 2 | 0.4 | 4 | 5 | 1 | 0.200 | 20 | 4 | 0.200 | 16 |
| 3 | 0.8 | 8 | 9 | 2 | 0.222 | 72 | 16 | 0.222 | 56 |
| 4 | 1.6 | 16 | 17 | 3 | 0.176 | 272 | 48 | 0.176 | 224 |
| 5 | 3.2 | 32 | 33 | 6 | 0.182 | 1,056 | 192 | 0.182 | 864 |
| 6 | 6.4 | 64 | 66 | 11 | 0.167 | 4,224 | 704 | 0.167 | 3,520 |
| 7 | 12.8 | 128 | 132 | 21 | 0.159 | 16,896 | 2,688 | 0.159 | 14,208 |
| 8 | 25.6 | 256 | 263 | 42 | 0.160 | 67,328 | 10,752 | 0.160 | 56,576 |
| 9 | 51.2 | 512 | 526 | 83 | 0.158 | 269,312 | 42,496 | 0.158 | 226,816 |
| 10 | 102.4 | 1,024 | 1,052 | 165 | 0.157 | 1,077,248 | 168,960 | 0.157 | 908,288 |
| 11 | 204.8 | 2,048 | 2,103 | 330 | 0.157 | 4,306,944 | 675,840 | 0.157 | 3,631,104 |
| 12 | 409.6 | 4,096 | 4,205 | 660 | 0.157 | 17,223,680 | 2,703,360 | 0.157 | 14,520,320 |
| 計 | | | | | | 22,967,060 | 3,605,063 | 0.157 | 19,361,997 |

- Tは binary64の0.1 × 2^m（exact値は表ファイルの `T_exact`）。
- m0〜m2はR＝1（下限1と切り上げの効果）で、比が0.157から外れる。これらのroundのevent数は全体に対して無視できる。
- 参考：既存VP配分 `[48,45,42,39,36,33,29,26,23,19,16,13,9]`（各軸、`50be68b80`）で両軸を重み付けしたevent数は、471,030,556 → 73,947,076（0.157）。これは費用でもQ3の結果でもない。LD3がVPで適合するかは未評価である。

機械可読表：[SCHEDULE_LD2_LD3.csv](config_run01/SCHEDULE_LD2_LD3.csv)・[SCHEDULE_LD2_LD3.json](config_run01/SCHEDULE_LD2_LD3.json)。両LDの13個のphysical IDと、整数化前の目標値（paper_total_target）も含む。

## 2. LD3のP0辞書

[p0_run01/result.json](p0_run01/result.json)（commit `50640b74c`）

| 項目 | 値 |
|---|---|
| head / tail | one-body＋二体DF fragment 0〜2 / fragment 3〜9 |
| 非identity Pauli項数 | 184（LD1・LD2と同数） |
| **λ_P**（辞書のPauli係数の絶対値和。configurationで使用） | **0.8137790393725127** |
| identity由来のscalar | 0.09261516589219046（保存DFのs_Rとbit単位で一致、差0.0） |
| 保存DF descriptorのλ_R（別欄・参考） | 0.5832704246339966。出典：保存context `/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/H4_context.json`（sha256 `8396605f…`、selector `prefixes/3`、索引 `7a8bc7f13` SOURCE_INDEX） |
| λ_P / λ_R | 1.395（LD1 1.155、LD2 1.282） |
| 辞書sha256 | `be4ee9cda676a4eb3a885912f41f340ee29a3dac09ea8d9402fc26c6fda8da79` |
| 数値gate | 81/81 PASS（operator・sector・closureは1e−9、stateは1e−12）。LD3のgate値の最大は `full_H_split_fro`＝6.0×10⁻¹⁵ |
| 個々のlabelのsector漏れ | 136/184（LD2と同じ。合計としてはsectorを保つ） |

- λ_P/λ_Rが計画時の想定（LD1・LD2並み）より大きく、event比は計画の見込み0.11〜0.13より大きい0.157になった。
- **LD2の対照**：同じrunでLD2も通し、保存P0（`4220f3006`）と比べた。辞書、JW係数、operator、common operator、共通する66件のcheck値がbit単位で一致した。このため、LD2とcommon operatorの成果物は再保存していない。
- 実行環境：プロジェクト専用venv（Python 3.11.0rc1、numpy 1.26.4、scipy 1.14.1、openfermion 1.6.1）。P0と同じ。所要3.2秒。

## 3. LD3のconfiguration

[config_run01/result.json](config_run01/result.json)（commit `421258d40`）

| 項目 | 値 |
|---|---|
| **Pauli LD3 configuration ID** | **`dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847`**（[CONFIGURATION_LD3.json](config_run01/CONFIGURATION_LD3.json)） |
| κ | 元のexactな共通κ（≈24.3053）。LD2のspecとbit単位で同じ |
| 整数化 | Bの `paper_event_mapper`（`ce0517de`）。目標＝⌈κ(λ_P T)²⌉、R_m＝max(piece数, ⌈目標/q_m⌉)。LD1・LD2と同じ |
| LD2 specとの差 | `L_D`・`split`・`law` だけ（h・M・ΔPF・K・binding・profile・rmse_policy・event_mapping・κは同じ） |
| split | λ_R欄＝λ_Pのexact値（出典はLD3辞書の `lambda_P`）、scalar＝保存s_R、tail_dictionary＝LD3辞書（sha `be4ee9cd…`、P0 locator `50640b74c`） |
| B validate | PASS |

**元configuration（adapterの入力）**
- LD3のDF元configuration `6493f4fd…` を [ORIGINAL_DF_CONFIGURATION_LD3.json](config_run01/ORIGINAL_DF_CONFIGURATION_LD3.json) に保存した。
- 作り方は、a9abf8deのPREPARE_MENU（B `437ba4ac`）と同じ入力経路で、prefixだけを3にしたもの。
- これはPauli adapterへのspec入力にすぎない。PR-DFのLD3候補として提案・評価したものではない。

**再現の対照（同じrun内）**
- DF元のLD1・LD2を再生成すると、MENU `a9abf8de` の保存configurationと全体が一致した。
- Pauli LD1・LD2を再生成すると、保存P1 PLAN（`e8641b84f`）と全体が一致した（ID `d905d219…`・`d7dd4f52…`）。

## 4. 実装：固定SOURCEを保ったprefixの拡張

- 固定moduleは変更していない。P0 `fd111f989`・P1 `05447d1d4` とbyte単位で同じであることを、実行時に確認している。
- 新しいfileは3つ。凍結commitは `2e1949a0b`。
  - [`src/trottertracks/df_head_pauli_tail_ld3.py`](../../../src/trottertracks/df_head_pauli_tail_ld3.py)
    - P0：固定 `diagnose` の本体を、prefixのloop `(1, 2)` だけを差し替えて実行する（ASTの変更はこの1箇所）。JW変換・global集約・identity処理・gateは固定codeのまま。
    - 元configuration：PREPARE_MENUと同じ経路。
    - P1：固定の `generate_configuration` を使う。dictionary identityの中のP0 residual locatorだけが、辞書の出所に従う。保存LD1・LD2のlocatorでは、従来と同じ結果になる。
  - [`scripts/run_df_head_pauli_tail_ld3.py`](../../../scripts/run_df_head_pauli_tail_ld3.py)：`--p0`（P0）と `--configure`（metadataのみ。P0の成果をcommitした後に実行）。
  - [`tests/check_df_head_pauli_tail_ld3.py`](../../../tests/check_df_head_pauli_tail_ld3.py)
- 技術テスト（作成者自身による検査で、独立レビューではない）
  - 新テスト：9件PASS。
    - 人工5 fragmentのfixtureで、既定prefixでの固定 `diagnose` とのbit一致、LD3のhead/tail分割、scalar不一致時のSTOP、prefix範囲外と固定loop改変の拒否を確認した。
    - 実際の入口（`execute_p0`・`execute_configure`）を、人工辞書で通した。
    - 保存metadataだけを使い、LD1・LD2の構成が再現することを確認した。
    - 保存arrayの読込とcircuit生成は、テスト中は禁止している。
  - 既存のP0テスト5件、P1テスト6件もPASS。
- 実行の順序：凍結 `2e1949a0b` → P0 `50640b74c` → configure `421258d40`。各run 1回で、retry・閾値の変更・fallbackは0。

## 5. 既存成果との照合（§10.5）

- 照合した範囲：2026-11-13以降に更新されたremote branch（A・B・C・C2・D・D2を含む）のfile名とcommit件名、および稼働中のprocess。
- 同じsnapshot・分割・辞書規則・κのLD3辞書、configuration、scheduleはなかった。稼働中の関連jobもなかった。
- 見つかったLD3の名前を持つ資料は、すべて旧条件のものだった。
  - rank7・rank12・d100の資料
  - C1のxdf prefix budget
  - 2026-09の方向性検証
  - Track CのPR-DF側
- D2の `d_pr_df_ld_split_q3_resume_20261116` はPR-DFのLD1・LD2だけで、LD3は0件だった。
- 再利用したもの：
  - 保存H・state・snapshot・context
  - LD2の辞書とschedule（再生成した値は照合にだけ使用）
  - 固定P0・P1・B・PREPARE_MENUの経路
  - 既存VP

## 6. 行っていないこと・保全

- 新しい信号生成、path sampling、回路のbuild・compile、QPY監査、価格予測、Q3、shot配分は行っていない（いずれも0件）。
- 固定SOURCE、保存結果、旧STOPは変更していない。ロードマップ本体も編集していない（影響は [HANDOFF_B](HANDOFF_B.md) に記載）。
- 新しい性能手法は導入していない（P0は3秒、configureは7秒）。

**mandatory STOP。**
