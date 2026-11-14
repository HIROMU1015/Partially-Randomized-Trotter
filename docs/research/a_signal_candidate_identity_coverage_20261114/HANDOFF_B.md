# A → B：signal intakeの引継ぎ

[REPORT](REPORT.md)／[SIGNAL_COVERAGE.csv](SIGNAL_COVERAGE.csv)（`signal_status`・`signal_sources`の列）／[SAVED_ROUND_AUDIT.json](SAVED_ROUND_AUDIT.json)。

新しい信号の計算は行っていない。以下は保存物の受入方法の整理である。

## 1. そのままintakeできるpacket（S1）

| 系列 | packet | Bの既存intake |
|---|---|---|
| PR-Pauli LD1/LD2 | `714247619:…/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json` | eecbc→T.create/bind。caf1417の2評価 |
| Full-H κ2 | `ce6aacfa:…/FULL_H_K2_QDYADIC_KAPPA2/B_SIGNAL_INPUT.json` | C.prepare→T.create（99930491）がPASS |
| Truncated B0I p7・S2・0.1 | `5fb2c32e:…/B0I_P7_S2_D01/B_SIGNAL_INPUT.json` | 同上 |

- 再取得・再評価は不要である。
- Full-H R448（`5fb2c32e:…/FULL_H_K2_QDYADIC_R448/`）は13 rawを保存済みだが、比較候補ではない。κ2と結合しない（Bの拒否testのとおり）。

## 2. PR-DF LD1/LD2（S1B）

- 13 rawは保存済みで、Cの価格coverageとのphysical IDの照合は26軸すべてで一致した。
- 不足しているのは、Bの**版付きdecoder-only bridge**だけである。新しい信号は不要。
- 旧固定gridを維持し、Pauliの辞書・κ・scheduleへ変換しない。

## 3. Full PFとB0I p7 shortlist：round単位の状態

**状態ごとの扱い**

| 状態 | 内容 | 必要な処理 |
|---|---|---|
| S2 | 現行SOURCEの保存direct mean（数値gate PASS） | intakeにはpacket化が必要 |
| S2L | 旧SOURCEのmean | 再資格化が必要 |
| S3 | 保存meanがない | 計算は承認待ち |

**S2のpacket化に必要な項目**（metadataだけで、新しい計算はしない）
- configuration（Full PF：`11cf8ca6`／`a799697f`／`70f7b317`。B0I：`a6e55777`／`defc5e88`／`424eecbf`）と、各roundのidentity・law_binding（`deterministic`）
- raw＝corrected＝保存`direct_PF_mean`、B=1、logB=0
- reference：保存reference pair（`bba15083`）のその時刻の値
  - S-1・B0I packetと同じreferenceである。
  - Track Aのrecordに含まれるreference（T=0.8）は別の保存値で、差は約1e−15。packetにはbba15083を使う。
- scalar：cを一回（Full PF）、またはc＋B0I tail identityを一回（B0I）。保存recordは`scalar_once_at_total_T_in_direct_route=true`
- source：round fileのcommit・path、step cell、数値gate（power unitarity・norm）

**intakeの禁止事項**
- 全13roundが揃うまで、そのcandidateのrisk・N評価は始めない。部分的なround集合で、全round比較を作らない。
- H_effの固有値shiftや二時点の照合で、未取得roundを補完しない。
- S2Lを無条件にS2へ昇格しない。Cの前例（`c5466946`）では、Track Aの旧値は「現在のpower診断が欠ける」として新規directで評価し、screenのm6は1件の受入再資格とした。

## 4. 推定器の版とcanonical ID

- 新しいdecoder viewを作るとcanonical IDが変わる。元physical ID・元canonical IDとの対応は、[SIGNAL_COVERAGE.csv](SIGNAL_COVERAGE.csv)の`physical_id`・`canonical_id`列と、BのREADINESSで保持する。
- Full PF shortlistの元canonical IDは、A follow-upの`C_HANDOFF.json`にある。B0I shortlistのcanonical IDは、T2のconfiguration generatorで決まる（このCSVでは空欄）。
- packet化の際に、B側で生成して照合する。

## 5. 承認が必要な点（ユーザー）

- **新しいround meanの算出**：Full PF・B0I shortlistで計22件ずつ。m3の旧meanを使わない場合は計24件ずつ。
  - 実行する場合は、A（またはC）の既存direct経路と既存gateを使う。
- S2Lの再資格化を行うか、新規directで置き換えるか。
- 揃えた後のrisk・N評価の範囲。
