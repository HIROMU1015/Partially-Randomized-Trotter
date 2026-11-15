# HANDOFF：PR-Pauli信号の共通入口（B2・B向け）

想定読者：B2（configurationを渡す側）、B（packetを取り込む側）。結果と根拠は [REPORT.md](REPORT.md)。

## 場所

| 項目 | 値 |
|---|---|
| branch | `codex/pr-r-dependent-n-model-20261111` |
| SOURCE | `4e5b8bef634e3f071fe63df9683d7245df984d98`（最新。正規化子の上限と、binary64で表せないstepの判定を追加した）。初版は `93a5ef3b9fed337cc4d26ee684defd304fa0eac0` |
| 本体 | `src/trottertracks/pauli_signal_entry.py`（`load_configuration`・`plan`・`physical_key`・`saved_index`・`packet`・`b_intake_check`） |
| 実行器 | `scripts/run_pauli_signal_entry.py`（`--plan`／`--execute`／`--assemble`） |
| Bの照合 | `scripts/check_pauli_signal_entry_b_bind.py`（Bの `bind_packet` をBのcodeで呼ぶ） |
| 検証 | `scripts/validate_pauli_signal_entry.py`、`tests/check_pauli_signal_entry.py` |
| 保存信号の一覧（既定） | LD1/LD2 `714247619`、LD3 `cdbce383f`、LD5 `ac46d100e` のMEAN_HANDOFF（`E.SAVED_PACKETS`）。新しいpacketを公開したら `--saved-packets` で足す |
| Python | `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`（入口）。Bの照合は3.12の環境 |

## 呼出し方

すべてrepository rootで実行する。`--config` には次のどれかを渡す。
- file：configuration、MEAN_HANDOFF、B2のNEEDSの候補行、B spec
- `COMMIT:PATH`

**1. 計画**（metadataだけ。配列・平均値の計算なし。数秒）
```
python scripts/run_pauli_signal_entry.py --plan --config <CONFIG> [--configuration-id <ID>] [--out <DIR>]
```
標準出力には、`counts`・`reuse_rounds`・`rebind_rounds`・`request_rounds`・`unsupported_rounds` と、roundごとの対応区分が出る。`--out` を付けると、`PLAN.json` も書く。

**2. 取得**（新しいdirectoryに1回だけ。再試行・再開なし）
```
python scripts/run_pauli_signal_entry.py --execute --config <CONFIG> --source <HEAD> --out <REPO_DIR>/run01 [--approve-new-means | --dry-context]
```
- 未対応のroundがあると、何もせずに止まる（`STOP_UNSUPPORTED_ROUNDS`）。
- re-boundのcarrierは、常に書く。metadataだけの処理である。
- REQUESTのroundは、次のとおり扱う。
  - 何も付けない：計算しないで止まる（`STOP_REQUESTS_NOT_APPROVED`）。
  - `--dry-context`：contextと参照値を作り、平均値の直前で止まる。
  - `--approve-new-means`：既存providerで計算する。**新しい科学計算なので、利用者の承認を得た場合に限る。**
- 成功のstatusは `ENTRY_CARRIERS_COMPLETE_NOT_RPE_QUALIFIED`。

**3. 組立て**（取得したrun directoryをcommitした後、そのcommitで実行する）
```
python scripts/run_pauli_signal_entry.py --assemble --config <CONFIG> --source <SOURCE> --raw-commit <RAW> --run-dir <REPO_DIR>/run01/ --out <REPO_DIR>
```
- 全roundが同一identityの再利用なら、`--source`・`--raw-commit`・`--run-dir` は不要。
- 出力は `MEAN_HANDOFF.json` と `ROUND_SUMMARY.csv`。

**4. Bの `bind_packet` での照合**（任意。Bのcodeを取り出して呼ぶ）
```
git archive b6d4dea49 src | tar -x -C <B_DIR>
<python3.12> scripts/check_pauli_signal_entry_b_bind.py --b-root <B_DIR> --handoff <COMMIT:PATH|FILE>
```

## 出力の形式

**PLAN.json**（`A_PAULI_SIGNAL_ENTRY_PLAN_V1`）：roundごとに次の欄を持つ。

| 欄 | 内容 |
|---|---|
| `decision` | REUSE／REQUEST／UNSUPPORTED |
| `reuse` | REUSEのとき、`IDENTICAL_IDENTITY` か `REBOUND_SAME_PHYSICAL_KEY` |
| `saved` | 再利用元の `mean_source`・configuration・候補ID・round |
| `request` | REQUESTのとき：provider・kernel・context（共通演算子とtail演算子の場所）・参照値・実行経路 |
| `changed_from_saved_round` | 同じround・同じ辞書の保存記録との差分の欄 |
| `provider_support`・`provider_reasons` | 対応区分と、その理由 |
| `job` | q・T・δ・K・R・event数・配分 |

**MEAN_HANDOFF.json**：LD1〜LD5と同じ形式（`A_PAULI_ALL_ROUND_MEANS_ONLY_NOT_N_PACKET`）。
- 各行の `mean_source.scientific_source` は、その記録を算出したSOURCEである（per-round SOURCEが正本）。
- 再利用した行は `reused_without_reanalysis=True`。packetの `reused_rounds`／`new_rounds` で区別できる。
- re-boundの記録には `reuse` 欄があり、元の `mean_source`・configuration ID・候補ID・round・物理keyを持つ。
- top levelの各欄
  - `configuration_source`：入力がGitの場所ならそこ、そうでなければrun directoryの `CONFIGURATION.json`
  - `dictionary_source`
  - `B_intake_dictionary_identity`：`8c8ce542e` の `b_intake_dictionary_identity`
  - `raw_result_source`
  - status：`ALL_13_PAULI_LD{p}_MEANS_AVAILABLE_NOT_RPE_QUALIFIED`

## Bでの取り込み方

- Bの `rpe_pauli_signal_intake.bind_packet`（`b6d4dea49`）は、LD5と同じ引数でこのpacketを受け取れる。
  - helpers：`_ld3_metadata_helpers(root)`
  - target：LD2の保存target view（`91349e703` `B_MEANS_LD2.json`）
  - 記録：各行の `mean_source` から読む
- 確認結果：保存LD1・LD2・LD3・LD5から組み立てた4 packetと、人工の4例は、いずれもPASSした。
- `accept_saved` は、組立てSOURCEの固定の一覧（LD1/LD2・LD3・LD5）を持つ。新しいpacketを受けるには、一覧に足すか、`handoff_source` を受ける汎用の分岐が必要である（Bの作業）。

## B2との分担

- B2：候補のconfigurationを作る（分割・κ・Δ_PFなど）。shot探索も行う。
- 入口に渡すもの：NEEDSの候補行、またはB spec。
- configurationが無いL_D（NEEDSの `CONFIGURATION_AND_SCHEDULE`）は、先に `scripts/run_df_head_pauli_tail_split_stage1.py --p0/--configure --prefix P` で作る。
- 入口の出力のうち、B2が使うもの：
  - 取得要求のround数と内容（新規計算の量）
  - 対応区分（`SUPPORTED_NOT_VALIDATED_FOR_PAULI` は、実計算の前に利用者の判断が必要）

mandatory STOP。この入口は、新しい科学計算の実行を許可しない。
