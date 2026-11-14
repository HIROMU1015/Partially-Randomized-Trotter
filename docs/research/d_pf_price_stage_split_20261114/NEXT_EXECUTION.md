# 段階別の実行手順（未実行）と開始条件

- 固定SOURCE：`1fd47847b3300395f1b0c0de1868ed7b83905497`（branch `codex/d-price-records-c-bridge-20261114`）
- 実行承認：最大34件の新規compileは、指示書で承認済みである。
- 現在の状態：ユーザーによる比較policyとholdout許容差の採択が確認できないため、**Stage 1も開始していない**。

`62ae1bd5` の全セル一括flag（`--execute-approved-full-pf`、`--execute-approved-t3`）は、このSOURCEから削除した。使わないこと。

## 共通事項

- **実行場所**：このworktree（sparse設定に、各routeの出力dirを追加済み）。固定SOURCE、またはそのbyte一致の子孫commitをcheckoutした状態で実行する。
- **interpreter**：`/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python`（Python 3.12.3、Qiskit 1.3.0）を使い、`-B` を付ける。
- **実行時の自動検査**：
  - 実行するcaller、T3 caller、stage moduleが、HEADとbyte一致すること。
  - native／wrapperが `af284d85` のblobと一致すること。
  - provider `c31ef424` のbyteが一致すること（子process内で検査）。
  - CELL_PLANと全cellが一致すること。
  - stage別のwrapper数。
- **並列化**：子processは、stageのcellをfork workerで並列に処理する。worker数は、既存の規則（idle CPU／available RAM÷2 GiB）で決める。人工的な上限は置かない。
- **失敗時の扱い**：
  - 失敗したcellは、部分保存する。
  - 失敗の後は、新しいcellを開始しない。
  - `STOP_<stage>_PRICE_CHECK` で終了する。
  - retry、別seed、別formula、別幅、追加round、閾値緩和は行わない。

## Stage 1：CALIBRATION（新規22 wrapper）

**開始条件**（すべて必要）：

- ユーザーが、比較policyとholdout判定条件（許容差・分母）を事前に採択していること。
- 技術preflightがPASSしていること（引数なしで実行したとき、2本のcallerがstage分割と保存M5/m4の参照を表示する）。

**手順**：

1. 2本のcallerでCALIBRATIONを実行する。同時に実行してよい（namespaceは別）。

   ```sh
   PY=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
   $PY -B scripts/run_full_pf_shortlist_prices.py --execute-approved-stage CALIBRATION   # 10 wrapper
   $PY -B scripts/run_truncated_p7_prices.py --execute-approved-stage CALIBRATION        # 12 wrapper
   ```

2. 各 `…/calibration/run01/` の5ファイルをcommitし、pushする。このcommitをEとする。
   - 対象：RESULT、PRICES、SIDECARS_UNBOUND、IMPLEMENTATION_EVIDENCE、C_INPUTS
   - sparse worktreeなので `git add --sparse` を使う。
3. 証拠commitを埋め、C guardを実行する。

   ```sh
   $PY -B scripts/run_full_pf_shortlist_prices.py --bind-stage-evidence CALIBRATION --evidence-commit E
   $PY -B scripts/run_truncated_p7_prices.py --bind-stage-evidence CALIBRATION --evidence-commit E
   ```

   `SIDECARS.json` と `GUARD_CHECK.json` ができる。これをcommitし、pushし、remote照合する。
4. Cへの引渡しは、[HANDOFF](HANDOFF.md) のpacketで行う。新規22件と、保存M5/m4の2件の参照で、計24件になる。
5. ここで **mandatory STOP**。Dはfit、係数凍結、m6予測を行わない。

## Stage 1とStage 2の間（Cの担当）

Cが行う作業：

- 24価格を受理し、正式guardにかける。
- 12 modelを分離し、それぞれm4/m5の2点だけで `fit` する。
- 係数、design、SOURCE、許容差を固定commitに保存する。

これにはユーザーの別途の明示承認が必要で、Dは代行しない。

## Stage 2：HOLDOUT（新規12 wrapper）

**開始条件**：

- Cの12 modelが固定commitで公開されていること（形式は [HANDOFF](HANDOFF.md)）。
- ユーザーから、次段階に進む指示があること。
- 比較policyと許容差が採択済みであること。

**手順**：

1. HOLDOUTを実行する。

   ```sh
   $PY -B scripts/run_full_pf_shortlist_prices.py --execute-approved-stage HOLDOUT --frozen-models <C_COMMIT>:<PATH>   # 6 wrapper
   $PY -B scripts/run_truncated_p7_prices.py --execute-approved-stage HOLDOUT --frozen-models <C_COMMIT>:<PATH>        # 6 wrapper
   ```

   compileの前に、次を自動で検査する。
   - CALIBRATIONの7ファイルがHEADにcommit済みで、byte不変であり、dirがcleanであること。
   - CALIBRATIONが完了状態で、placeholderが残っていないこと。
   - HOLDOUTのcellが、取得済みのcellと重複しないこと。
   - 凍結modelがroute別の6 groupで揃い、それぞれ公開済みのm4/m5入力だけでfitされていること。

   実行後に、CALIBRATIONを再照合する。
2. 公開（commit E'）→ `--bind-stage-evidence HOLDOUT --evidence-commit E'` → commit、push、remote照合 → Cへ引渡し → **mandatory STOP**。
3. holdoutの判定（`validate`）、再fit、`QUALIFIED_MODEL` への昇格、m7〜m12の外挿、Gには、自動で進まない。

## STOP条件（指示§11と同じ）

次のいずれかが起きた場合、部分結果と原因を保存してSTOPする。

- 入力・ID・照合の不一致：SOURCEやinput hash、candidate ID／physical ID、Full PF／Truncatedのblockやscalar、compiler profile・control・wrapper。
- gateの不成立：index0、回路の作用、compiled vector。
- recordの不整合：価格record、sidecar、assertion。
- 手順の不成立：保存receiptの受入、Stage 1とStage 2の混同、係数凍結前のHOLDOUT。
- 実行の失敗：compileの失敗、OOM、継続不能な資源不足。

一時的な資源競合だけでは止めない。他の担当者・他のユーザーのprocessや設定には触れない。
