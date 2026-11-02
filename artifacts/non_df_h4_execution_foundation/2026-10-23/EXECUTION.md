# H4実分子pilot開始手順（次回の明示一括承認後のみ）

この文書のactivation/executeは**今回実行していない**。候補contractは3 flag falseのまま、科学grantは未発行。下記は承認後にCodexが追加実装なしで実施する手順である。ユーザーの明示一括承認を受領する前にapproval JSONを作ったり、activate/executeしたりしない。

使用worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-execution-foundation-20261023`。

1. `git status --short`が空、SOURCE.jsonの技術file hashと現行fileが一致することを確認する。科学SOURCEはこのclean HEADに固定する。環境の空き資源を新たに観測し、共有環境は変更しない。保存証拠検査はstdlibの`python3 -B scripts/verify_h4_execution_foundation.py`で実施できる。
2. 承認されたcontract digest・clean HEAD・未使用namespaceを下記の承認JSONへ結合する。approval fileはworktree外の`/tmp`等へ排他的に書き、clean SOURCEを保つ。chatの明示承認を根拠として保存し、単なる候補文書を承認とみなさない。
3. activateはpackage版とPython/packageのSOURCE/binary hashを読み取りsealし、新claim directoryを一度作成し、新grantを発行する。旧SOURCE・旧grant・旧STOPは使用しない。
4. executeはgrant/承認/contract/SOURCE/runtime/namespace/消費状態を検査して消費を固定し、入力生成1processを起動する。snapshot/receipt封印に成功した場合だけ4 caseを別々のfresh interpreterで起動する。1件でも失敗した場合は科学retryせず部分証拠を保存する。
5. event/価格join・数値gate・20精度行と全子孫回収を検証し、結果とSTOPをローカルcommitする。N/G nullを成功費用に置き換えず、優劣・希少母平均の保証を推測しない。追加科学stage・push・詳細科学レビューへ自動移行しない。

現在指定版が入っているproject専用環境：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-execution-foundation-20261023
H4_PYTHON=/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

上記BLAS/OpenMP指定は小行列と4独立caseの入れ子thread overheadを避けるprocess内設定であり、CPU/RAMの共有上限制御やOS設定を変更しない。runtime seal後に環境を変更しない。

明示承認受領後のapproval JSON生成例：

```bash
"$H4_PYTHON" -B - <<'PY'
import json
from pathlib import Path
from trotterlib.h4_execution_contract import ROOT, completed_contract, git
from trotterlib.isolated_qiskit_execution import digest
plan = completed_contract()
approval = dict(schema='h4_explicit_batch_approval_v1', SOURCE=git('rev-parse', 'HEAD'),
    contract_digest=digest(plan),
    pack_namespace=str(ROOT/'artifacts/non_df_h4_pilot/2026-10-23/run01'),
    approve_H4_input_and_four_cases_once=True, adopted_proposal=True,
    retry=False, push=False, scientific_review=False)
with Path('/tmp/h4-explicit-batch-approval.json').open('x') as f:
    json.dump(approval, f, ensure_ascii=False, indent=2)
    f.write('\n')
PY
```

明示承認と上記fileが結合された後のみ：

```bash
H4_PACK="$PWD/artifacts/non_df_h4_pilot/2026-10-23/run01"
"$H4_PYTHON" -B scripts/run_h4_non_df_pilot.py --activate \
  --contract-file artifacts/non_df_h4_execution_foundation/2026-10-23/H4_execution_contract_DRAFT.json \
  --approval-file /tmp/h4-explicit-batch-approval.json --pack "$H4_PACK"
"$H4_PYTHON" -B scripts/run_h4_non_df_pilot.py --execute --pack "$H4_PACK"
```

SOURCE、contract digest、namespaceのどれかが異なれば拒否する。既存pack、承認claim、CONSUMED、STOP、case claimは返却しない。同じgrantをrun ID変更で再投入しない。input/case失敗後に上記コマンドを繰り返さない。別SOURCEでの技術修正・科学retryが必要なら、新しい範囲として報告し、今回の一括承認を拡張しない。

保存先は`input_run/input_generation/data/`のMO/Pauli/DF/state snapshotとreceipt、`run/<case>/`のraw/corrected signal・signed bias・normalization・allowance・covariance/MC SE・C/N/G・event/price join、process stdout/stderr/ledger/STOP、pack直下のgrant/approval/SOURCE/summary/STOPである。
