# B2共通プログラム：公開入口と現在の機能

固定SOURCE：`08b75526b561cbcf97ae58c7a75fefeb226911dd`、
branch `codex/b2-rpe-method-optimizer-20261116`。
入口は`--help`と公開codeを確認済み。保存H4の実行照合は、このSkillのREPORTを正本とする。

## 起動する実物

- CLI：`scripts/run_rpe_method_optimizer.py`
- 実装：`src/trottertracks/rpe_method_optimizer.py`（`prepare` / `optimize`）
- 公開H4問題入力：`configs/rpe_method_optimizer/h4_df10_problem.json`
- CLIの引数は`--problem`、`--output`、任意の`--ingest`だけ。

リポジトリrootで、対応するproject Python環境を使う。SOURCEがcheckoutに無ければ、
公開commitの**自分の独立worktree**を用意する。B2の作業worktreeや他担当branchを書き換えない。
このSkillの所在と実行programの所在が異なっても、それぞれのSOURCEを記録して使う。
新worktreeがsparse checkoutなら、同じ公開SOURCEの`configs/rpe_method_optimizer`も展開して
問題入力が存在することを確認する。別の問題設定を代用品にしない。

保存H4の起動例（project環境をactivate済み）：

```bash
RPE_COMPARE_OUT=$(mktemp -d /tmp/rpe-cost-compare-h4.XXXXXX)
PYTHONPATH=src python -B scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --output "$RPE_COMPARE_OUT"
```

出力先は毎回新しいdirectoryを使う。CLIは既存出力を上書きできるため、過去のrunを再利用先にしない。
`--saved-only`、`--resume`、`--epsilon`、`--budget`等の未実装引数を作って渡さない。
要求精度や予算は、programの既存problem schemaの入力として渡す。

## 入力の対応

この版のJSON schemaは`RPE_METHOD_OPTIMIZER_PROBLEM_V1`。

- Hamiltonian：`problem`。`hamiltonian_data`にsnapshot/receiptのlocatorとSOURCE検査情報が必要。
  データreceiptからfragment数や次元を取得するのはprogramの仕事。
- target：`target`。精度：`accuracy.epsilon_E_decimal`（Ha）、decoder/loss/claim scopeも`accuracy`。
  元H/state/snapshot、保存target proxy等の採択事項と照合し、勝手に物理ground資格へ変更しない。
- RPE設定：`rpe`。候補範囲・shot上限：`methods`。これらは採択済み問題設定から引き継ぎ、
  Skillが個別LD・shotを利用者に選ばせたり、新しい範囲を正式採択したりしない。
- 保存証拠：`evidence`。探索則・予算：`search`。公開H4設定は`max_new_q3_per_method=0`。
  実行範囲と許可に一致することを確認し、既存problemを上書きせず必要な入力copyを作る。

H4のsampleはprivate snapshot/receiptの絶対pathを含む。利用可能性を確認し、無い場合は
正しい同一SOURCEのlocatorが必要と報告する。別データやdummyへ差し替えてPASSにしない。

## この公開版で出来ること・未対応

- 保存catalogの再利用、候補列挙、各候補自身の保存単価によるexact費用、有限shot探索、
  保存Q3による応答、方式別の確認済み最小値、次に不足するQ3要求の出力。
- 方式menuは`FULL_PF`、`TRUNCATED_PF`、`PR_PAULI`、`PR_DF`の4方式。
  Full-H pure RTEを含む5方式比較の完了とは表示しない。
- 新しいsignal・価格取得・compile・科学Q3を起動するexecutorは未実装。
  予算を正にしても未対応は解消しない。`pending_request`を返し、Skillから旧個別runnerへ迂回しない。
- `--ingest`はprogram形式の外部結果を取り込んで次要求を更新する入口。
  正しいSOURCE/同一物理候補/配分/評価契約の既存結果に限る。
  完全なjob checkpoint再開や科学workerの再起動機能ではない。未実装の再開手順を補わない。

CLIは診断用`INGEST_DEMO.json`と`FIXTURE.json`も作り、明示された人工MSEを使う。
これらは科学Q3ではなく、主比較の`RESULT.json.methods`へ混ぜない。
保存H4の接続テストではこの既存の人工診断を認め、新しい科学計算は起動しない。

## 出力を読む

`RESULT.json.methods`が主結果、`PER_METHOD.csv`がその表示表。
方式ごとの`best`に構成、`N_X/N_Y`、MSE区間、`cost/cost_display`、証拠locatorがあり、
`stop_reason`、`held`、`pending_request`、`search_range`、`not_ready_candidates`を併記する。
`NEEDS.json`は不足証拠/次要求、`CHECKS.json`は実行SOURCEと診断。
`REPLAY.json`、`OWN_PRICE.json`は別の補助結果。`FIXTURE/INGEST_DEMO`は人工結果である。

`methods.*.new_calls`とtraceのanswer SOURCEから、主結果は保存のみであることを確認する。
人工fixtureのcallback回数を新規科学Q3件数として合算しない。
この版の費用は保存価格条件に束縛された値であり、MODEL参照値/条件付き閉包を正式E[C]やGへ昇格しない。

## 公開版が更新されたとき

新しい固定SOURCEのhelp/schemaと科学executorの有無を読み、この文書を実物に合わせる。
H4保存入力の直接実行とSkill手順の実行を同じSOURCE・入力で照合し、構成ID、X/Y整数配分、
exact費用、MSE区間、停止理由、保留/未対応を比較する。時刻や出力directoryだけの差と科学結果の差を区別する。
接続確認で新しい科学Q3・signal・sampling・compile・fitを起動しない。
