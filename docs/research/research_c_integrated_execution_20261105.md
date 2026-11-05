# 研究C統合実行の接続と保存配置

2026-11-05 JST。科学条件は `9ff0ef7449c5a53e559be837647937cf1d011391` の統合契約（最大172,835 logical units）のまま。新接続コードは固定181ファイルを変更せず、既存APIを指定順で呼ぶ。契約のstatus文字列や過去SOURCE/STOPを書き換えない。

## 保存先

実行rootは `/media/abe/T7/research-c-integrated-20261105`。ホスト側T7はrw、実行sandbox内のro表示とは区別した。root内へ固定SOURCE closure、保存input、候補非依存shellとその来歴をbytes同一で配置し、fresh outputの契約相対pathを保つ。symlinkによるroot外参照やmount変更は行わない。旧T7アーカイブも変更しない。

## 接続

- `c_integrated_fit_worker.py`：paired specを一回生成、既存gamma0 restoreと既存有限trajectoryを呼ぶ。raw、checkpoint、局所Callsを保存する。
- `c_integrated_downstream_execution.py`：candidate selection1268、同J/U D141、final618を既存APIで実行。候補/PF/native共通stateを保持して二重計算しない。
- `c_integrated_execution_rpc.py`：物理sector評価と長寿命native workerへのJSON接続。元contextはpickleで移し、原物理HやCARを再計算しない。
- `c_integrated_execution.py`／`scripts/run_research_c_integrated_execution.py`：固定依存・選択・recipe凍結・比較・最終STOP。`--execute`なしはpassive preflight。旧run()や消費済みgrantを呼ばない。

新規driverは実行SOURCE補足へ固定する。固定済みoptimizer・目的・精度・PF/RPE・compiler・会計のSOURCEは同じbytesを維持する。配線の人工検証はpytest34件PASS（新25＋既存adapter9）と人工sector context pickle往復。科学性能・GO・外部再現の証拠ではない。

## 並列化と資源

共通原H評価は一つのeval processで再利用し、二neutral restoreは別processへaffinityを分割する。独立18trajectoryには利用可能affinity全体を分割する。native側は候補process数を空きCPU/RAMと既存run実測peak RSSから選び、各process内の既存resource_planがそのaffinityに応じてround/nativeを並列化する。BLAS/OpenMPは自job内で協調、JAX/Matplotlib/Numba/cache/tempはT7のjob領域のみ。速度向上率は未測定、wall/CPU/RAMの人工capは置かない。

## STOPと記録

個別taskの正常終了はworker terminalであり、統合継続が許される。新しい中間mandatory STOPを作らない。失敗時はpartial/journalを保存して依存処理を止め、retry/resumeしない。最終 `INTEGRATED_STOP.json` は統合全体で一回保存する。科学実行状況はT7の `artifacts/research_c_integrated/2026-11-05/execution_progress.jsonl`、各fresh出力のraw/journal/result、最終STOPで確認する。局所Callsを二重加算せず、stage集計は完了stage分を示す。失敗途中の実消費は局所journalを含めて別に集計する。

重要科学レビュー・新surrogate・集合外条件・論文化GOへ進まない。得られた結果の根拠・未資格範囲・raw所在を報告時に更新する。
