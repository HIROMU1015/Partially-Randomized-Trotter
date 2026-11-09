# 数値検証の並列化・高速化：再利用ガイド

今後の検証で、既存の高速化を探し直さず使うための入口。2026-11-10のH8 Full-H RTEを適用例として、再利用部品・適用条件・実測・科学条件の保全をまとめる。ここにある実行例は新しい科学条件やsamplingの開始許可を与えない。

## 再利用する部品

|目的|コード・入口|そのまま使える範囲と接続部分|
|---|---|---|
|作用検証の並列Aerと小gate cache|[`h8_full_h_parallel_aer.py`](../scripts/h8_full_h_parallel_aer.py)：`configure` / `action_circuit` / `simulate_statevector`|分子loaderに依存しない。Qiskit circuitと初期vectorを受け取る。既存provider libraryとscriptsのimport pathを用意する|
|人工作用・定義変更テスト|[`test_h8_full_h_parallel_aer.py`](../tests/test_h8_full_h_parallel_aer.py)|正負角・open/closed control・global phase・adjoint・入力非変更・gate定義変更後のcacheを確認|
|thread・作用経路の人工比較|[`check_h8_full_h_parallel.py`](../scripts/check_h8_full_h_parallel.py)：`benchmark` / `spawn`|この2 commandは人工入力だけ。`validate` は保存H8入力での科学検証なので、その研究の実行範囲に従う|
|実資源の観測|[`pf_common_run_contract.py`](../scripts/pf_common_run_contract.py)：`resources()`|affinity、idle CPU、適用cgroup上限、available RAM、diskを読み取り観測。この関数の流用に同fileの過去のStage/grant契約は不要|
|content-aware fingerprint|[`h46_b3_fast_fingerprint.py`](../scripts/h46_b3_fast_fingerprint.py)：`use_fast_fingerprint` / `ContentKeys`|元のcanonical v2 SHAを保つ。key/memoはcall-localで、呼出し中の入力非変更が前提|
|compiled価格cache|[`h46_long_time_native.py`](../scripts/h46_long_time_native.py)：`compile_cached`|完全なnative wrapper fingerprint＋固定compiler identityでreuse。別compilerや別回路に価格を転用しない|
|局所gate由来のcompiler位相会計|[`h8_compiler_phase_bookkeeping.py`](../scripts/h8_compiler_phase_bookkeeping.py)|現在のH8固定compiler向け。小Operatorで実際のgate/parameterを確認する。他compilerや未知gateへ位相則を無検証で流用しない|
|vectorized event作用|[`h8_full_h_pure_rte.py`](../scripts/h8_full_h_pure_rte.py)：`tensor_action` / `EventOracle`|little-endian局所作用とsaved-basis algebra。`tensor_action` は1～4量子ビットの局所行列用。分子固有のbasis registry/state/scalarは呼出し側から渡す|
|spawnと資源連動のtrajectory並列化|[`h8_full_h_price_parallel.py`](../scripts/h8_full_h_price_parallel.py)|設計例。候補、入力、reference、保存先、seed lawはH8固定なので、runner全体の他研究への無変更適用はしない|

動作確認環境はPython3.12、NumPy1.26.4、Qiskit1.3.0、Aer0.15.1 CPU double。別version・gate family・qubit数へ使うときは、該当する小さな作用テストと性能測定を先に行う。

## 作業の順序

1. 同じ入力と科学条件で、loader、signal、sampling、native build、fingerprint、compile、vector action、I/Oの時間を分ける。CPU使用率はprocess単体とサーバー全体を区別する。
2. 既存のcache・sparse/sector・vectorized関数を確認する。大きなHamiltonianや全register unitaryを密行列化して高速化しない。
3. 独立taskのprocess並列と、1 vector action内のthread並列を別に決める。前段PASS後にのみ後段を開始する逐次依存は維持する。
4. 小さな人工回路で、thread数と旧経路に対する数値一致・実時間を測る。実資源とquotaを尊重し、固定の少数core上限や過去のworker値を無条件で継承しない。
5. 新しい作用経路では、正負時間・actual adjoint・control枝・両軸Hadamard/compiled作用のうち変更に関係する部分を確認する。必要な科学検証は許可された研究範囲内に限る。
6. 固定SOURCE・入力・candidate/cohort/seed・compiler・shot/normalizationを保ち、freshな技術出力領域に結果を残す。価格用回路を変える最適化は、検証用作用経路の高速化と区別する。

## Aer作用検証の接続例

対象研究で用意した、対応provider libraryと`scripts`をimportできる環境で使う。`simulate_statevector` 自体はH8入力・referenceをloadしない。

```python
from h8_full_h_parallel_aer import configure, simulate_statevector

# threadsは、その環境と回路で実測した値を渡す。
# checkpoint_directoryは新しいdirectoryかNone。
configure(threads=measured_threads, directory=checkpoint_directory)
after = simulate_statevector(circuit_to_check, initial_vector)
```

現在の実装は、Aerが直接扱えないgateを最大3量子ビットの厳密Operator行列に置き換える。既存のcontent-aware keyでgate class・parameter・control・definitionを確認し、同じ呼出し内で再利用する。一般の大きなcustom gate、条件付きclassical操作、未知のlayoutを通す汎用lowererではない。

gate行列はwritable C-order NumPy配列を使う。今回の人工検査ではread-only C-orderをAerへ渡すと不一致を生じたため、採用前に修正した。`ContentKeys` のidentity memoを異なる呼出しへ持ち越さない。入力は1回の呼出し中に変更しない。これらは上の人工テストに含めてある。

global phaseを明示的に保持し、statevectorからphaseをfitしない。価格取得対象のnative/compiled circuitはそのまま保存・compileし、局所行列への置換は検証用の作用計算だけに使う。

現在の同期Aer実行関数は、Python executor threadを残さないため、使用中のAer 0.15.1の内部実行関数を利用する。Aer更新時にはこの接続とthread挙動を再確認する。

## workerとthreadを組み合わせる

- 一つのvector actionを高速化するときは、実測したAer thread数を使う。BLASによる小さな局所行列計算とは設定を分ける。
- 独立trajectoryの価格化はprocessで並列化し、cosine/sineは同一trajectoryとしてpairedで保存する。worker完了順でcandidate・seed・集計対象を変更しない。
- 多thread Aerの後にworkerを起動する場合は、今回の例では`spawn`を使用した。worker initializerで同じ保存contextを復元し、親の大きなmodule/contextをpickleしない。
- task数、実際のidle CPU、空きRAM、測定したpeakを使って投入数を決める。逐次代表の重いvector検証と、その後のcompile中心workerではpeakが違うので、新しい実測で見積りを更新する。
- 他利用者のprocess、共有venv/OS/cgroup/scheduler設定を変更しない。必要な環境変数・cache・outputの変更は対象job内に限定する。

## 実測と限界

[H8の測定記録](../artifacts/resource_applicability/df_h8_full_h_pure_rte_light/2026-11-10/performance_check_parallel.json)では、17q・3681 native instructionsの人工回路で、旧定義展開＋1 threadが4.447秒、新局所行列経路＋32 threadsが.861秒、約5.17倍。旧経路とのvector差は7.92e-13。

[thread単独の記録](../artifacts/resource_applicability/df_h8_full_h_pure_rte_light/2026-11-10/performance_check.json)では1→16～32 threadsで約2.5倍、64 threadsは遅かった。32という値はこの測定に対応し、将来の全qubit数・回路への推奨値や上限ではない。短い人工測定による僅差をthread数の普遍的最適性と扱わない。

保存H8の条件付き小event・両軸compiled作用は[検証記録](../artifacts/resource_applicability/df_h8_full_h_pure_rte_light/2026-11-10/parallel_validation/parallel_validation.json)でPASS。full22528-event trajectoryとn9の最終資源費用は計算中であり、人工速度倍率を全runへ掛けた終了時刻や費用保証は出さない。

## checkpointと科学標本

`*_action.json` と `*_vector.npy` は計算済み状態の保存で、`ACTION_COMPUTED_NOT_YET_CHECKED` はcorrectness PASSではない。現在のrunnerはこれらから任意の途中状態を自動resumeしない。SOURCE/input/作用回路の対応が確認できる明示的な再開処理がある場合に限り利用する。

同じcandidate/cohort/index/seedを技術移行先で再計算しても独立標本は増えない。旧partialを別sampleとして混ぜず、保存event列の一致を確認する。新しいseedで失敗標本を都合よく置換しない。未観測rare eventの条件付き価格はunknown/nullを保持する。

## 関連する過去の実行層

- [共有サーバー向け実行基盤](server_parallel_validation_execution.md)：task manifest、dry-run、checkpoint/resume。
- [PR-2 S2の並列実行層](pr2_s2_parallel_execution.md)：固定cellのspawn、seed・集計順・paired価格cache。

これらの元研究の契約・STOPは保存する。記録された既定worker/thread/RAM数を新しい研究の恒常上限にはしない。新研究には、その科学条件に合う既存機能だけを選んで接続する。

H8の実際のjobと残作業は[引継ぎ](../artifacts/resource_applicability/df_h8_full_h_pure_rte_light/2026-11-10/HANDOFF.md)。追加の検証・最適化は対象研究の実行許可の範囲で行う。
