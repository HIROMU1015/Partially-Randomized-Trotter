# H6 full m5_best：明示basis修正・cosine一軸の再計算起動

2026-10-19。**利用者の明示承認とfresh gateに基づき、一回再計算を起動した。結果・最終native exitは未監査。** 旧STOP・58受理軸を保全し、新しい入力生成政策として独立実行する。終了後はmandatory STOPし、後続計算へ自動移行しない。

SOURCE：`02cb8b124302af334d0b02fe53213d3e5a932b78`。plan SHA256：`9c70e90b354537732cc734ad5bf2a9ec5b440d49d900bd17b3006093ed2f8491`。branch：`track-a-h6-m5-phase-repair-run01-20261019`。

[修正候補の調査・人工検証](track_a_h6_m5_phase_repair_design_20261019.md)、[一回scope](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01_start/2026-10-19/plan_scope_projection.json)、[起動snapshot](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01_start/2026-10-19/start_snapshot.json)。

## 1. 承認と固定した範囲

ユーザーの「実際にH6の再計算まで進めてよい」を、直前に提示した固定cosine一軸・最大1compile/2再生への承認として、新しいSOURCE/入力/plan/grantへ結合した。[承認・消費のprojection](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01_start/2026-10-19/approval_and_consumption.json)。過去のgrantを再利用せず、新grantを一度だけ消費した。公開projectionは正本plan・有効grant/configの代わりにならない。

H6直鎖・隣接1 Å/STO-3G、保存tol-only1e-8 DF19、full m5_best/prefix19/q2、T=.8/P=0、symmetric-directional/full Hadamard **cosine**、保存low状態一つ。formula IDは`PF_M5_BEST_20D64C2_BINARY64_V1`。保存wrapper QPY、初期low、通常replayで完全再現済みのwrapper vectorを使用する。元vector NPZ内のlow/beforeとの完全一致も、作用前に確認する。

新SCF/DF/state/full-H reference生成、signal、sampling、sine、shot/費用再会計、fit、ランキングは0。候補探索・他58軸の再取得はしない。新SOURCEの技術修正は孤立したadapterに限り、共有Qiskit・既存native・旧SOURCEを変更しない。主比較AD03の採用保留は維持する。

## 2. 実装と必須gate

[contract](../../scripts/h6_m5_phase_repair_contract.py)、[native adapter](../../scripts/h6_m5_phase_repair_native.py)、[launcher](../../scripts/run_h6_m5_phase_repair.py)、[metadata準備器](../../scripts/prepare_h6_m5_phase_repair.py)を固定した。明示basis変換の数値coreは、人工検証済みprototypeからscope guardを移しただけで、ASTが一致することを確認した。prototype本体は変更していない。scalarを正確に蓄積し、数学的πで折り返して最後に一度丸める。事後の位相補正やgate角度のfitは行わない。

厳密に次の順序で進む。予約は非返却・再使用不可。

1. 保存入力の復元・metadata/array来歴join。
2. nativeを明示basisへ変換し、QPY・receiptを保存。
3. 修正後回路を一回作用させ、保存wrapper vectorとのraw差がfiniteかつ**1e-9未満**か判定。FAILならcompileを起動しない。
4. Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・num_processes1で、cosineを一回compile。QPYを受入判定前に保存。
5. compiled回路を一回作用させ、修正後vectorとのraw差、元wrapper保存vectorとのend-to-end raw差を、**それぞれ1e-9未満**で判定。

raw vectorを正規化せず、phase alignmentを受入へ用いない。数値不一致、非finite、未知gate/parameter、予期しないlayout、使用SOURCE/input/runtime/結果の不一致、native異常・回収漏れ、実OOM・継続不能、保存/SHA異常はSTOP。自動retry・追加compile・再fit・閾値緩和をしない。

PASSしても一軸・一状態の技術pilotであり、旧STOPをPASSへ書き換えない。費用は診断用integer countsだけで、受理費用行は0。compiler選択肢を維持していても入力生成政策が異なるため、新pipeline identityを記録し、旧compiler option hashだけで旧費用との互換性を認定しない。

## 3. 検証・封印・fresh資源

[人工11件](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01_start/2026-10-19/artificial_validation.json)PASS（1.216秒）。数値AST/bit一致、境界1e-9・nonfinite・phase-onlyの拒否、人工13qubitのQPY保存/1compile/2作用/3raw gate、compile前STOP、stage順序、DRAFT/旧grant拒否、spawn終了回収、一回限りを確認した。作者自身の技術検証であり、独立レビューではない。人工回路のcompileを実H6実行とは数えない。

[起動前22項目](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01_start/2026-10-19/preflight_checks.json)PASS。実DRAFT CLIのexecute/supervise拒否と、scope/compile数/再生数/sine/SOURCE/pipeline/input/runtime/resources/output/approval/raw閾値の12種改変拒否を確認した。旧DRAFT/INVALIDも保持した。

[封印](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01_start/2026-10-19/source_input_binding.json)は使用SOURCE・8入力role・過去runtime・mpmath・全installed Qiskit配布物を含む1,964file。Qiskit配布物はRECORDのSHA/bytesも照合した。[保全](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01_start/2026-10-19/preservation.json)は2,422fileが起動snapshotで不変。既存SOURCE/raw/結果/grant/ledger/marker/STOPと、直前のprototype/結果を保護した。他worktreeのHEAD/statusを凍結していない。

[fresh gate](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01_start/2026-10-19/fresh_gate.json)を通過し、17:06:21 JSTに起動。空きRAMは起動前約469 GiB、入場proxy約25.4 GiB。入場proxyは元取得runの実測RSSと候補入力の命令展開上界から求めた計画値で、runtime上限や厳密な必要量の上界ではない。命令数の登録上界は4,076,368。CPU/RAM/wallの任意のhard capは追加していない。

一軸内の必須gateに依存順序があるため、productive workerは一つ。監視parentは別processで動き、さらに外部wait4が終了code/reap/RSS/wallを残す。保存入力・既検証core・既存再生器を再利用する。回路・状態の生成や重複compileを増やす並列化はしない。

各compiler callbackではpass名・時間・global phaseのbit表現・gate countsを観測する。basis入力なので毎回全命令を走査せず、DAGのcountsを用いる。これは低負荷の観測で、gate identityや最初の原因passを証明するものではない。最終gate型/layoutは全件検査する。host/上位cgroupの一時的なPSI上昇・CPU競合・無関係な資料更新だけでworkerをkillしない。

## 4. snapshotの限界と次の担当

この資料は起動時点の記録である。現在のphaseと予約数は上のsnapshot JSONに保存しており、**全3raw gateのPASSや計算完了はまだ主張しない。** 後からこのsnapshotを結果扱いに変更しない。QPY/NPZ・有効grant/config・資源観測log・実pathはprivateに保持した。

**Codex（計算終了後）**：新旧SOURCE・結果SHA・3raw gate・stage数・native exit/reap・保全を監査し、PASS/STOPのどちらでも結果を報告してlocal commit後mandatory STOPする。

**ユーザー（結果報告後）**：新入力生成政策の継続検証・sine取得等を採用するか判断する。今回の承認を後続Core・再会計・主比較採用へ流用しない。

この起動資料はlocal commitのみ。公開pushは行わない。
