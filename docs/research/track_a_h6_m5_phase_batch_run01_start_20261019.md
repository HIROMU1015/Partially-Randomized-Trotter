# H6 full m5_best：1qubit位相蓄積修正の一回再計算を起動

2026-10-19。**利用者の再計算承認とfresh gateに基づき、問題のcosine一軸を起動した。最終のraw gateとnative exit/reapは未監査。** 共有Qiskitを変更せず、元の高速Rust再合成器の位相蓄積だけを隔離したpassで扱う。旧STOP・58受理軸・AD03保留を保持し、終了後mandatory STOP。pushは行わない。

SOURCE：`96cd7cd4c3aff6558d8a6b72e8e6074dcb39a950`。plan SHA256：`96b278e817cf97ec03aedc208d5d0903dfae10a297c8449830a3cafe303e794c`。branch：`track-a-h6-m5-phase-batch-run01-20261019`。

[直前の調査と修正案](track_a_h6_m5_postrepair_phase_static_20261019.md)、[前回のSTOP結果](track_a_h6_m5_phase_repair_run01_results_20261019.md)、[今回のscope](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01_start/2026-10-19/plan_scope_projection.json)、[起動snapshot](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01_start/2026-10-19/start_snapshot.json)。

## 1. 承認・対象・重複計算の削減

ユーザーの「問題だと思われるところを再計算して」を、提示済みの位相蓄積修正を実装・人工検証し、固定H6 cosine一軸を一回実行する承認として記録した。新SOURCE/plan/grantを使い、旧grantを再使用していない。[承認と一回消費のprojection](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01_start/2026-10-19/approval_and_consumption.json)。

対象はH6直鎖1 Å/STO-3G、保存tol-only1e-8 DF19、full m5_best/p19/q2、T=.8/P=0、symmetric-directional/full Hadamard cosine、保存low状態一つ。公式は`PF_M5_BEST_20D64C2_BINARY64_V1`。Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・num_processes1を維持するが、新しいcompiler-policy/pipeline identityの実験であり、旧費用へ混合しない。

前回の明示basis変換後QPYとvector、誤差2.326360070803798e-11のPASS receiptを再利用する。前回全体のSTOPは残したまま、SOURCE02cb8b12…、終了証拠、独立した終了監査、保存artifactのSHAに結合した受入条件を新契約へ明示した。保存済みのPASS部分を再生成しないため、実施量は**compile一回・compiled回路の作用一回**に縮小した。新loweringとlowering後再生は0。

3raw検査を維持する。

| 照合 | 今回の扱い | 基準 |
| --- | --- | --- |
| 明示basis変換後 vs 元wrapper vector | 保存PASSをSHA結合して再利用 | 1e-9未満、前回2.326e-11 |
| compiled vs 保存変換後vector | 新たに判定 | finiteかつ1e-9未満 |
| compiled vs 元wrapper保存vector | 新たに判定 | finiteかつ1e-9未満 |

新compile前に保存入力を検証し、上から順に判定する。FAILなら即STOPし、以後の照合は未実施として残す。phase alignment・vector正規化・事後の位相補正・誤差閾値変更はしない。sampling、signal、SCF/DF/state/reference生成、sine、shot/費用会計、fit、rankingは0。主比較AD03や後続Coreへの承認ではない。

## 2. 隔離実装

[compiler pass](../../scripts/m5_phase_batch_compiler.py)、[契約](../../scripts/h6_m5_phase_batch_contract.py)、[native adapter](../../scripts/h6_m5_phase_batch_native.py)、[launcher](../../scripts/run_h6_m5_phase_batch.py)、[metadata準備器](../../scripts/prepare_h6_m5_phase_batch.py)。

元のaccelerated Rust kernel、行列→gate列生成、basis/target、opt1の反復controllerを再利用し、1qubit再合成pass一箇所をproject専用passへ置換する。kernel内では、定数ParameterExpressionのproject内subclassが各binary64位相寄与を正確なFractionで受け、継承するsymbolic式にも同じ有理数値を保持する。置換loop途中のfloat化は拒否する。pass終了時に数学的πで折り返して一度だけbinary64に丸め、DAGを通常の数値phaseへ戻す。shared classやruntime fileのmonkeypatchはない。

この方法は、記録済みH6誤差を引く処理ではない。元kernelが実際に生成した各位相寄与を蓄積し、count・増分bit列のSHA・正確な和・最終phaseをpass別ledgerへ残す。保存誤差やstatevectorをcompilerへ渡さず、角度簡約・PF係数・opt levelを同時に変更しない。実H6の改善、原因の確定、実回路のgate列完全一致はまだ主張しない。

## 3. 人工検証・fresh gate

[人工13件](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01_start/2026-10-19/artificial_validation.json)PASS（0.489秒）。正確加算と途中float拒否、元Rustへのdispatch、人工のgate metadata一致、正負/2π角のraw全演算と制御付き演算、固定profile、出力の上書き拒否、DRAFT実CLI拒否、旧/拡張grant拒否、stage一回限り、偽workerのexit/reap・科学STOPを確認した。人工回路の1e-12はtest条件であり、実H6の1e-9未満を変更しない。作者自身の確認で、独立AIレビューではない。

[起動前22項目](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01_start/2026-10-19/preflight_checks.json)PASS、12種のmetadata改変を拒否。[使用closure](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01_start/2026-10-19/source_input_binding.json)は2,011fileで、旧full Qiskitに加え固定SymEngine配布物のRECORD/SHAも確認した。[保全](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01_start/2026-10-19/preservation.json)は旧証拠2,570fileが起動snapshotで不変。

[fresh gate](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01_start/2026-10-19/fresh_gate.json)を通過し、19:00:52 JSTに起動。直前の実効空きRAMは約468 GiB。今回の入場proxyは直前実測peak RSSの2倍と保存QPY bytesの2倍から求めた約10.3 GiBで、厳密な必要量の上界やruntime capではない。CPU/RAM/wallの任意のhard capはない。

productive worker一つ、監視parentと外部wait4は別process。一軸内の依存関係を守り、保存PASS再利用と元Rust kernelで重複計算・Pythonでのgate再合成を避ける。無関係なworktreeのHEAD/statusは凍結しない。host PSI・CPU競合・監視遅延だけで科学workerをkillせず、実OOM・own異常・使用入力/結果破損・数値不一致はSTOPする。

## 4. 結果との区別・次の担当

この資料は起動時点の記録で、保存最初のPASSを含めても**全3検査のPASSやH6の修正完了を認定しない。** 最終QPY/vector・raw差・phase ledger・native exit/reapは終了後の監査が必要。QPY/NPZ、有効grant/config、実pathと詳細logはprivateに保持する。

**Codex（計算終了後）**：3検査の実施/reuse/未実施、exact phase ledger、SOURCE/input/runtime、結果SHA、exit/reap、旧証拠保全を確認し、PASS/STOPのどちらでも結果を報告・local commitしてmandatory STOPする。

**ユーザー（結果報告後）**：この一軸結果を確認し、次の修正検証やsineへ進むか判断する。後続計算・費用受理・主比較採用へ自動移行しない。

公開pushは0。
