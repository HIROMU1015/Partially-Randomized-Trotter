# Kernel上界の調査・接続実装

2026-11-04 JST。技術的到達点は**CURRENT_KERNEL_BOUND_DERIVED**。
導入済みbinaryで使用経路を確認し、C-contiguous complex128 GEMVT/DOTとscale/subtract、
glibc non-FMA hypotについて、明示したFP mode/range条件の下の保守的上界を導出した。
実H4の上界・norm・予算・全support guardは未評価で、科学実行SOURCEはまだ未完成。

## 再利用と変更範囲

固定参照は前段commit `728c90a2d4d4335ef5b7237808c15b182e694bb9`。
`virtual_bridge_adapter.py`のKernelBounds/Growth/action_growth/rotation_growth/trajectory_growth/
signal_error/guard_completion/classify_axesを再利用した。
`run_virtual_bridge.py`と元test、VALIDATION、計画を確認し、固定SOURCEを変更していない。
既存の8 fieldへ導出係数を直接渡す。

| 新規ファイル | 役割 |
|---|---|
| [kernel_bounds.py](kernel_bounds.py) | path_boundsが導出済みκ/aを既存KernelBoundsへ接続。FP mode/profile/layoutを区別するselector、有限性の十分条件screen、hypotのexact rational証明budgetとabs上端 |
| [artificial_dispatch.py](artificial_dispatch.py) | saved H/state/DFを読まないdeterministic artificial arrayでdispatch確認。kernel変更・代替実装ではない |
| [trace_dispatch.gdb](trace_dispatch.gdb) | 自分の子processのCBLAS/実kernel/ufunc/hypotとMXCSRを読む。共有library・全体設定を書き換えない |
| [dispatch_observation.txt](dispatch_observation.txt) | 実際の最終gdb出力の必要行を抽出。仮想のtraceや推測による穴埋めなし |
| [test_kernel_bounds.py](test_kernel_bounds.py) | exact complex演算、subnormal、FMA/非FMA、reduction、hypotの解析budget、null/strict/科学停止を人工入力で検証 |

SSE/AVXの命令graphを確認できたので、特定reduction順に依存しない保守的forest boundを採用した。
kernel全命令の形式検証やBLAS全体auditを完了したという主張ではない。
公開release SOURCEとdev buildを同一視せず、実binaryのpathを根拠にする。
以前web取得に失敗したSOURCEの一部は、今回GitHub pluginで取得できた。
取得経路の失敗をSOURCE不存在の証拠にしていない。

## 実環境と抽象モデルの区別

今回確認した事実：人工C/complex128 pathのPrescott GEMVT/DOT実行、threadごとの出力分担、
NumPy AVX2/FMA ufunc、CPython→glibc hypot、RC/RNE・FTZ=0・DAZ=0の観察。
導出はこのbinary/profile/layoutと、同じ算術状態の継続を前提にする。
別BLAS、F/strided matrix、別FP modeへ一般化した抽象仮定だけの係数は採用しない。

`selected_path_contract`は提示された観察profileを照合して、限定したpathの係数を返すだけ。
将来processのFP modeの安定性や、実H4入力の有限性を自動的に証明するものではない。
返却値は常に`scientific_execution_ready=false`とし、必要な将来チェックを併記する。
未来の科学runnerへ任意の外部係数を注入するCLI、承認証明機構、grant/ledger等は追加しない。

元`installed_kernel_contract()`は固定SOURCEの記録としてUNRESOLVEDのまま保全する。
旧--runもそのまま停止する。新helperを使う統合SOURCEを作る段階で、今回の限定profile・range・mode条件を明示的に接続する。
この技術結果だけで旧実行をPASSへ格上げしない。

## 検証

新人工13件、既存bridge adapter16件、前段bridge math15件、**計44件PASS**。
実行条件・所要時間・コマンドは[VALIDATION.txt](VALIDATION.txt)。
testには人工70次元も含むが、分子/H/state/DF入力やm17のtrajectoryは使わない。
数値fixtureの一致はsource対応/数式の代用ではなく、実装の取り違えを検出する補助。
hypotのpolynomial budgetは、全変数区間の誤差伝播をFractionで評価する数学的不等式の確認。

科学runnerの--runは実入力では起動しない。
停止分岐はmetadataをmockし、np.loadを禁止した人工検査だけで確認した。
同様に実H4のcomponent、行列norm、平均、CDF、誤差上端、予算、sampling、回路build/compile、費用評価は0。

## 科学SOURCEに残る作業

1. Codexが、今回の限定layout/profile・全threadのFP mode条件をruntimeで確かめる薄い統合処理を作る。
   matrix/vectorのshape/stride、alpha/beta、実環境のmode継続を確認し、未対応経路はUNKNOWNにする。
2. 保存入力→numeric context一回構成→残差/majorant→finite envelope→guard→strict判定を接続する。
   finite envelopeへは平均でなく全supportのboundary上端を渡し、hypotのrange条件も別途検査する。
3. fresh出力、部分結果、中断記録、96/160/224 bitの固定call範囲を持つ科学SOURCEを公開固定する。
   古い--runのSTOPを解除したり、旧結果を上書きしたりしない。

これらは次段案で、今回未実装・未実行の部分を「完成」と扱わない。
具体的な新科学SOURCEとscopeを固定した後、ユーザーの別の明示実行許可が必要。
今回B/Cの条件付き抽象モデルや別kernelへの研究条件変更は行っていない。

旧TECHNICAL_ADAPTER_PREPARED_KERNEL_UNRESOLVED、scientific_execution_ready=false、
U_exec=null、全体UNDETERMINED、旧INCONSISTENT/u_num=nullと原SOURCE/raw/STOPを保全する。
他branch/worktree、SERVER2 H8、dirty main・他ユーザーのprocess・共有環境を変更しない。
今回成果の公開照合後にmandatory STOP。
