# A/C独立の限定科学実行計画・B未解決条件

2026-11-04 JST。**技術準備時点。実H4科学call0。公開SOURCE後の別途明示実行許可が必要。**
本packageを公開するcommitを新SOURCEとし、最終チャットで40桁SHAと公開照合結果を提示する。
旧D結果89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f、旧SOURCE6942e5176d2dec115ac3c271e48f84377abda915、
旧包含d2e7a7db／f4905fff／m1734840ce9、V1e520e6e3／V278c49d0bは無変更で保全する。
対象はH4/隣接1.00Å/STO-3G/保存legacy DF12、Partial L_D3/δ=.02、保存physical state ray・4電子sector70、
m0..17の保存q/r/K/N/α/β/ε/E_ref。α・N・ε・U_cat・energy/成功目標は変更しない。

## SOURCE・入力identity

`run_three_gaps.frozen_source()`は次の8ファイルを実行HEADと照合する：
`common.py`、`alpha_adapter.py`、`reference_adapter.py`、`decoder_finite.py`、`run_three_gaps.py`、
`fixed_inputs.json`、`PROOF_AND_SCOPE.md`、本計画。
直接使うinput/SOURCE17ファイルのcommit/path/blob/SHA256は[fixed_inputs.json](fixed_inputs.json)に固定。
主な入力は以下。H/stateの新しいdecodeはCを許可した実行時だけ行う。

| 入力 | 固定commit | path・役割 |
|---|---|---|
| D_RESULT | 89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f | `artifacts/rpe_branch_energy_connection/2026-11-04/run01/RESULT.json`、SHA256 a80ce0376ff2e29241911fee901f327a49bfb7a33495d340caf7236370712b68。既存phase/branch/alias/literal energyを再利用 |
| AXES | d2e7a7db683e116243eae061c0e4c87865054e9e | `docs/research/rpe_18round_ideal_enclosure_execution_20261104/AXIS_RESULTS.tsv`、SHA256 0346da604f125e41c5afd073a91e8eb46afe4c2e1c950937d5de9eb33e588e0d。保証付きU_cat_upper/N/ε/α |
| ROUNDS | f4905fff592351d33ba07b927892fb9532c99711 | 旧包含準備`fixed_inputs.json`。36軸の保存literal binding |
| H | 4591b8d75a2a6a55c74a51225f0c61648023c8ee | manifest記載の`.hamiltonian.npz`、SHA256 13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b。保存DF係数 |
| STATE | e22d7718c47358ff650dbc04e8e72f6ab1c8bf41 | `artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz`、SHA256 793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4。raw stateと完全sector basis |
| 直接再利用SOURCE | manifestの各commit | 旧D adapter/Arb helper/V2純関数loader/DF整数excitation関数。decoder・旧float materializer/eighはCで呼ばない |

## 承認対象と科学call範囲

ユーザーが別途許可する対象は**A、C、またはA+C**を区別する。BはH4実行SOURCEが未成立で、本計画の科学実行対象外。
承認はユーザーの指示そのものであり、承認原文保存・証明書・grant発行等の仕組みは作らない。
ここに新しい固定grant条件はない。既存の他研究grant/ledger/marker/STOPには触れない。

| 対象 | 数値条件 | 許可後の新科学call | 成功判定 |
|---|---|---|---|
| A | Arb160 bit・各軸1 pass | 保存36軸のtと指数上界各1回、exact上端和1回 | 全t>0、外向きp_upperのexact和≤1/20 |
| B | 未設定／H4 call0 | なし | primitive/全距離の一様証明が未取得のためUNKNOWN固定 |
| C | Arb/Acb256 bit・1 pass、trial lower offset=1e−8 Ha固定 | 保存H/state decode各1回、完全sector/係数Hermitian検査、T(h)+12 T(g)構成、12 square、1 LDL*、1 norm²/Rayleigh、1 η/energy移送 | 全pivot下端>0で最小固有値bracket取得。保存目標へのu_E+η_ref strict比較とalias marginを別判定 |

Cのtrial offsetは検証用shiftであり、η_refの仮定やenergy目標の変更ではない。
係数のexact Hermitian性が確認できなければCはUNKNOWN。許容差によるsymmetrizeを行わない。
全pivotの正値が未確認ならshift変更・精度追加・pivoting・別固有値solverへの切替はしない。
Rayleighのstateを新たに最適化・対角化で作り直さない。

A/Cは別々の判定を保存する。一方の数学的条件が未確認でも、他方が独立に許可されていれば予定の1 passを行える。
入力identity・runtime・SOURCE不一致は全体preflightで停止する。片方の実行例外はその課題のUNKNOWNとして保存する。
旧Dのphase/branch/energyや36軸mean包含を再計算しない。

## runtime・高速化・資源

Python3.11、IEEE binary64、NumPy1.26.4、python-flint0.9.0／FLINT3.6.0の既存環境を使う。
V2 loaderのimport依存SciPyは既存1.14.1を使用する。install・共有環境変更は行わない。
Arb/acbの外向き包含とexact rational端点を保持し、出力表示を根拠にstrict判定をやり直さない。

Aは36 scalar、Cは70次元、12個のnative FLINT積と一度の下三角LDL*。
T(g)と整数excitation mapを一度構成し、entryごとに分子・DF・Fock写像を作り直さない。
全スペクトルの対角化、全RPE候補・全測定履歴の列挙を避ける。
人工統合検証でこの経路を確認した。小さい固定workloadのため既定ではA/Cを同じprocessで逐次評価し、
process pool/GPUへの転送を必須化しない。実行前にCPU affinity・GPU可用性・MemAvailableを確認し、
native FLINT経路の必要資源を使う。CPU/RAM/workerやphase/total wall-timeに人工上限を設けない。
共有環境・他ユーザーprocess・他研究・Codex C・SERVER1/2の作業領域は変更しない。speedup倍率を主張しない。

## 出力・STOP

namespace案：`artifacts/rpe_three_gaps_validation/2026-11-04/run01/`。今回は未作成。
許可された対象を明示した**初回1 run・各1 pass**で新directoryに書き込む。既存outputは上書きしない。
`A.json`／`C.json`は選んだ対象だけ、`RESULT.json`には個別結果・未実行課題・B UNKNOWN・SOURCE/inputを記録する。
省略した課題を別のnamespaceで自動起動しない。retry/resume、精度追加、予算/decoder/対象変更は自動では行わない。
結果に合わせてαやtrial shiftを調整してPASSにしない。保存値・原出力・旧STOPは保全する。

新しいsampling/RNG replay、回路build/compile、費用評価、分子生成、SCF/DF生成、
PF/RTE operator生成、eigsh/eigh、H4 decoder全域走査は0。
新しい科学作業は上記A/Cの許可された範囲だけ。

科学実行なしの照合：

```bash
<research-python> docs/research/rpe_three_gaps_preparation_20261104/run_three_gaps.py
```

公開SOURCEへの**A+Cの別途明示許可後だけ**の例：

```bash
<research-python> docs/research/rpe_three_gaps_preparation_20261104/run_three_gaps.py \
  --run --tasks A C --output artifacts/rpe_three_gaps_validation/2026-11-04/run01
```

AのみまたはCのみを許可する場合は`--tasks A`または`--tasks C`とする。
A/C成立後もB、固定PRNG/IID、全実回路・compiled/noise・実state preparation・分子ground energyへ拡張しない。
結果・研究記録の必要な更新と公開照合をまとめて行い、mandatory STOP。

今回の技術準備もcommit/push・remote SHA/公開成果物照合後mandatory STOP。
`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、両既存bridge U=nullを維持する。
