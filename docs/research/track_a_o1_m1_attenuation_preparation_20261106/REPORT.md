# O1 M1：r4/r8・q64限定信号検証の実行準備

2026-11-06、SERVER 2（GPUA100-gleap02）。base `8820c41bc5beb32208447fe78fee6854088cbcd5`。**最小runnerと人工テストのみ。実H4入力のdecode、operator/保存値再計算、H4回路build/compile、shot/価格/総費用計算、samplingは未実施。新科学実行の明示許可は未取得で、準備公開後mandatory STOP。** 新しい承認guard・grant発行・承認証明は追加しない。

## 最小接続と再利用

[runner](../../../scripts/run_o1_m1_attenuation_validation.py)は、既存`o1_m1_matched_split.load_h4_a_snapshot/prepare_matched_a`、`o1_m1_circuit_signal_validation.validate_matched_circuit_signals`、独立CAR/Fock/paired-polynomial参照、`o1_m1_resource_bridge.prepare_matched_resources/configure_law/observed_signal_bounds`を呼ぶ薄い接続である。既存SOURCEは変更しない。

`run_o1_m1_c0_pilot.c0_margins`とJSON保存処理を再利用し、**shot計算を含む`account_radius_round`は呼ばない**。有限法のnormalization/attenuation、補正mean・raw複素信号、PF/RTE/model/実装phase、実半径下界と予算判定だけを保存する。価格provider、sampler、optimizer、transpiler、RPE全round入口を呼ばない。

実行時はr=4→8の順に既存validatorを各1回呼び、各回がPauli/DF両辞書を同じ分割で扱う。登録点は`(r,dictionary)=(4,P),(4,DF),(8,P),(8,DF)`だけ。既存validator FAILまたはC0のphase/radius条件不適格が出たら、rawを保存して後続rへ進まずSTOPする。r/K/q/deltaのCLI変更口は設けない。既存出力directoryは拒否し、旧結果・STOPへ書き込まない。自分が新規作成したoutput内で想定外例外が起きた場合は、取得済みraw/rowを保持した`ERROR_STOP`・`mandatory_stop=true`を保存して例外を再送出する。これは科学FAILのPASS化や自動retryを意味しない。

## 固定条件・SOURCE・入力

- H4-A、linear H4、1 Å、STO-3G、保存DF rank10、8 system qubits、共通`T_eff/V_square`、faithful tail identity、signed factor・one-body・核scalarを維持する。
- 元Hの保存`state`を厳密基底状態と仮定し、no-prep。`DF_state`へ変更せず、state solve・新DF fit・rank/辞書探索はない。
- `delta=.05`、q=64、K=2、r=4/8、P/DFの4点のみ。PF/RTE/stat/model/実装/reserve位相配分`.06/.03/.15/.03/.01/.02`、`beta_RPE=.30`を維持する。stat枠は位相配分の整合判定にだけ使い、shot数を導出しない。
- Qiskit1.3.0と既存Gaussian/native builderを再利用する。compiler価格処理は対象外。runtimeのNumPy/SciPy/OpenFermion/Qiskit/qiskit-nature版を科学実行時に記録する。
- 入力：commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。今回取得・decodeしていない。科学実行時は入力byteのhashを確認後、既存loaderで読む。
- 新runner SHA256：`1b99dae1fc65cbc9e97390aaa6577b149a7413403d0209f307f743cf5fb3bf84`。
- 再利用C0 driver SHA256：`0b6da463853bc3b6ba881cae636e6169ef751e3311e5dcc8f92d1cb42c24f8e3`、既存validator：`b84594b4b0f63732d02e8157f83b3677e4608da3ba934fb206265dfaad29b292`、resource bridge：`d1dc2c6d813873a793dadc98d31d0a3e347ca24b4b91e2c81fc1f16ebdeb064d`。

runnerは使用する固定COREをbase `8820c41...`のbyteと照合し、新driverとCOREのSHA、実行SOURCE commit、入力provenanceを新resultへ保存する。これは科学的SOURCE/inputの再現情報で、ユーザー承認の保存・認証機構ではない。

## 実結果を見る前に固定する判定

既存validatorのbinary64/program-scale式をそのまま使う。`n` system qubits、`d=2^n`、保存factor数L、既存operand scale Sに対し、`m=4n^4+8d+16L`、operator toleranceは`m*eps/(1-m*eps)*max(1,S)`、basis toleranceは`32*eps*d*max(1,max_basis_length)`。implementation toleranceは両者の最大に`(1+q*r*(K+2))*(1+delta)`を掛ける。人工atolをH4へ転用しない。

各rの新payloadからC0と同じ式で`eta=d*max(operator,basis,implementation tolerance)`を作り、半径allowance eta、複素allowance 2eta、位相allowance `asin(2eta/(rho_PF-eta))`を用いる。非正半径・asin domain外はSTOP。**r2で得た数値allowanceをr4/8へコピーしない**。program lengthに伴う変化を記録し、観測差に合わせた緩和・clamp・再fitは行わない。

同じq/delta/r/K・辞書lambda・normalizationを既存`observed_signal_bounds`で照合し、PF/model/実装の位相差＋allowanceとfinite-RTE複素誤差diskを分離する。RTE phaseはdisk/reference radiusのasin、観測半径下界は`attenuation*(reference_radius_lower_bound-disk)`。各PF/RTE/model/実装配分、正半径、C0と同じ総配分式`.06+.03+.15+model_bound+implementation_bound <= .30`で判定する。reserveはstatへ再配分しない。許容値・配分・r集合は結果を見て変更しない。

## controlled coverageと対照の扱い

既存validatorは、Gaussian基底・one-body half、決定論的に選ぶPauli/DF event、scalarと両halfを含む代表complete controlled stepを`diag(I,U)`と照合する。evolution/eventへのphase alignmentは行わない。q64の補正finite meanは独立CAR/paired-polynomial参照と照合する。**q64全trajectoryの完全列挙や、その長いcontrolled回路全operatorの保証ではない。** 選定eventを価格MC標本・期待値推定へ流用しない。

C0 r2/K2の対照は[固定報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md)と[元signal_m6 raw](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc6432a69dee24aec8bad19968a643aaee8ac5ee/artifacts/track_a_o1_m1_c0_science_20261106/run01/signal_m6.json)を引用する。50eeのresultはcarry accountを含むが、signal rawは旧partial branchにある。runnerは旧rawを読み込んで再解析せず、引用先だけ記録する。r2再実行、旧FAIL/allowance/SOURCE/STOPの変更はない。

## 人工テストと実行承認の境界

[新テスト](../../../tests/test_o1_m1_attenuation_validation.py)と既存`tests/test_o1_m1_circuit_signals.py`で**19 passed in 4.71s**。架空2-mode入力だけでq64/r4,8の独立mean・controlled代表検査、q*rのnormalization合成、tauの変化、r依存allowanceを確認した。意図的に元HとDFが異なるtoyはmodel予算FAILとなり、validator PASSと物理予算PASSを区別できた。invented scalar/mockで各phase/radius/domain FAIL、固定4点だけの呼出、最初FAILで停止、既存出力拒否を確認した。想定外例外のown ERROR_STOP保存と、同じ出力への再進入拒否時のbyte保全も確認した。`np.load`、sampler、shot accounting、価格provider/compileを禁止するfixtureを用い、実H4値は使わなかった。

既存M1 runtimeをread-onlyで使用、Gaussian依存は既存専用path、cacheは`/tmp/o1_m1_attenuation_preparation_cache`、bytecode/pytest cacheは無効。科学的性能・実H4 correctness・減衰改善の証拠ではない。新科学結果はまだない。

**現在STOP**：指示文の作成は科学実行許可ではない。ユーザーが、公開した本SOURCE・固定保存入力・上の4点・明示したcontrolled coverageについて、新operator生成、元state信号/normalization/attenuation/半径/phase評価と必要な実回路buildを明示許可した場合だけ、**Codex**が別science worktree/runで実行する。追加の承認証明書・grant発行は不要。

承認後もshot導出、価格MC、価格compile、量子測定sampling、全round総費用、格子拡張・候補再ランキングは対象外。限定検証が通過しても総費用改善・0.1 Ha/95%達成保証や正式採択へ自動進行しない。不適格でも条件変更・再実行へ進まない。科学runは自身のraw/resultを新namespaceへ保存し、終了後mandatory STOPする。

次の担当：**ユーザー**が上記限定科学実行を許可するか判断する。**Codex**は許可後のみ別science worktreeで対象4点を実行する。本準備agentは成果公開・remote照合後STOPする。
