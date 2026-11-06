# Track A B2：総RPE回路・費用接続の実装

2026-11-06 JST / Codex主エージェントによる実装・技術確認。
起点 `2e17f4d39ec3733c5e1f58eda6871d69b06a651c`。
専用branch `codex/track-a-b2-rpe-connection-20261106`。

**最小adapter・人工検証のみ。実H4入力の読込、科学計算、回路build/compile、価格MC、shot数・総費用算出、保存結果の科学的再解析、詳細科学レビューは実行していない。** Agent Aのworktree・SOURCE・実行状態を変更していない。

## 接続したもの

| 接続 | 実装・検査範囲 |
|---|---|
| B2 → round | `B2RPERound`がgeneration-prefix `L_D`、`r/K/delta/q`、既存`RPERoundSpecification`を明示照合。forward half sweep → canonical finite-RTE → reverse half sweepを既存request/native builderで保持 |
| 信号 → M1 | `B2SignalRecord`が元H、DF、ideal-tail PF、finite corrected、rawの5複素信号を別fieldで受ける。すべて同じ元H state identityとround/law/SOURCE/compilerに結合 |
| 半径・shot | `account_signal_round`から既存`account_radius_round`を呼ぶ。PF半径下限とcorrected誤差diskを渡し、raw半径を再減衰させない。`A=b_K^(-qr)`と`raw=A*corrected`を照合。実装位相誤差と数値allowanceも明示入力 |
| X/Y価格 | 既存`MatchedNoPrepCostProvider`にbuilder注入経路を追加。同一drawのevent列を両軸で共有し、draw間では既存canonical samplerのfresh streamを使用。Gaussian、scalar/identity相対位相、測定wrapperを含むwhole circuitを既存compilerへ渡す |
| 全round | `aggregate_b2_rounds`は既存`aggregate_registered_rounds`を使用。欠測・重複・未取得価格・混合contextを拒否し、軸別`N × E[C]`を合計。古典価格標本数を量子shot数に掛けない |
| 強PF | 同じ全DF factorsから既存`prepare_xdf_basis`/`StrongBuilder`を使用。Full S2と標準三段Yoshida4。substeps・basis policy・relative fusionを明示。PRと同じcontext/compiler/測定付きfull no-prep価格・信号会計を使用 |

`q=2**m`、`t=q*delta`は既存round型の定義。round数・energy精度からのschedule選定や誤差配分の最適化は実装していない。既存集計と同じく、一つの集計のbase deltaは共通。異なるr/Kはround別lawに、固定prefixはarm identityに保持する。

## 再利用と最小差分

起点branch内の以下の実装は改変せず使用した。

- `pr2_matched_accuracy_m1_execution._prepare`：旧B2と同じgeneration-prefix、`extract_identity_phase`、coefficient threshold 0のpreparation。
- `make_df_partial_s2_repeated_request` / `DFPartialS2RepeatedRequest`：既存canonical event samplingとstep検証。`QiskitDFPartialS2RepeatedCircuitBuilder`は人工operator対照。
- AX `partial_native_from_step_requests` → `build_partial_native` と `build_native_hadamard_wrapper`：ordinary-controlled evolution、順序・Gaussian・核scalar・tail identity・event相対位相、X/Y wrapper。
- `StrongBuilder`、`prepare_xdf_basis`、既存S2 scheduleと`product_formula._get_w_list('4th')`：全signed factors、same-basis合算、relative Gaussian fusionのanalytic vacuum phase。

起点に存在しなかったM1資産は固定SOURCE
`8a13539e51bb113e82cb86ddb608e2e6e27786ca`から同じ公開pathへ取り込んだ。

- `o1_m1_resource_bridge.py`：会計・集計・価格統計・whole-wrapper compile。変更は`_compile_pair`へのwrapper注入と`monte_carlo_with_builder`への既存MC本体の共通化のみ。既存M1入口のmethod名・fingerprint生成規則は維持。
- `o1_m1_matched_split.py` / `o1_m1_circuit_signal_validation.py`：上記の既存import依存として原内容で取り込んだ。**B2ではM1専用の`T_eff/V_square` preparation・信号評価を呼ばない。**
- `rpe_hadamard_interrogation.py`の既存`wrap_controlled_no_prep`と`pauli_signal_adapter.py`の既存identity位相修正を同版から取り込み、M1依存を整合。
- `test_o1_m1_{matched_split,resource_bridge,price_mc,circuit_signals}.py`と、人工テストが利用する`scripts/run_o1_m1_h4_operator_correctness.py`の参照helperを原内容で取り込んだ。後者の実H4 runner/mainは実行していない。

新規`src/trotterlib/b2_rpe_connection.py`は、上記の間に欠けていた入力契約と呼出接続だけを担当する。M1専用preparationをB2と偽装せず、PF、event sampler、Gaussian emitter、shot式、費用式を再実装しない。新規テストは`tests/test_b2_rpe_connection.py`。

## 人工検証

**計219件（新規47件＋再利用172件）すべてPASS。**

| 人工テスト | 件数 |
|---|---:|
| 新規B2 RPE接続 | 47 |
| 固定版M1 algebra/circuit/resource/price | 34 |
| 既存repeated partial-S2 | 22 |
| 既存B2 signal実装 | 9 |
| 既存AX native | 53 |
| 既存strong baseline | 25 |
| 既存RPE resource/wrapper | 29 |

最終実装で一括215件PASS（72.07秒）。canonical order-2の負位相4ケースを追加後、変更した新規テスト47件を再実行し全PASS（25.98秒）。既存172件のSOURCE・テストはその後変更なし。`git diff --check`もPASS。原内容で取り込んだ9 pathは固定M1 SOURCEとのbyte一致を確認した。初期のテスト参照API誤りと専用環境の不足依存は修正済み。

新規テストは、人工二modeの複素・非可換one-bodyとsigned DF factors、架空round、mock価格を使用する。人工4 fragmentで`L_D=3/r=4/K=2`も検査した。実H4のデータではない。人工B2の小回路のみ、実Qiskit compilerへのwhole-wrapper接続も検査した。

重点検査：非ゼロprefixのforward/reverse順序、独立CAR Hamiltonian再構成、scalar/identity一回計上、control-0 identityとcontrol-1 event積、X/Y信号、raw/corrected半径・normalization、全5信号のDF state置換拒否、round/law/SOURCE/compiler/scope不一致拒否、paired価格・軸別shot、全round欠測拒否、既存総和式、同条件強PF。強PFのspin routeは厳密なinterleaved alpha/beta liftだけを許し、off-spin成分の投影・閾値削除はしない。既存spin/fusion人工テストも再利用。

テスト環境は既存専用venvを読取利用（Qiskit 1.3.0）。不足するQiskit Nature 0.7.2、Qiskit Algorithms 0.3.1、threadpoolctl 3.6.0は今回worktreeのignored `.venv/extra`だけに配置した。共有venv・Git認証・共有設定を変更していない。

再現する人工テストの入口（依存を導入した専用Pythonで実行）：

```bash
PYTHONPATH=.venv/extra:src python -m pytest -q \
  tests/test_b2_rpe_connection.py \
  tests/test_o1_m1_matched_split.py tests/test_o1_m1_resource_bridge.py \
  tests/test_o1_m1_price_mc.py tests/test_o1_m1_circuit_signals.py \
  tests/test_df_partial_s2_repeated.py tests/test_pr2_matched_accuracy_m1_execution.py \
  tests/tracks/resource_applicability/test_ax2a_native_df.py \
  tests/management/test_research_c_c1_h4_strong_baseline.py \
  tests/test_rpe_resource_accounting.py tests/test_rpe_hadamard_interrogation.py
```

## 公開経路

ローカルGitの著者情報が未設定でcommitが成立せず、通常pushもHTTPS認証不足で失敗したため、ユーザー指定の接続済みGitHub連携で専用branchのcommitを公開する。共有Git設定・認証を変更しない。remote SHA・tree・全17変更pathの内容を検証済みローカル内容と照合し、最終チャットで公開SHAを報告する。

## Agent Aの結果を接続する入力と未評価事項

呼出側は`RPEConnectionContext`へ、元H identity、保存DF Hamiltonian hash、元H state identity、sector、spin orbital順序、今回SOURCE、compiler/backend、費用指標・full no-prep scopeを渡す。stateは元H基底状態を仮定する契約であり、旧DF基底状態へfallbackしない。identity宣言は実stateの来歴・一致を科学的に証明するものではなく、Agent Aの対応確認が別途必要。

model誤差、energy精度、成功確率、alias/branch条件は未指定なら`None`で保持できるが、その状態でshot会計は拒否する。各roundのschedule、PF/RTE/統計誤差・軸別失敗配分、5信号、normalization、実装位相誤差、数値allowanceも明示入力。元H/stateに対応する信号の評価driverは今回作っておらず、旧B2信号を移植しない。

`model_error_bound`・energy/alias条件は共通条件の識別に保持する。energy保証への変換・branch復元の認証は行わない。観測差＋allowanceは経験的入力であり厳密誤差上界ではない。出力は常に`not_certified`。価格の標準誤差は標本診断だけで、総費用CI・X/Y共分散の保証は追加していない。人工PASSは実H4 correctness・RPE成功・PF優位性を示さない。

## 次の限定科学検証とSTOP

- **ユーザー**：Agent Aの元H/DF12/state/order対応を確認し、energy精度・成功確率・alias/branch・全round scheduleと誤差配分を採択する。
- **Codex（別途明示承認後）**：固定SOURCE・入力・条件で、指定したB2 `L_D3/r4/K2`と同条件強PFの限定信号・半径・normalization・controlled correctness検証を行う。実H4のbuild/compile、価格MC、shot/総RPE費用まで含める場合は、その範囲・round・価格cohortも別途明示する。新候補探索・最適化・詳細科学レビューは含めない。

**本作業は実装・人工検証・専用branch公開後にmandatory STOP。Agent Aとの統合、実H4適用、追加最適化・価格測定へ自動進行しない。** 既存SOURCE・input・raw/result・grant・marker・STOPは過去版を含め保全。
