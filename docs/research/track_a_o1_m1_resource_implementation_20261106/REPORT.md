# O1 M1：matched PR＋RPE resource接続の最小実装候補

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。起点はM1計画`ec699ab1edb975dca8a47183388c4604becc1733`、既存科学検証は`424ea9d1f345f15ec863ee9102a997b0b3fbd858`。**最小実装と人工unit testまで完了。実H4の新しい科学結果・shot数・総費用はない。公開後mandatory STOP。**

## 実装と再利用

- `src/trotterlib/o1_m1_resource_bridge.py`：既存M1 splitを受け取り、共通Gaussian `T_eff`の半step、辞書別のfaithful `V_square` event、核scalar一度を接続する。P側をgeneric Pauli prefixへ渡さず、両辞書で同じ物理分割を保持する。元`state`を保存したまま、入力・係数・stateに依存する共通contextと辞書hashを作る。sector投影・新selector・再DFはない。
- 既存`o1_m1_matched_split`、`o1_m1_circuit_signal_validation.one_body_circuit`、`df_rte_tail`、`QiskitDFRTEEventCircuitBuilder`、`append_pauli_event`、finite-RTE lawを流用する。`build_controlled_trajectory`は明示event列のみを受け、各`q*r`箇所を個別に指定する。RNGやsamplingは実装しない。
- `rpe_hadamard_interrogation.wrap_controlled_no_prep`へ既存wrapperのH／evolution／Sdg／H／測定を切り出し、既存DF builderも同じ関数へ接続した。既存DF resultのmetadata・bit mapping・fingerprint・controlled/phase規約を変更しない。新関数はcallerが資格確認したancilla-last `diag(I,U)`回路を受ける。
- `MatchedNoPrepCostProvider`は上記controlled回路を両軸の測定込みwrapperへ渡し、既存`transpile_and_measure_cost`とweighted metric accumulatorを使う。完全な有限trajectory集合の正しい重み付き期待値に限定する。欠損、重複、lawと異なるeventをbuild前に拒否する。**旧204代表eventを等重み価格平均・MC標本として利用できない。** 各軸でwrapper全体をcompileし、state preparationは含めない。law（tail/r/K/δ）、実backend、compiler、物理context、arm、qと価格scopeを区別する。canonical化できないbackendは受理しない。
- `build_strong_df_baseline`は既存`c1_h4_strong_baseline.prepare_xdf_basis`／`StrongBuilder(relative_fusion=True)`／native S2 scheduleを直接流用する。保存された全signed factorと`T_eff`、scalarを保持し、明示したsubstep数とbasis policyで強決定論DF-PFの回路を作り、同じproviderのdeterministic入口へ渡せる。basis間のanalytic vacuum phaseと同基底融合は既存実装を使う。spin policyは厳密なspin構造がある場合のみ選択できる。旧run契約・input loader・旧価格は使わない。baselineのarm identityとoperator資格はcallerが固定する。
- `account_radius_round`は既存`RPERoundSpecification`、`RPEErrorAllocation`、`finite_rte_attenuation`、実半径`_hoeffding_shots`、typed compiled costを再利用する。正当化された未減衰参照半径下界、finite-RTE補正mean誤差disk、PF/model/実装位相誤差を明示入力する。単位半径defaultはない。`A=B_K^(-q*r)`を一度適用し、`rho_obs,lb=A*(rho_ref,lb-epsilon_Z)`と両軸shot算術へ接続する。非正半径・不十分な誤差配分を不可とし、既存位相モデルの`0<beta_RPE,beta_stat<=pi/2`を維持する。
- 価格がある場合に限り各軸のshot数×full no-prep期待価格を接続する。classical MC数をshotへ掛けない。PR価格をdeterministic扱いした場合やr/K違い、backend・context・scope違いを拒否する。`aggregate_registered_rounds`は指定した全roundが揃い、予算内で同じcontext・arm・compiler・backend・scopeの場合だけ集約する。未知価格や未検証roundを0扱いしない。

全ての算術結果を`guarantee_status="not_certified"`とし、既存unit-radius/effective-PF-eigenstateの保証を拡張しない。finite lawの設定用許容値は選択したKの既存step residualに基づくadmission値であり、科学的RTE誤差予算や実H4判定許容値ではない。

## 許可済み人工テスト

**47 passed in 41.68s**。新規`tests/test_o1_m1_resource_bridge.py`の11件と、既存`test_o1_m1_matched_split.py`、`test_o1_m1_circuit_signals.py`、`test_rpe_hadamard_interrogation.py`の36件。`git diff --check`も通過した。

新規テストはinvented 2-modeの複素factor・正負weight・identity・核scalarで、P/DFの明示trajectoryとcontrolled相対位相、両軸信号、独立CARによる強DF-PF operatorを確認した。完全lawのweighted mock価格と両軸full wrapper、別のinvented 2-qubit controlled gateの既存本物transpile primitiveも確認した。実半径、`q*r`指数の二重計上防止、MC数非乗算、欠損／重複event、law/backend/価格scope不一致、未知価格・欠損round・失敗予算を検査した。既存DF wrapperの人工回帰も通過した。

実行には既存専用環境をread-onlyで使用した：`track-a-o1-m1-minimal-implementation-20261106/.venv/bin/python`と`track-a-o1-m1-circuit-signal-preparation-20261106/.venv/gaussian_deps`。Python 3.12.3、NumPy 1.26.4、SciPy 1.14.1、OpenFermion 1.6.1、Qiskit 1.3.0、qiskit-nature 0.7.2。cacheはこの作業専用`/tmp/o1_m1_resource_impl_*`、bytecodeとpytest cacheは無効。pytestはpassive package aliasでeager研究importを避け、上記4 testファイルだけを選択した。新規testと既存M1 testは保存NPZ読込を禁止し、新規testはRTE sampling APIも禁止した。

**人工テストPASSは科学性能・全RPE保証の証拠ではない。** 保存H4数値をdecodeせず、旧4条件の再解析、実H4 operator／signal／回路生成／compile、実条件shot／費用計算をしていない。旧SOURCE・raw・結果・grant・marker・STOPを変更していない。

## 未解決事項・必要な固定条件

このproviderは小規模な完全exact期待値の接続候補である。trajectory数は`(sum_k n_components^(k+1))^(q*r)`となり、実H4・長roundを効率的に全評価できる完成pipelineではない。大きいdomainは既存MC primitiveの薄い接続、またはM1の同一入力・compilerに対するproxy較正／holdoutを別途用意し、価格不確かさを測定失敗確率と分ける必要がある。旧proxy domainを流用しない。実H4のnative回路・operator同値性は未検証である。

ユーザーが固定する必要があるのは、energy精度`epsilon_E`、総失敗確率`alpha_tot`、alias-free energy区間・branch、資源metric、compiler/runtime/backend/gate set/routing、PF/RTE/model/実装・統計誤差配分、両軸／round失敗予算、δ/r/K/substepsの比較範囲、必要horizon、直接／MC／proxyの評価domainである。元Hの保存`state`はDF/PFの固有状態とは限らず、各roundの半径・PF phase・元H/DFの差と目標energyへの対応を別途資格確認する。`aggregate_registered_rounds`は登録roundの算術集計であり、energy精度やbranchを自動認定しない。Pauli/DF/強DF-PFを同じ物理context・達成精度・成功確率・compiler・no-prep境界で比較する条件は未固定である。

主baselineは強決定論DF-PF。UWCは既存simple shift/BLISS等の実装範囲とOOの`NotImplementedError`を区別する。旧grouped UWCのPF+QPE/layer/STAR価格はRPEのfull no-prep価格と互換ではない。今回UWC変換・価格接続は追加していない。旧F2 prep込み価格、unit-radius仮定、短時間4条件を全RPEへ外挿していない。

## 次の承認範囲とSTOP

固定H4入力は計画どおりcommit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`の`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。H4-A、保存DF rank10、元`state`、共通`T_eff/V_square`、faithful tail identity、全signed factorとscalarを維持する。今回このbinaryは読み込んでいない。

次は**ユーザー**が上記比較条件と実行domainを固定し、保存H4 decode／新roundのoperator・controlled phase・PF/RTE/元state信号・半径／誤差検証、強baseline build、両軸full no-prep compile、horizon・shot・期待価格・全round集計のうち実施する範囲を明示承認する。価格MCやproxyを使う場合はその条件と追加接続も範囲へ含める。**Codex**は承認範囲内だけを実装・科学検証する。半径・branch・誤差・価格domainと全必要roundが揃うまでは総費用改善を主張しない。

この実装候補・人工test・本REPORTを専用branchへcommit/pushし、remote SHA・公開内容とlocal treeの一致確認後、**mandatory STOP**。今回の許可から科学計算、詳細科学レビュー、RPE総費用評価へ自動進行しない。
