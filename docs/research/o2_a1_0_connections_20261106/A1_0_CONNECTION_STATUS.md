# A1-0 接続状況：SOURCEの特定と人工入力による接続

2026-11-06 JST。基点は前回公開 `a2fcbe212ebd66b563dd1413768e5569bf5fdfed`。
前回の環境・入力hash・5ファイル取得照合は[HANDOFF](../o2_a1_server1_handoff_20261106/A1_SERVER1_HANDOFF.md)を継承する。
今回の対象は保存linear H4、1.00 Å、STO-3G、legacy DF12、4電子/8qubit、L_D3、m0/q1/δ=t=.02/r1/K2、X/Y、最終opt2だけ。
**opt2 finite targetのSOURCE上の定義と、CDF/selected-only価格の接続コードを特定・用意した。実科学入力への適用と数値的な接続の完了は別であり、未実施である。**

## 1. 今回進んだ点

| 論点 | 今回の到達点 | 残る科学処理 |
|---|---|---|
| threshold差 | cutoff→identity抽出→tail正規化の順を確定。random tailだけの差ではない | 追加finite-target/phase/normalization誤差の数値上界 |
| opt2 target | threshold0、同じpreparation、K2 paired Taylor/attenuation/selected policyとして一意に指定 | 保存NPZから対応する完全payloadを取得し来歴をbind |
| catalog/CDF | 保存WP06bのsupport libraryは完全な確率catalogではない。再構成経路と不足を限定 | 完全catalog、実CDF、53-bit support/ν/momentsの取得 |
| 接続コード | 新規 `a1_connections.py`：sampler入力→実CDF→既存law adapter、固定bank、selected-only full価格 | 実入力は一度も渡していない。価格のreal依存も未呼出し |
| 層別 | 既存bank/mass/meanを再利用。既存cost用条件付き生成はそのまま使えない | canonical実qに従う測定前受理と有料取得の小接続 |
| full価格 | selected evolution一つ、X/Y wrapper各一つ、既存compiler/cacheへの経路を用意 | 実pathの不足build/compile、期待価格のconfidence |

## 2. opt2 finite targetをSOURCEから固定する

原科学cohortの閲覧用carrierは `4591b8d75a2a6a55c74a51225f0c61648023c8ee`。元dirty executionの来歴と、このcarrier/本作業commitを区別する。
SOURCEは本基点の以下を指す（既存ファイルは変更していない）。

- `src/trotterlib/research_direction_full_opt2.py::_worker_preparation`：固定NPZをload、LD3 split、`prepare_df_partial_s2(..., identity_policy="extract_identity_phase")`。coefficient_atolを省略するので既定0.0。
- `df_rte_tail.py::extract_df_diagonal_tail`：componentをfragment/supportの固定順に並べ、`coefficient_abs > coefficient_atol`をretainし、その**後**にidentityを抽出する。
- `df_rte_tail.py::extraction_to_symbolic_rte_tail`：残る非identity係数のL1と`coefficient_abs/lambda_r`、sign、basis metadataを保持する。
- `research_direction_full_scope.py::_finite_inputs`：固定t=.02/r1/K2のconfig/distribution。`rte.py::finite_taylor_operator`はdegree K+1まで、`finite_rte_operator_moments`はcorrectedとattenuatedを分離する。
- `apply_run_threshold_policy` / repeated boundary builder / Hadamard wrapper：元selected回路とphase/readout。Xはreal、YはSdg→Hのimaginary、bit0→+1/bit1→−1。

この定義のalgebraic finite targetを `μ_fin,b^0`、実binary64 CDF/53-bit格子で同じ回路を選ぶ平均を `μ_q,b^0` と書く。canonical weightは元のw=1。
μ_finは元のattenuated有限event-meanで、corrected signal・元H exact signal・実q平均へ暗黙に置換しない。
形式的にはm0の中央演算は `P_3(−it H_R^0)/B_2(t λ_R^0)`、`P_3(A)=I+A+A²/2+A³/6`。
その両側に既存D halvesとscalar/identity phaseをSOURCEと同じ順で置き、保存DF全体のphysical-sector groundで期待値を取る。
これは演算定義の指定であり、今回operator/state/P3/Bの数値を生成したものではない。SOURCEのbinary64丸めと理想algebraic演算の差は別途numerical項に残す。

### WP04からそのまま移せるもの／移せないもの

前回記録どおり、WP04 signalはcutoff1e-12/preparation `89145d…`、opt2は0.0/`50c77f…`。
保存dropped L1=`4.8294649659693557e-14`、dropped count=108は元の数値記録であり、新しいopt2誤差ではない。
同じ抽出前係数・basis・順序であることが結ばれる条件の下で、元の`threshold_operator_error_bound`はthresholdによるHamiltonian差を扱う材料になる。
しかし、それだけではP3、B、抽出identity phase、各labelの実qが一致することは示せない。全droppedが非identityとの仮定も追加しない。

| 保存材料 | 再利用の条件と限界 |
|---|---|
| H/state/sector identity、D部分の定義、符号規約 | 同じsnapshot/H/partition/physical stateとSOURCEをbindして参照できる。完全state vectorの保存証明ではない |
| WP04の218-label、λ、normalizer、信号/bias/半径 | threshold0 lawの入力・真値・上界として直接移送できない |
| PF/Taylorの既存評価関数 | threshold0入力で再評価するためのコードを再利用できる。旧数値保証が自動継承されるわけではない |
| 既存dropped L1上界 | SOURCEの差分項を限定する材料。有限targetへ変換した数値上界は今回未取得 |

新しい差の数値上界は、同じm0の正規化された有限operatorとscalar phaseを比較するか、既知のtriangle/telescoping評価で取得する。
新しいPF係数fit、全grid、全path列挙、他roundを要求しない。独立した重要科学レビューや新定理を本作業で開始していない。

## 3. 保存catalog/CDFの調査で分かった限界

前回既知のNPZ/WP04/opt2/価格rawを再調査一式として複製せず、canonical research_direction/RTE保存JSONと元WP11 opt2のworker/checkpoint JSONを対象に不足payloadを探した。
完全な係数付き順序catalogまたはorder/label CDFは見つからなかった。この範囲外にも存在しないとの断定はしない。

新たに区別した保存資産は、`artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json` の `/support_basis_library`。
`definition_count=324`という**保存値**とbasis/support/operation-count/hash/proof metadataがあるが、各componentの係数・probability・元抽出順はない。
これはcanonical WP06bの基底定義数であり、実53-bit positive-support component数でも、C1の別packetを流用した値でもない。
WP06aの2 representative IDも完全catalogではない。旧opt2 rawのseed/fingerprint/価格からlabel列を新生成すれば、禁止されたseed replay/RNG実行になるため行わない。

完全payloadの取得先は、同じNPZをload→LD3 split→threshold0 preparation→`rte_preparation.symbolic_tail.components`。
phaseに必要な`tail_extraction.deterministic_identity_coefficient`、λ、tail/preparation hashも同じpreparationから保持する。
この再構成は新科学処理として後日の明示許可を要する。別のcatalogを代用しない。

## 4. 実CDFからνまでの薄い接続

[新module](../../../src/trottertracks/algorithm_codesign/a1_connections.py) の `connect_sampler_inputs` は、準備済みcomponents/distributionと**既存 `rte._validate_components` 関数**を引数で受ける。
samplerと同じfsum検査→各probabilityをその和で除算→binary64配列化を保ち、NumPy1.26.4 choiceのcumsum→最後の要素で除算を行う。
orderの入力配列も同じ手順でCDFへ結ぶ。sampler、生成順、normalization、RNG自体は変更・再実装しない。
既存 `law_from_choice_cdf` にnormalized CDFを渡し、`OccurrenceLaw`へ接続する。choice入力配列/CDF/和を保持して、後日のideal-vs-actual比較に使える。

```python
# 実科学payloadへの呼出しは許可後だけ。SOURCE/preparation/NumPyをsourceへbindする。
connection = connect_sampler_inputs(
    preparation.rte_preparation.symbolic_tail.components, distribution,
    source=bound_source, validate_components=rte._validate_components)
cv_bank, order_bank, five_strata_bank = proposed_control_banks(
    connection, law_id=bound_law, a="df-fragment-3:Z0Z1", b="df-fragment-6:Z0")
```

`proposed_control_banks` は前回の3control、order-only、order/label共通細分を既存 `ControlBank` で表す。
ν/joint/stratum質量は同bankの既存接口を利用し、今回の実科学値は評価していない。
positive binary64入力でも、途中CDF境界の丸めにより実格子質量0になり得る。人工例でこの拒否を確認した。a/bが実qで非positiveなら別IDへ差し替えずSTOP。
実CDFのbinary64値はfloat.hex等の可逆表現とSOURCE/NumPy versionへ結ぶ。53-bit格子の境界質量・全質量1・zero/positive support・同一positionのjoint条件を許可後に検査する。
PRNG/fresh-IID random-wordモデルとdata独立性は別条件。Parent dyadic proposalは使わない。

## 5. control・強いbaseline・有限confidence

controlと5層の定義、a/bの選択規則、train1024/axis、B1 pilot64/層/axis、η=1/8、cal数式、production8192/方式/axis、range guard、失敗配分は[前回草案 §§3–7](../o2_a1_server1_handoff_20261106/A1_SCIENCE_EXECUTION_CONTRACT_DRAFT.md)を維持する。全て未承認の提案。
B0にも同じvariance-sensitive保証、B1-O/B1-OLにも有料配分学習を認める。B1-OLに吸収される定数補正をCV固有利益へ二重計上しない。

既存 `rte_order_stratified_cost_validation._compile_length_strata` とconnected-cluster処理にはorder固定/label選択・same/different層取得がある。
しかし、条件付きcost検証専用の生成・compile・旧数値lawを含み、canonical実qのfresh Hadamard測定取得adapterではない。runnerを起動して代用しない。
初期案はcanonical pathを古典生成し、測定前に固定stratumへ受理する方式。既存 `EventMetadata`/bank.values/stratified_meanを再利用し、取得・棄却費・全正層の有限quotaを外側の小接続で扱う。

前回のattempt cap `ceil(2M_h/p_h)` はproduction数がcalibration依存である点に注意が要る。
**各stage開始前に、そのstageの受理quotaでM_hを固定する**読み方に具体化する。train/cal quotaは前もって、prod quotaは独立calibration後・prod前に凍結する。
一つの全campaign capが必要なら8192の上限を含む最大quotaから事前固定する。数式・提案予算・候補集合は変更しない。
capで打ち切った不完全層を削除・再正規化したり、成功runだけを選んだりしない。

保存coordinate ε/αをmodel内の参照幅/配分として使う案と、厳密RPE保証は区別する。旧510・V2 binary tailはCVへ流用しない。
range D、cal variance上界、centering控除、正の残余幅、十分N/各層配分を許可後に計算し、cap超過ならSTOP。今回これらの新数値はない。

## 6. bias・state・半径を閉じる必要範囲

| 接続 | 初期finite-mean診断に必要な条件 | 初期scopeで不要な拡大 |
|---|---|---|
| threshold | opt2 targetを固定し、WP04値を利用する場合だけthreshold差の正当な伝播を追加 | 218-labelへのlaw変更、全PF grid再計算 |
| finite-grid law | `b_law ≥ |μ_q^0−μ_fin^0|`。CDF丸めとalgebraic weight/rotationの数値差を分離して上界化 | 全path期待値のoracle利用、新proposal探索 |
| centering | 実qの正確なνまたはe_j上界、`Σ|λ_j|e_j`控除 | 未知共分散/最適λの無料利用 |
| state | 同じ保存DFの4電子sector groundをmeasurement engine内で利用。保存fingerprintだけではvectorを復元できない | H_D/PF controlへの置換、gap/E0を推定器へ供給 |
| native/numerical | 初期候補はideal gates/no backend noise/no native synthesis。実装丸め・policy等価性の許容を明示する | TをRZへ換算、native価格UNKNOWNを0扱い |
| radius/branch | μ_finの座標推定自体には半径下界不要。RPEへ移送するなら正当なρ_lbとphase/alias/branch条件が別に必要 | near-one→ρ_lb=1、1round→18round成功/energy精度の認定 |

`b_law`の評価には、正規化したreference categorical lawと実qのL1距離・product-lawの標準的な伝播を使える。binary64入力列は和が数学的に厳密1とは限らず、そのまま確率lawと呼ばない。
reference正規化の違い、angle評価、selected policy等価性、phase実装の差はnumerical項を含めて扱う。
threshold/b_law/centering/native/numericalを混ぜて一つの無料0にしない。保存εへ既に配分済みのPF/RTEを無換算で二重控除もしない。

既存state生成は `finite_rte_signal_validation` のfull DF sector行列→eighの最低vector。保存JSONには定義とfingerprintがあり、vector配列はない。
後日のsimulationに必要なら同じsectorでground一つを再構成する範囲を明示許可する。旧validator丸ごとはPF固有state・複数grid・shot評価も行うので起動しない。
元物理Hとの対応未閉鎖、V1/V2の条件付きstatusを維持する。新しいbias・radius・残余幅は今回未評価。

## 7. selected full価格：実装済み接続と未取得価格

新 `selected_full_prices` は既存 `apply_run_threshold_policy`→repeated builder(boundary_optimized)→指定軸のfull measured wrapper→既存compile/cacheを呼ぶ薄い接続。
m0/LD3/threshold0/r1/K2/controlledと元compiler profileを検査し、**full-basis比較回路をbuild/compileしない**。mockテストでselected evolution一つ、wrapper/価格二つを確認した。
独立取得した片軸trialには `axes=(axis,)` を渡し、もう一方のreadoutを余分に生成・compileしない。両軸指定も保持する。
実科学requestは未入力、real builder/compile依存は本関数から未呼出し。

既存 `TranspiledCircuitCostCache.get_or_transpile` はactual circuit/数値parameter/phase、compiler、backendに結ぶcacheを持つ。
これはread-only lookupではなく、missでcompileし、persistent指定時はDBを作成・更新する。将来の許可scope内で自分専用cacheを使い、共有cacheへ接続しない。
元opt2の `_compile_paired_requests` はこのcacheを設定せず直接compileする。調査したWP11 artifactsには再利用可能と確認できた対応persistent DB/QPYはなかった。
保存8 trajectory rawには同path/同profileの価格記録があるが、fresh path全体のcache/全stratum平均ではない。seed/fingerprintだけで任意pathへ価格を移送しない。

cache hitでも各量子shotのfull費用を `TrialCost` に計上する。B1は各条件付きpathの価格を実取得/同一cacheから取得し、原law平均を一律に代用しない。
初期報告を実現campaign費用に限定する案を提示する。期待費用のconfidenceを要求する場合、正当な価格range・独立取得/有効な同時保証が追加で必要で、旧mean/SEや5% gateは代用にならない。
T/nativeはUNKNOWN。1Qは既存posttranspile counts等から正しいscopeで取得できる場合だけ別記し、未取得のままならUNKNOWN。

## 8. 人工検証と許可境界

新9件＋既存15件の計24件を人工確率・手書きevent・mock価格だけで検証する。最終結果はREPORTに記録する。
初回testではvalidatorの直接importがpackage全体を初期化し、Qiskit内部warningを発生させたため修正した。
最終版はvalidatorを引数に取り、testでは既存の純粋関数一つだけをAST抽出して人工入力へ適用する。科学module/Qiskitが未importであることも検査した。
初回library初期化内部の全call数は記録しておらず、「全library内部の回路処理0」とは認定しない。A1/H4科学入力の投入・再構成・測定・full回路生成/価格評価は実施していない。

技術的接続は進んだが、実payload・bias上界・状態/価格が未取得なので科学測定開始の条件は未閉鎖。
必要な次の数値処理は [SCIENCE_RUN_SCOPE](A1_0_SCIENCE_RUN_SCOPE.md) のQ1–Q5に限定した。新科学実行許可なしに進まない。
