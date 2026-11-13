# signal → 全RPE精度・N → G：最小入出力

2026-11-13 JST。[設計](DESIGN.md)／[REPORT](REPORT.md)／[有限科学案](SCIENTIFIC_PLAN.md)。**仕様は既存APIを使用する。未採択のL1経験的claim・全体RMSE・十分Nを正式採用しない。** 今回の入力実行例とテストは人工データのみ。

## 1. 他担当 → B：candidateとsignal

入力の単位は既存V2 `configuration`＋全round `means`＋`target`。candidate生成/λ・finite-K signal/normalizationは供給側が担当する。Bは新H/state生成、回路作用、μやBの再計算をしない。

| 必須又は保持するfield | 内容 |
|---|---|
| configuration | 既存`PR_RPE_RMSE_V2`正本。H/state/snapshot/decomposition/sectorの出典、formula・実幅、h/M、共通κ/全round R/event/signed allocation、profile/control/full wrapper/P0、rmse_policyを保持 |
| mean.m | 0..M、duplicate/foreign不可。欠測roundは配列から落として「完了」としない |
| mean.identity/law_binding | config.rounds[m]の全identity/lawに一致。元physical ID/SOURCEとのbridgeを保持。旧データのIDだけを付け替えない |
| mean.source | 保存数値carrierのcommit/path/selector等。implementation_source/precisionと必要なら別記。ユーザー許可を証明するpacketは不要 |
| mean.raw | `{real: μ_X, imag: μ_Y}`、有限binary64・[−1,1]。取得不能/underflow/未解決を区別し、nullを0へ変換しない |
| mean.corrected | 供給側のnormalization補正mean、診断metadataとして保持。sampling確率には使わない |
| mean.normalization | `{kind, log_B, ...供給された数値精度/closure/source}`。値を再評価せず供給元の状態を保持 |
| mean.bias | PF/exact-tail/finite-K/reference等の保存誤差情報とSOURCE。signal空間の差をenergy L2に自動変換しない |
| mean.uncertainty/qualification | raw適用誤差・state/target・law/implementation・数値precisionの区分、null/empirical/enclosure等の資格。単なるdictの存在で科学資格は付かない |
| mean.targetとtarget | 同じtarget descriptor。元H_DF binding、quantity=`ORIGINAL_DF_H_TARGET_ENERGY`、source、energy_shift_ha=0、proxy/真target/誤差の意味 |

meanは既存形式のまま：

```python
mean = dict(
    m=m, identity=canonical_round['identity'], law_binding=canonical_round['law_binding'],
    source=numerical_source,
    raw=dict(real=raw_X, imag=raw_Y),
    corrected=dict(real=corrected_X, imag=corrected_Y),
    normalization=dict(kind='FINITE_LOG_B_SUPPLIED_BY_PRODUCER', log_B=log_B),
    target=target, qualification='UNQUALIFIED',
    bias=supplied_bias_or_nulls, uncertainty=supplied_uncertainty_or_nulls,
)
```

固定A schedule由来なら既存`mean_from_a`がphysical request/structure/profileを照合してcopyする。他candidateのproducerは元physical/V2 bridgeを供給するか、このcanonical mean recordへ正当なsource付き対応を示す。`mean_from_a`が任意の新分割/新formulaに使える万能物理adapterとは扱わない。

現在の`prepare`はrawだけでなくcorrected/finite logBの診断metadata欠測もblockする。足りない場合は供給側へ返し、BがB=1等を作って通さない。metadata照合はraw値の誤差認証を意味しない。

## 2. Bが受けるshot menu・統計契約

```python
allocation = dict(
    allocation_id='producer-defined-whole-vector-id',
    source=allocation_definition_source,
    parameterization=allocation_policy_descriptor,
    shots=[dict(m=m, N_X=N_mX_exact_int, N_Y=N_mY_exact_int) for m in range(M+1)],
)
prepared = S.prepare(configuration, means, allocation['shots'], target=target,
    shot_policy=dict(meaning='USER_SUPPLIED', source=allocation['source']))
```

`allocation_id`/parameterizationはこの配分の通常metadataとしてpreparedの外で保持する。APIは正整数X/Yを許すが、採択policyのX=Y・共通scale・group等の制約は供給されたmenu生成元の契約に従う。有限menuを未承認の26独立連続探索へ拡張しない。共通κ/物理scheduleは変えない。

NはPythonのexact int。言語間の巨大整数にはdecimal文字列を用いる場合も、入口で厳密にintへ復元して検査する。binary64へ丸めてから整数へ戻さない。MC backendのN上限はNumPy int64であり、超過は明示的なNUMERICAL/BACKEND_UNSUPPORTED扱い。別分布近似や小Nへの切捨てはしない。MC反復数と量子shot数は別field。

別の統計契約recordに`claim_level, target/law, count_rule, decoder, unresolved_loss, loss_support, numerical_scope, confidence/family, precision, repetitions, seeds, source`を持たせる。これは未確定の科学契約を表す最小run設定で、承認証明機構ではない。conditional診断か全risk上側判定かを先に選ぶ。

許可後の評価は既存API：

```python
simulation = S.simulate_prepared(prepared, repetitions=specified_MC_trials,
    seed=specified_seed, policy=adopted_IID_policy,
    allow_conditional_unqualified=adopted_conditional_scope,
    batch_decoder=F.decode_sampled_counts)
summary = S.summarize(simulation['trials'], target_energy_ha=declared_target,
    unresolved_loss=adopted_loss_contract, **adopted_CI_arguments)
```

欠測・物理未資格の既定gateは維持する。条件付きscopeを許可された場合だけ`allow_conditional_unqualified=True`。元mean qualificationやtarget packetをQUALIFIEDに書換えてgateを迂回しない。実データの呼出しは次の有限科学実行の明示許可後だけ。

## 3. B → N出力：三段階を分ける

| 出力水準 | 保持する値 | controllerへの意味 |
|---|---|---|
| 未評価/欠測/数値不安定 | config/allocation/source、全roundの欠測、理由、入力N、summary=null | N_X/Yとrmse_boundはnull。候補を勝手にdiscardしない |
| 経験的・条件付き診断 | evaluated_N_X/Y、full status counts、conditional RMSE、未解決率/CI、解決分MSE、prior外、target/law/claim、MC誤差前提 | `RPE_SHOT_DIAGNOSTIC_ONLY`等の明示status、N_X/Y=null、global bound=null。旧nominal・L1診断と同じく十分Nではない |
| 外部証拠により所望claimで適格な配分 | exact全round N_X/Y、同target/decoder/law/全事象lossの有効上側RMSE、confidence/数値/適用誤差とSOURCE | 現V2のerror contractを満たす場合だけ`assess`へ。最小Nや数学的無条件global保証まで意味を拡張しない |

既存N packetの形へ供給する場合、`configuration_id, policy, source, evidence_kind, rounds, rmse_evidence`を使う。診断packetの各roundには`identity/law_binding, evaluated_N_X/Y`を保持し、`N_X/Y=None`。`rmse_evidence.status='UNQUALIFIED', rmse_bound_ha=None, covers_state_target_alias_prior_decoder_all_rounds=False`を維持する。`summary.population_RMSE_CI`の名前だけで十分Nへ昇格しない。

正式に適格な外部N packetの現在の受入条件はcontroller本文が正本：

- `status='RMSE_SCHEDULE_EVALUATED'`、policy/config/SOURCE一致、全round exact N、evidence_kindを科学/人工で区別。
- 科学recordには科学的に採択・外部資格付けされたerror contract、target/state/prior/decoder context、全round/aliasを覆う証拠が必要。
- PRではfinite_K・signed_piece_kappa条件のQUALIFIED証拠、truncated PFなら元Hへの切断error証拠も必要。
- `rmse_bound_ha`がtarget以下というだけで他の前提を省略しない。confidence付きmodel claimを無条件保証へ読み替えない。

現コードはこれらを生成する科学的N solverではない。今回実装していない証拠producerを「接続済みの十分N評価器」と報告しない。L1経験的viewを必要とするなら別claimで保存し、既存guardを緩めない。

## 4. Gと選択への接続

適格N packetと、同configuration/round/candidate/law/price-domain/LOWERING/compiler/cohortの価格だけを`controller.cost`へ渡す。Cは測定を含むfull-wrapper RZ/shot、P0。normalization/B²やn_pathsを掛けない。Nが有資格でもC MODEL false/全law未資格ならGはnull。

同configの異allocationは内側層で別keyとして評価し、採択された有限family/未評価配分を保持する。既存`select`にduplicate configを渡さない。後に内側層を実装する場合も既存`cost`を再利用し、qualified配分の1代表と探索coverageを外側へ返すだけにする。既存`rpe_r_selection`のround別R最小化は固定alpha/beta等の旧HP契約下の機能で、全decoder結合Nや共通κの代用ではない。

price interval・ΣN b・共有covariance/同時confidenceは前回[平均価格精度計画](../h4_mean_price_validation_strategy_20261113/HANDOFF.md)を再利用する。4方式で同じ目標・claim水準を維持し、候補ごとのshot数だけで最適性/PR優位を決めない。今回G/winner=null。

## 次の担当

**signal担当**：候補正本/全mean/誤差/sourceを供給。**ユーザー＋B/A**：claim/loss/shot family/統計精度・有限scopeを採択。**Codex B**：許可後、供給signalからだけ統計/N資格と費用接続を評価。**C/D**：compiled平均の同law/domain/精度を供給。今回の技術成果を公開照合後mandatory STOP。
