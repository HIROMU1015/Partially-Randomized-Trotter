# Risk intervalの意味・入力・whole-shot接続

2026-11-13 JST。[REPORT](REPORT.md)／[テスト](TESTS.txt)／[次の科学条件](NEXT_SCIENCE_PLAN.md)。元signal受入・evaluation_targetはB2906の[HANDOFF](../pr_ld1_ld2_statistical_caller_20261113/HANDOFF.md)をそのまま使う。本書は統計区間の差分だけを定義する。

## 条件付きMSE/RMSE

元decoderが解決したeventをDとする。DECODEDは正しいbranchの保証ではなく、解決済みの誤branch/aliasも含む。L=(Ehat−Etarget)^2を実直線上で評価し、Y=1_D L、p=P(D)、v=E[Y]を考える。p>0なら解決条件付きMSEはv/p。

固定n回のIID全RPE trialについて、元summaryからv_hatと解決/未解決件数を取得する。未解決率の元Clopper–Pearson区間を失敗確率α_pで求め、その補集合で[p_L,p_U]を作る。外部から**全解決outcomeに一様にL≤L_D**という根拠を与えられた場合、Y∈[0,L_D]なので、

`r_v = L_D sqrt(log(2/α_v)/(2n))`

`[v_L,v_U] = [max(0,v_hat−r_v), min(L_D,v_hat+r_v)]`

がHoeffdingの両側区間となる。二つの区間が同時に成立しp_L>0なら、

`conditional MSE ∈ [v_L/p_U, min(L_D,v_U/p_L)]`

と囲み、sqrtでRMSE区間へ移す。両区間の同時失敗確率は高々α_p+α_vで、p/vの独立性は不要。p_L=0では上端を返さずUNKNOWNとする。解決0観測のときに条件付き母集団riskを定義・認定しない。

Yの未解決部分が0であることは**分子の定義**であり、未解決eventの全事象lossを0とする操作ではない。supportがない・SOURCE/適用scopeが不足する場合もUNKNOWN。観測lossが与えた上界を超えた場合は拒否し、clipしない。prior幅や観測最大lossから一様supportを自動生成しない。

## 全事象riskとenergy資格

元summaryの経験的診断・CP・bounded-loss出力は保持する。追加する`risk_intervals.model_risk`は、外部入力`model_loss_contract.adopted=True`とSOURCE、元summary_optionsのBOUNDED_INTERVAL、全outcome用一様support、独立のα_modelが揃った場合だけ生成する。False/未指定なら、未解決0観測でもUNKNOWNである。

この場合も、既存summarizeの二つのloss端点に対するHoeffding envelopeを再利用するだけである。元decoderの未解決energyはNone、特定の補完推定器は未定義のまま。宣言したloss familyに条件付くmodel riskと、元energy推定器のRMSE保証を同一視しない。

`scientific_energy_contract.status=UNKNOWN`、`sufficient_N/global_RMSE_bound/G=null`を維持する。元Minkowski予算、finite-K/物理近似、raw/targetの数値誤差、保存state、actual samplerへの資格は追加計算・根拠が必要。rawモデルに含まれる物理biasを二重加算しない。新機能は正式N packetを発行しない。

## 有限familyと外部入力

`R.confidence_family(members, failure_probability=α_F, source=...)`は受動metadata処理。各memberは次を指定する。

| field | 意味 |
|---|---|
| id | family内の一意なcall名 |
| entry | 既存K.prepareが作る元config/whole配分/target/統計契約/実消費signal_input |
| repetitions/seed/source | その固定MC callの反復数・seed・結果SOURCE |
| failures.resolved_probability | CPのα_p。外部指定 |
| failures.resolved_contribution | Hoeffding分子のα_v。外部指定 |
| failures.model_risk | 全事象model envelopeを要求する場合のα_model |
| conditional_loss_support | `{upper,source,scope:'ALL_DECODED_OUTCOMES'}`。upperはHa² |
| model_loss_contract | 未指定/Falseが可能。Trueなら`{adopted:True,source,loss_support:{upper,source,scope:'ALL_DECODER_OUTCOMES'}}` |

全member/metricについて`Σα ≤ α_F`を元exact scalar会計で検査する。端数を避けるには既存`{numerator,denominator}`形式を使う。CPへ渡すbinary64 confidenceはfailure budgetを超えない向きへ変換し、表現不能な要求は拒否する。これ自体がSciPy内部の認証数値保証になるわけではない。

familyはデータを見る前に固定する。候補間/配分間の独立性はunion boundに不要だが、各call内のIID repeated trial仮定は明示必須。新candidateや別seed/追加反復を同じslotへ後付けしない。未計画のcall、LD/SOURCE/law/target/whole-vectorの取り違えはMC前に拒否する。複数LDを意図的に異なるmemberへ登録することは可能である。

同時被覆は新出力のCP・conditional interval・model intervalだけを対象とし、旧diagnosticのmarginal CIや未計画の追加metric/探索範囲には拡張しない。任意停止や途中で反復数を増やすconfidence sequenceは実装していない。計画の事前固定は科学的条件であり、承認証明・署名・grant発行を要求する仕組みではない。

## 呼出し方

```python
from trottertracks import rpe_risk_intervals as R
from trottertracks import rpe_signal_statistics_caller as K

# members中の全値はユーザーが別途採択する。ここでは実数値を設定しない。
family = R.confidence_family(members,
    failure_probability=family_failure_probability, source=family_source)

# 将来、有限科学実行の明示許可後だけ呼ぶ。元K caller/R3をそのまま使用。
evaluation = K.diagnose(entry, repetitions=repetitions, seed=seed,
    source=result_source, batch_decoder=batch_decoder,
    risk_options=dict(family=family, member_id=member_id))
```

risk_options=NoneならB2906と同じ挙動。新出力は`evaluation.risk_intervals`で、元evaluation_target/signal_input/allocation/diagnosticは不変。固定familyはSOURCEとともに保存し、原trialを保持する。区間計算を保存実trialへ後付けする場合も新しい数値再解析なので、今回の技術許可には含めない。

## N評価・C会計への引渡し

`risk_upper_input`はconditional_MSE_upper、all_event_model_MSE_upper、scientific_energy_RMSE_upperを区別する。前二つはモデル内Ha²上界が得られた場合だけ値を持ち、最後は常にnull。allocation/source/target/law/信号bindingは元evaluationの同じ欄で参照する。

Cの既存`supplied_evaluations`へ付加証拠として渡せるが、`n_packet=None`のままであり、C/Wの正式資格gateを通さない。conditional intervalが目標より小さいこと、nominal N、有限cohort価格や未検証modelだけから正式Gを計算しない。元energy契約を満たすrisk証拠・N資格と適格な同identityのE[C]が揃った後だけ既存cost/selectへ接続する。

**mandatory STOP**。本納品で科学水準・loss・precision/confidenceやN探索範囲を採択しない。
