# A/C → B：PF有効Hamiltonian診断のinterface

2026-11-13 JST。[REPORT](REPORT.md)／[人工packet](ARTIFICIAL_PACKET.json)／[有限案](SCIENCE_PLAN.md)。**実科学/保存signal再解析/N/RMSE/C/G0、人工15＋回帰8PASS。手法はPROPOSED_NOT_ADOPTED。** Bのrisk/shot/controllerを重複実装していない。

## 呼出しとPF再利用

```python
from trottertracks.pf_effective_hamiltonian import evaluate, common_step_schedule

# Uは既存A/Cの数値step、deltaは実幅。元SOURCE/identityを保持する。
# 実分子への以下の数値callは、固定candidateとscopeへの別途承認後だけ。
diagnostic = evaluate(
    U, delta, state=saved_state,
    operator=operator_descriptor,
    reference=reference_descriptor,
    branch=explicit_branch_policy,
    tolerances=explicit_numerical_policy,
    reference_hamiltonian=None,  # 供給する場合は追加Hermitian eigensolveを含む
)
```

PF formulaや回路を新規生成するAPIではない。C86f37の`Context.pf_step`、A既存`expm_step`等で作られたUを供給する。C stepはscalarを含まないため、Full/B0には元constant、B0Iにはconstant+保存tail identityを**step時間で一回**加えてから渡す。PR exact-tailなら元controlのconstant+tail identityを一回含める。operator.scalar_policyは`INCLUDED_ONCE_AT_STEP_TIME`を必須とし、scalar値・内訳・SOURCEもcallerのdescriptorに保持する。

head単独をPR control全体に見せず、B0/B0IをFull PFに見せない。sourceやcandidate IDの再ラベル・新canonical ID生成を行わない。real callerでは既存B validator/physical bindingでUの由来を確認し、このmoduleへopaqueな正本identityを渡す。

## 入力descriptor

| 引数 | 必須情報・意味 |
|---|---|
| operator | kind=`FULL_PF/TRUNCATED_PF/PR_HEAD/PR_EXACT_TAIL`、source、identity、元target_hamiltonian、retained_hamiltonian、scalar_policy。formula/prefix/順序/signed weights・sector・state binding・scalar内訳も供給元のidentityに保持 |
| reference | energy、definition、source、hamiltonian。targetと一致させる。保存solver proxy/解析的toy/認証energyを区別し、ground資格を勝手に付けない |
| branch | `{mode:'PRINCIPAL', source:...}`又は`{mode:'ENERGY_WINDOW', purpose:'EFFECTIVE_SPECTRUM_WINDOW', window:[lo,hi], source:...}`。windowは2π/Δ未満、RPE priorから自動転用しない |
| tolerances | source、unitarity/reconstruction（無次元）、generator_hermiticity_ha/energy_cluster_ha（Ha）、phase_guard_rad（rad）。数値resolutionの指定で、energy RMSE予算ではない |
| reference_hamiltonian | optionalな元full/同sector Hermitian行列。渡す場合だけ新固有値分解を呼ぶので、実科学scopeへそのcallを含める |

stateはnorm1e−12 gateを通る供給数値state。再正規化しない。branchはreference energyを使って自動選択せず、`reference_norm_less_than_pi`もPFのbranch・ground対応の証明にはしない。

## 出力の意味

| field | Bへの解釈 |
|---|---|
| status | `EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE`は数値診断取得。`UNKNOWN`は数値/分枝/入力未解決、`NOT_APPLICABLE`は有限K corrected/raw等の対象外 |
| H_eff | real/imag配列。選択branch上のgeneratorであり、元Hそのものと同定しない |
| spectrum | energy/固有vector、選択整数枝、state weights、近接energy groupとweight。縮退内の個別重なりはbasis依存。minimum groupが実際のstateで支配的とは限らない |
| spectral_shift_diagnostics | `minimum_branched_energy_minus_declared_reference_ha`。**branch依存のspectral値**で、rpe_biasではない |
| reference_comparison | optionalな元H固有basisとの全overlap行列、H_eff−Hのoperator norm。縮退対応は一意でなく、入力reference energyも置き換えない |
| diagnostics | unitary/Schur/gram/eigen/exponential residual、branch cut/window余裕、state/weight norm、optional Δ‖H_ref‖。経験的数値診断でrigorous enclosureではない |
| scientific_method_status / qualification | `PROPOSED_NOT_ADOPTED`、alias/ground/数値誤差認証false。外部仮定がなければ科学資格を追加しない |
| rpe_bias_ha / sufficient_N / global_RMSE_bound / G | **全てnull**。Bが別の全RPE誤差契約で判定する |

最小値シフトをtrot budgetへ入れるには、正しいbranch・比較対象full H/切断Hの意味・state overlapと他のspectral成分・全round共通step・decoder/alias/未解決loss・誤差coupling等の証拠が必要。取得したspectral値だけでfinite-shot RPEのRMSEに昇格しない。raw直接riskがPF効果を含む経路では、同じbiasを再加算しない。

## 全round/PR条件

`common_step_schedule(rows)`は、各mにT/qPFと`step_key={kind,binding,formula,prefix,actual_delta,scalar,source,...}`を受ける。keyにはsigned順・数値recipeも含められる。全0…M・同key・exact T=qPF×actual_deltaを確認するが、operator比較はしない。出力`COMMON_STEP_DECLARED`もqualificationではない。

PR exact-tailのkeyからだけR/Kを省略できる。exact control自体はR/K/κに依存しないからである。一方finite-K corrected/rawのkeyをそのように縮約することはできず、このkindはAPIが拒否する。early roundの幅/公式/prefix例外、ceil-gridのactualΔ変更、scalar不一致があるなら共通H_effにしない。

新PR controlを作る際、`pr_rpe_signal_provider._macros`をfreshに呼ぶとcorrected/raw/normalizerも計算する。**exact-tail行列だけが必要な限定pilotへ、その余分な科学callを無断で追加しない。** 既存cache/supplied operatorの利用又はSOURCEを明示した最小接続が必要。今回は人工macroでのinterfaceとR/K非依存だけを確認し、実H6例外や実PRへの移送を認定していない。

次の担当：**ユーザー**がbranch/reference・方法採用範囲と有限pilotを決定/承認。**A/C**が許可された物理step/spectrum証拠を供給。**B**が既存risk/shot評価へ接続する。今回の公開・照合後Aはmandatory STOP。
