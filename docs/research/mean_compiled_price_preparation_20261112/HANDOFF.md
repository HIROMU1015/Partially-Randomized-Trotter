# B向けHANDOFF：平均compiled価格の受動providerと期待値計算の核

実装SOURCE `09489eedbbd61c7396138634a896b8d8ecc75161`、[REPORT](REPORT.md)／[SOURCE索引](SOURCE_INDEX.json)／[将来科学案](SCIENCE_PLAN.json)。新15＋旧3の人工/技術テストPASS。新しい実H4/H6平均価格、S、残差、fit、N/Gは0。C1資格false/固定係数/結果/STOPは不変。

## 取込みとcallback

既存C branchをfetchし、B自身のSOURCE保全期間と差分を確認して、C固有module/testを選択して取り込む。B共通pipeline・N・optimizerを過去版へrestore/mergeする必要はない。

```bash
git fetch origin codex/r-variable-compiled-price-design-20261111
git restore --source=09489eedbbd61c7396138634a896b8d8ecc75161 -- src/trottertracks/mean_compiled_price.py tests/check_mean_compiled_price.py
PYTHONDONTWRITEBYTECODE=1 python -B tests/check_mean_compiled_price.py
```

testは既存C `check_r_variable_compiled_price`/B `check_pf_pr_rpe_pipeline`の人工fixtureと、C1 `Accounting`、旧connected-clusterのpure formを再利用する。Bでそのfixtureが無い場合は固定C sourceから必要な人工fixtureだけを用意し、実分子calibrationテスト全体を代わりに実行しない。

```python
from trottertracks.mean_compiled_price import MeanPriceProvider

c_provider = MeanPriceProvider(
    records=supplied_saved_prices,   # Bが既にSOURCE/identityを受け入れたrecord
    pipeline=common_api,             # B所有API。既定値はC基点の共通API
    selections=chosen_cohorts,       # {(candidate_id, "X"/"Y"): cohort}
    schedule_context=external_schedule_context,
)
# c_provider(task, requested_rows) はcopy/missingだけ。予測関数を呼ばない。
```

候補・軸に1レコードを返し、複数cohortがあれば明示選択なしにはSELECTION_REQUIRED/null。C1/Bのcohort名やSOURCEを変更・poolしない。記録が無ければMISSING/UNAVAILABLE、別prefix/δ/制御やPFはOUTSIDE_DOMAIN。PFはBが既存PF価格providerへrouteする。数値付きDIRECTはFINITE_COHORT_ONLYと正のn_paths、modelはqualification=falseとcandidate/domain/binding/compilerを要求する。保存FAILEDは失敗のまま。未知price/covariance/CIを0にしない。

既存B `run_missing`へ人工N/価格を渡す結合試験では、totals=None、lookup結果を保存後の自動再呼出しなしを確認した。実N×C/Gは呼んでいない。mean式やraw入力をproviderへ渡す機能は無い。

## Record・schedule対応

返値は既存の`identity/axis/classification/cohort/n_paths/source/mean/model/expectation/uncertainty/normalization_is_price_multiplier`を維持し、次のmetadataを追加する。

| field | 意味 |
|---|---|
| `price_status` | SAVED_DIRECT_FINITE_COHORT / UNQUALIFIED_MODEL / MISSING / OUTSIDE_DOMAIN / SELECTION_REQUIRED / FAILED |
| `mean_price_scope` | candidate全体・qPF/qRPE/R・formula/δ/prefix/K・11piece allocation等。旧fixed-R descriptorは新modelの資格ではない |
| `schedule_mapping.status` | NOT_SUPPLIEDまたはECHOED_NOT_SCIENTIFICALLY_VALIDATED |
| `schedule_mapping.context` | Bから受けたsource/schedule_id/M/kappa_schedule等の外部use-contextをcopy |
| `full_RPE_price_adopted` | false |

`law_binding`にはcandidate ID/R/K/qPFと元sampler定義SOURCEが必要。N側lawとの対応付けはBの既存SOURCE照合を保ち、宣言文字列を付け替えない。finite meanやモデルが表す期待値と、全lawの実compiled期待価格保証を分ける。normalization Bは価格へ再乗算しない。

RPEのκ_m scheduleをcluster係数κ1〜κ3と混同しない。Cはκ_m→T/q/M/Nを計算せず、schedule contextを資格しない。physical candidate/時刻/qPF/Rが価格domainを決める。旧`binding.T`は参照値を含むため、candidate.Tを用いる元のexact identity/時刻照合を保つ。新M/非dyadic κで旧qRPE/validatorが合わない場合は、Bが共通API/identity/use-contextを固定してCへ渡す。今回の確認対象B公開コードは724d/7dd6で、新RMSE APIの完成結合ではない。

## 期待値計算APIの範囲

`MeanKernel`、`MeanPiece`、`full_wrapper_mean`は、**supplied closed-state reward model**を入力して平均を計算する別API。callbackから自動実行しない。P/g・signed piece別law・前後head・wrapper・scalar-once・terminal flushをSOURCE付きで準備することは物理側の科学作業で、今回未実施である。

固定状態ならイベント/outer反復の二乗法でqR列を避けられる。C1の累積Z角やphaseが有限に閉じる証拠は無く、3状態人工kernelを実候補へ流用しない。binary64のmass checkはroundoff boundではない。未確定のclosure/terminal/lawを0で埋める機能も無い。

`local_residual_mean`は旧 `_iid_sequence_form`で内部C−S incrementの期待重みを作る。単独返値はfull_wrapper_mean=null。component/basis条件付きκ値、signed pieceごとのlaw、全境界残差が別途必要である。`combine_mean`にはC1 affine **または** LOCAL_C_MINUS_Sの一方だけを渡す。Sへ旧raw κ(C)を足す、両残差modelを重ねる、RTE内部だけをfull価格と呼ぶ使い方は拒否する。

既存のclosed native DP/connected-cluster係数を実H4 DF10/H6 DF19 full-wrapperへそのまま移送しない。E[S]の計算器、compiler残差の校正、全law/CIの資格は別の証拠であり、関数が数値を返せることだけではBの採用価格にならない。

## 次の科学実行の前提

旧C2の40/16 wrapper案は保全したまま自動実行しない。新RMSE/M/κ scheduleで必要なphysical domainと価格精度をBが提示した後、[SCIENCE_PLAN](SCIENCE_PLAN.json)に沿って候補/ID/seed/校正・holdout/SOURCEを固定する。

最初のS0案はC1保存4cohort・計32 pathの新mean方式によるdevelopment診断（compile0）で、保存値の科学再解析にあたるため別の明示実行許可が必要。C1の原本roleを保持し、新方式のfresh independent validationとは扱わない。次のfresh wrapper pilotは低/長qの4点×n4×XY＝最大32という条件付き予算案だが、候補未選定・SOURCE未固定で実行readyではない。局所残差calibrationやR方向追加は必要性と境界classを確定してから別の有限範囲にする。

次：ユーザーが具体的な科学範囲を判断し、Cがphysical kernel/runnerとSOURCEを固定する。Bは自身の許可scope内でschedule contextと既存recordの接続を担当する。Cの今回作業は公開/remote確認後mandatory STOP。科学再解析、compile/fit/予測、N×C/G・M/R/κ採択へ進まない。
