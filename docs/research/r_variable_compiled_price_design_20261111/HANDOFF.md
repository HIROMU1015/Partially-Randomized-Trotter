# Codex BへのR可変compiled価格側HANDOFF

技術SOURCE：`216585dbdc26e3c1eea9f6054ee6357b139d31d0`。基点はCの公開 `4cb61ef4633467446122954ed6cc43462e0306a9`。作業branch `codex/r-variable-compiled-price-design-20261111`。BのR可変branchは調査時の公開remoteに見当たらず、未公開branch/worktreeの読取・変更はしていない。現時点の接続試験は固定B共通API＋人工/mockであり、将来のB最新版への互換保証ではない。

## 取込み単位と共通API

コードcommitの変更はC固有2ファイルだけ。共通pipeline・H4/H6 provider・N・会計を変更していない。Bが自分のSOURCE保全/科学STOPを確認した後に、次を候補手順とする。CはB領域へ実行していない。

```bash
git fetch origin codex/r-variable-compiled-price-design-20261111
git show --stat 216585dbdc26e3c1eea9f6054ee6357b139d31d0
git cherry-pick 216585dbdc26e3c1eea9f6054ee6357b139d31d0
PYTHONDONTWRITEBYTECODE=1 python -B tests/check_r_variable_compiled_price.py
```

依存：Bの既存`validate_task/identity/structure/measurement/candidate_id/run_missing`と、固定H4/H6 factory。H6 factoryの未取込みは[前回HANDOFF](../h6_provider_integration_nc_coverage_20261111/HANDOFF.md)のf58428f30で解決できる。共通coreを過去版へ戻さず、Bが必要なfactory差分だけを管理する。保存lookupの再現に必要な旧model/価格Git objectは元A/C branchをfetchして用意し、欠けたobjectを新compile/予測で補わない。

```python
from trottertracks.r_variable_compiled_price import (
    RVariableSavedPriceProvider, describe_request,
)
# p はB自身のpipeline module、storeは既存SavedArtifacts。
c_provider = RVariableSavedPriceProvider(store.read, pipeline=p,
                                        selections=explicit_price_selectors)
# c_provider(task, requested_rows) -> 既存形式のX/Y record
# 将来の許可済み run_missing(..., c_provider=c_provider) に渡せる。
# describe_request(task, row, 'X', pipeline=p) は価格を計算しないmetadata。
```

R0の保存lookupをrouteするだけで、新しいR価格モデルは未校正・未実装。H4 R4/H6 R16以外の価格はnullを維持し、固定R係数の転用を検出して拒否する。PFはR/K=None。対象外δ/prefix/formulaは明示エラー、固定providerのより厳密なH/state/compiler等の不一致は従来のunavailableを保持する。PR unit/cohort/path selector、旧providerのq校正/検証/外挿資格もそのまま。Bは要求batchをdomainで分け、H6 m0〜3等の別δ/別prefixをこのadapterへ渡さず、既存C/別価格laneとして保持する。全13roundのmissingを無条件に一つのC callbackへ流さない。

## 必須のbinding・状態

| 情報 | field/扱い |
|---|---|
| T/qPF/δ/R_outer/K/prefix/formula | candidateとcanonical ID。R変更時はIDとstructureを更新し、旧N/Cのidentityを書換えて流用しない |
| qRPE/round | q_RPE,mは別field。qRPEをqPFまたはRへ読み替えない |
| signed recipe | structure.signed_weights/allocation/event_count/scalar_policy/allocation_policy。H6 R≥11、元ABS_TIME allocation |
| SOURCE/input/DF/state/prep/compiler | identityのbinding/contract/profileをBからそのまま渡す |
| sampling law | 既存law_context＋contract内の明示law SOURCE。Rでτ_j/order確率が変わる。descriptorの文字列だけでlaw一致を資格しない |
| 価格と種別 | 既存mean.rz_count/classification/model/provider_quote。direct、保存model、未検証外挿、未取得を維持 |
| 単位 | deterministic/path/cohort_mean/model_mean/law_meanを区別。n_pathsは量子shot Nではない |
| 不確かさ | 既存uncertaintyを保全。R_price_scope.R_systematic_uncertainty=null、全law/joint/total CIの未資格を0へ変えない |
| Rの資格 | R_price_scope.R_evidence_statusがFIXED_R_SAVED_LOOKUP_ONLY、NEW_R_CALIBRATION_REQUIRED、PF_R_NOT_APPLICABLEを区別 |

`R_price_scope.join_key`はBのfull identityとaxisをhashした**結合用metadata key**。Rが違えば別key。trajectory/価格cache keyではない。実価格再利用にはさらにkind/unit/cohort/path/event digest、wrapper fingerprint、compiler bindingを元実装で照合する。cohort名が同じでもRを跨いでpoolしない。共通校正・同一pathのX/Yは依存groupを共有し、独立と扱わない。

既存taskは複数Rを別candidate IDで保持できる。最終集計の前に、Bが各(m,方式)で選択するRを明示し、候補menu全体を一つのRPE経路として足さない。C adapterはR最適化もN選択もG集計も実装しない。

## 将来のモデルrecord（今回値は作らない）

[MODEL_DESIGN](MODEL_DESIGN.md)の構造S＋3係数残差モデルはsystem別・2軸対応。将来、固定calibration SOURCE、係数/校正cohort、独立検証点集合(q,R)、δ/prefix/recipe/compiler、price unit/law、coefficient covariance/shared group、R/q systematicを明示したrecordをCが返す。校正点・有限独立検証点・未観測整数R・長qを区別し、unknown domainは資格false/nullを維持する。

PR有限cohortまたはmodel_meanは全law期待価格ではない。B既定qualified_expectationを満たさず、有限scenarioを使う場合もユーザーが科学的解釈/実行scopeを明示する。新R予測を既存factoryのmiss補完へ忍ばせない。全lawや未検証外挿の採用policyを変える共通差分はBが別scopeで管理する。

## 検証・次の段階

[人工/mock9件](TESTS.txt)：R/axis/qPF/qRPEの区別、H6合法allocation、PF無R、条件逸脱、SOURCE/compiler/law key、fixed factory委譲、新R null、誤移植拒否、mock Nのrun_missing・再呼出し抑止、旧共通/provider SOURCE不変、既存局所window counterの人工patternを確認した。実H4/H6の価格・law・性能を検証したものではない。N/G入口を禁止したmockで、実compile/予測/fitは0。

次はユーザーが[最小科学計画](MINIMAL_SCIENTIFIC_PLAN.md)のPhase 1範囲を判断し、A/Bが公開R menu・N条件と照合する。Cは許可後だけ、新R校正と独立検証を行う。今回は設計・技術実装を公開しremote照合後mandatory STOP。新Rの科学実行や全RPE最適化へ自動移行しない。
