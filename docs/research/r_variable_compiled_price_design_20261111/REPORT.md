# R可変compiled回路価格モデル：技術調査・設計（2026-11-11）

**固定RのMq/Affine係数を別Rへ移植せず、全wrapperの構造価格＋少数のcompiled残差補正を第一候補とする。** 新Rのcompiled価格が必要であり、今回はモデル校正・価格予測・科学的精度の検証は行っていない。Rを含むbindingと既存保存lookupの技術adapterを実装し、人工/mock9件PASS。実signal/N/価格取得/fit/sampling/数値予測/数値再解析/N×C/G/最適化/詳細科学レビューは0。公開照合後mandatory STOP。

成果：[モデル設計](MODEL_DESIGN.md)／[最小科学実行案](MINIMAL_SCIENTIFIC_PLAN.md)／[B向けHANDOFF](HANDOFF.md)／[固定SOURCE索引](SOURCE_INDEX.json)／[保存価格の候補条件照合](SAVED_PRICE_COVERAGE.json)／[技術テスト](TESTS.txt)。

## 基盤・調査と再利用

前回Cの公開4cb61ef46を基点に、同じC worktreeを再利用し `codex/r-variable-compiled-price-design-20261111` へ独立した。新Rの設計を前回の凍結provider/publicationと分けるためで、既存branchの統合・削除・履歴変更はない。コードSOURCEは `216585dbdc26e3c1eea9f6054ee6357b139d31d0`。A/B worktree・未公開変更・raw/checkpoint・環境/process/cacheにアクセス/変更していない。

| 既存部品 | 再利用判断 |
|---|---|
| H4 Mq・独立q検証（1cf87201） | R4/δ.1/p1/K2の保存証拠とpaired分析・構造会計を再利用。係数bはR固定で、新Rへ移植しない |
| H6 Affine/V2（73dcbe23、凍結b8583796） | R16/δ1.6/p7/K2のq1/2校正・q4有限検証、元11piece構造会計を再利用。q1の定数補正がq4へ移らなかった既存結果も維持 |
| RTE 2/3-event・connected-cluster | inclusion-exclusion、basis/order層別、独立holdout、window計数の方法を再利用。旧DF/短時間/制御構成の係数は移植しない |
| 旧中央RTE価格projection | occurrence内の局所期待window法は参考。deterministic sweep/control/ancilla wrapperを除いた価格なので、完成Hadamard価格として流用しない |
| NativeCounter・basis/Z-fold | compiler前の構造Sとして再利用。Strong/Reference等の別系列係数や解析native数をcompiled価格としない |
| h46_m5_plan / h46_new2_core | weightsと純粋allocationだけを再利用。古いSYSTEMS中のH4 DF12設定等を今回のDF10へ持ち込まない |
| h46_long_time_native / rte_compiled_cost | 元sampler→signed-piece compose→full wrapper→同fingerprint/compiler価格の取得方法を将来再利用。今回は起動しない |
| 固定B共通APIとH4/H6 provider | canonical候補ID/identity/scheduleと保存lookupを委譲。共通N/G機能を再実装・変更しない |

元public価格表6種類をcandidate条件だけで静的照合した。今回の条件に合うPRは、確認範囲でH4 R4とH6 R16。後続固定成果を含めH4 q8/16、H6 q1/2/4の保存価格がある。H6 p5/R32や他δ、旧DF12のRTEデータをR比較の同条件点にしない。globalな全履歴探索はしておらず、未取得と断定する範囲はこのSOURCE索引の中に限る。新しい平均/残差/価格比は算出していない。

## 主要な設計判断

H4はL=qPF R、H6もL=qPF Rで、Rを元順11 signed pieceに整数allocationする。H6は整数R≥11、元largest-remainder規則・tie順序を保持する。R変更でevent数だけでなく短時間τ_j・order確率/角度・basis共有・head/tail界面が変わり、compilerの最適化も関与する。

第一候補はS(q,R;trajectory)＋θ0＋θq q＋θL qR。Sが構造の依存を担い、3係数は完成wrapperのcompiled残差で将来校正する。固定RだけではqとqRを識別できず、2つのq×2つのRの矩形を使う。3セルで校正・残り1セルを係数凍結後の独立検証とし、追加のpair/triple補正は不適合の理由が局所構造に対応する場合だけ検討する。現時点で係数は未取得、価格予測能力・精度は未資格。

有限cohort平均/path/model_mean/全law期待価格を区別する。order2未観測、R/q systematic、共有係数/X/Y covarianceと総CIは未知のまま。全費用に必要な価格精度は将来の固定N重みと候補差から決め、RPE αを価格信頼区間へ読み替えない。今回Gや費用寄与の実数値は求めていない。

## 実装・技術確認と次の判断

[価格側module](../../../src/trottertracks/r_variable_compiled_price.py)はdescribe_requestとRVariableSavedPriceProviderのみ。Bの既存c_provider形式で固定H4/H6 providerをrouteし、新Rはnull、固定R価格の誤移植を拒否する。候補R・allocation・SOURCE/compiler/law・axisを結合keyへ含める。数値価格モデルやoptimizerは実装していない。Bの共通interfaceの変更はBが管理する。

[人工/mockテスト](../../../tests/check_r_variable_compiled_price.py)9件PASS。実H4/H6の価格をテスト値に使わず、mock N/C・純粋schedule metadata・人工order patternだけを使用。既存provider/共通コード不変、R/axis区別、新R null、誤binding拒否、run_missingの無再呼出し等を確認した。実compiled精度や独立科学レビューのPASSではない。[CHECKS](CHECKS.json)にscopeと保全を記録する。

最小の次案は、H4 R8のq8/q16とH6 R32のq1/q2に各別n8・X/Y（最大64完成wrapper）、別unitのlegal rare対応2組/system（最大16）で計最大80。新R移送や長qの追加点は結果とA/Bの公開menuから必要な方向だけ選ぶ。今回は計画のみで、科学的実行範囲はユーザーの別途明示承認を要する。**ユーザー**がPhase 1の範囲を判断し、**A/B担当**がN/R menu・価格unitと接続条件を照合、**Codex C**は許可後だけ科学検証へ進む。

既存SOURCE/input/raw/result/grant/ledger/marker/KEEP/STOPを保全した。commit/push・remote SHA/treeと主要成果物の照合後、mandatory STOP。新しいRの実価格取得や総費用最適化へ自動進行しない。
