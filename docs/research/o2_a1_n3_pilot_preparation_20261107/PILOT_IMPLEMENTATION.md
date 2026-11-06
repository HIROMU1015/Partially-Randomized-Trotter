# 752-shot予備検証：driver/serializerの準備

2026-11-07 JST。**実装準備・人工検証まで。科学実行入口は無効。実752-shot、PCG64生成、実state/gate作用、Born、実compileを開始していない。**

## 準備した接続

- [a1_n3_pilot_campaign.py](../../../src/trottertracks/algorithm_codesign/a1_n3_pilot_campaign.py)：PilotDriver、Envelope、LegacyCollector、deferred connect_legacy_pipeline、draft設定/32domain宣言。
- [run_a1_n3_pilot.py](../../../scripts/tracks/algorithm_codesign/run_a1_n3_pilot.py)：own root/src bootstrap付き。`--show-plan`は設定表示のみ、`--run`は実入力/RNG/出力作成前にPREPARATION_ONLYでexit2。
- [人工test](../../../tests/tracks/algorithm_codesign/test_a1_n3_pilot_campaign.py)：invented law・固定bit列・mock価格だけで全752件のプロトコルを確認。

旧a1_n3_campaign.Campaign.group/worker_trial/serializer/bit先行保存/価格cache/reconcile、a1_n3_connectionのfacade/conditional sampler/stream/集計、cv.fit_and_freeze、rank/range、5cのallocate_evaluation/pilot_meanを再利用する。
旧scientific SOURCE、exact kernel、manager/observerのbytesは変更しない。新しい数値engineや汎用統計libraryを実装しない。

LegacyCollectorはfuture callerから渡されたfresh Campaignだけを接続する。SOURCEのcampaign/purpose、共有STOP event、fresh journal/cache、SOURCE hashを検査し、旧B SOURCEやwarm cacheを拒否する。
新Envelopeはcampaign/purpose/method/axis/stageを持ち、Driverはそれを検査してから学習/評価に使う。target/scope/law、quota、domain、全正層、record再利用も検査する。
A1のtrainはA1固有streamを保ちながらB1-OL条件付きlawで取得する。既存validate_batchを使うため、検査専用のmetadata copyでstream tokenをB1-OLへ写すが、原record/bit/保存/fit用streamは変更しない。
5層のうち同じcontrol patternを持つ二層を合併せず、既存cell-ID集計を使う。

## 取得・fit・配分・評価の流れ

1. A1：5層×16/軸＝160。既存OLSを軸ごと一回、rank/range/実係数をguard判定前に保存する。
2. B1-OL：5層×8/軸＝80。sample varianceと実RZ平均から5cの配分proxyを一回評価し、128/軸・floor4のquotaを凍結する。
3. A1/B1-OL：各方式128/軸、計512。A1はunconditional Q、B1-OLは直接条件付きQとexact質量合成。
4. 全stage RZ、評価だけのplug-in variance×observed RZ、zero flags、探索的分類、actual all-attempt記録を保存する。正式ε/α保証や期待費用改善にはしない。

fitは既存cv関数を使う薄い接続にした。旧Campaign.learnはTRAIN64と旧provenance経路を固定しているため直接呼ばず、Envelope検査の後に同じfit/rank/range関数を呼ぶ。
各axisのguard成立を確認して次へ進む逐次順は候補として明示し、同axis内の全層は既存poolで処理する。CPU/RAM/workerの固定上限は新設せず、将来の実資源で決める。guard-first順も採択待ちで、性能最適とは認定しない。

## 保存とfailure

PILOT_DRIVER_INVOCATIONはexclusiveな一回入口で、既存結果を上書きしない。これは二重実行防止であり、ユーザー承認を証明する仕組みではない。
受信bit/metadataを*_DATAへ保存してから価格の完備性を要求する。価格欠測はUNKNOWN、次stageへ進まない。
failure時は既存RuntimeへSTOP要求を保存→pool drain→既存reconcileで全worker末尾を保全する。complete groupだけの診断表と、all-attemptの正式な取得/費用会計を分ける。
PILOT_RESULTはNOT_CERTIFIED。zero varianceセルがある場合に有望性を断言せず、途中STOPではNOT_EVALUATED_DUE_STOPを返す。

## 人工検証と実行準備の限界

[ARTIFICIAL_TESTS.txt](ARTIFICIAL_TESTS.txt)：**34 PASS**（新driver12、5c設計11、既存campaign11）。
全752のmockフロー、実際の旧worker接続をtoy40件で通す検査、bit→価格順、旧campaign/重複/範囲外quota/target/stream拒否、range guard、価格欠測保存、zero variance保留、fresh cache/SOURCE不変、最小環境CLI、--run拒否を確認した。

人工PASSは実入力のCV改善・所要時間・総費用・本検証の保証ではない。
現在の公開SOURCEは**準備版**。実サービス読み込み/PCG64生成/pool起動のactivation、manager投入CLI、最終execution profile/入力bindingは、採択されたseed/規則と別途科学実行許可を受けてから固定する。
core driver/serializerと旧pipeline connectorは用意したが、今回このactivationを実施・有効化していない。実科学SOURCEとして確定済みとは称さない。
