# 新A：長round移送・全support資格の限定範囲

基準79cba2fc4882bc42282f8feee7257a69f7628072。ユーザーのStage 1〜4条件付き実行許可に基づく。既存pilotのSOURCE・raw・STOPは変更しない。

物理条件はLinear H4 1Å/STO-3G/DF12、元H ground供給・準備費除外、Partial S2/LD3/h=.02/K2、ordinary diag(I,V)、canonical law、18round q=2^m、energy精度1.5936e-4 Ha、失敗率.05、位相予算.4 rad。参照compiler A/Qiskit1.3.0、全結合profile、固定2-register gadget、F/G境界、phase規約、Simon–Love型強baselineと既存D境界融合を維持する。

Stage 1では旧48 paired sampleと保存native QPYを再利用し、D融合数、RTE event/basis-run構造、CSWAP/resetと実DAG critical pathを照合する。旧q1/2/8価格を再取得しない。source記号数とnative価格を区別し、診断的なaffine fitを証明とはしない。

Stage 1で有望性が残った場合の追加直接価格は、既存schedule上の(q,r)=(4,1),(32,1),(128,1),(256,2)。各条件・通常/rare stratum各2標本、計16 paired realization、X/Y、baseline/candidateのみに固定する。q4は短round欠落、q32はanchor外holdout、q128/q256は最初のr変更境界を診断する。seed namespaceはnewA-transfer-fixed-20261106、具体的seed生成式をSOURCEで固定する。結果を見て条件・標本数を増やさない。T合成は行わず未取得を維持する。

r>1の差分は既存RTE builderへ同じr個の明示eventを渡すだけであり、PF、law、配置は変更しない。必要な差分correctnessだけ確認し、旧suiteを再実行しない。resource測定は同じrealizationでpairedに行い、rare stratum確率で加重する。depthは実際のnative DAGで測定する。

Stage 2で構造モデルとnative価格の適用範囲を確定できず、全18roundへの価格移送の不確かさが解消できない場合はUNDETERMINEDでSTOP。Stage 3・4へ自動的に進まない。短いholdout一致だけで全q/rの数学的証明に置き換えない。

Stage 2を通過した場合に限り、Stage 3で既存局所区間評価・telescoping会計を再利用して新CSWAP/phase/inverse等の全support資格を評価する。保証不足ならUNQUALIFIEDでSTOPしStage 4は実施しない。価格と誤差の資格が揃った場合だけ、同じraw信号単位で新native diskを含む半径・phase・整数shot・alias・全18round費用へ接続する。旧8598 shotsは無条件に流用しない。

恣意的なCPU/RAM/worker上限は設けず、独立task数・実affinity・実際の利用可能資源で並列化する。共有設定や他ユーザーには干渉しない。旧C/C2、新B、新C(SERVER1)、P-D/BF/BM/PR-3等のSTOPを維持する。新科学契約、承認証明、grant基盤は作成しない。科学的STOPまたは限定scope完了後mandatory STOP。
