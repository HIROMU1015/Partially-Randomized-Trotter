# 最小の次段階：供給signal・有限whole-shot family

2026-11-13 JST。**未承認・未実行。** 今回の作業はSOURCE調査/設計/人工testまで。この案は過去S3の1回実行許可を拡張しない。新しい承認証明書・grantは不要だが、対象SOURCE/入力/claim/有限範囲へのユーザー許可が必要。

## 0. 実行前に決める項目

| 必要な決定 | 推奨する最小方針・未確定事項 |
|---|---|
| 物理candidate | 他担当が生成する1つの既存V2 candidateと全13raw/normalization/errorを優先。Bが分割/R/PF menuやsignalを生成しない |
| claim | 最初は条件付き経験的screenか、有効population riskによるN資格かを明示選択。L1ラベルを正式採択しない |
| target/loss | 元H/proxyの意味、未解決loss/補完family、branch/alias、数値・law scopeを採択。UNDEFINEDのまま全RPE十分Nを要求しない |
| shot family | 供給baseベクトルに対する最大3個の共通scale等、低次元のwhole-vector menuを初期案にする。scaleの値/元配分/round-axis制約は未指定・未採択 |
| 精度・confidence | screen/確認に用いるmetric、bounded-loss support/統計法、MC誤差幅とfailure budgetを事前に固定。元S3の43,819回を自動採用しない |
| 費用 | 同candidate Cの価格資格・区間。欠測ならN診断だけとし、資源最適配分/winnerを表示しない |
| SOURCE | 現B stats/fast/identityを再利用し、新run callerが必要ならそのsourceを公開。元入力と過去raw/STOPは上書きしない |

3個という数は初回に小さな有限familyだけを調べる**提案scope**であり、CPU/RAM/worker capではない。精度を満たすscaleや最適配分の保証ではない。全round自由Nの26次元探索は最初に行わない。

## 1. 条件付き経験的screenを選ぶ場合

**対象**：1 candidate×全13round/26軸×指定最大3 shotベクトル。signalは供給済み同じ1組を使い、物理再計算0。

**呼出し範囲案**：最大3回の既存`simulate_prepared`、各`n_screen`固定MC反復と事前固定seed、各回の全trialを保存。scalar再比較・別candidate・価格取得・実Gなし。主loss/必要なら既存感度summaryを同じtrialから評価し、新drawを追加しない。`n_screen`は採択metric/精度から事前に指定し、未確定のまま実行しない。

**得られるもの**：配分別conditional RMSE・解決分MSE・未解決率/CI・prior外・MC不確かさ・input qualification。診断/未評価/数値不安定を区別する。目的は供給signalのdecoder挙動と次のrisk法の必要性を判断することで、点RMSEで十分N/最適N/候補のglobal infeasibilityを認定しない。

screen後に統計法や配分を選び直す場合は、同じdataを未調整の確認CIとして使わない。追加配分・新seed・反復増・保存trialの新CI解析は、次の有限案としてユーザーへ返す。

## 2. 必要Nを資格付けする場合

経験的screenと同じ低次元familyを使えるが、全事象lossとvalid上側risk、target/raw/lawのclaim範囲が先に必要。次のいずれかを選ぶ。

1. **有限familyの同時評価**：最大3ベクトルの各上側riskへ合計αを配分し、固定nの全評価後、qualifiedな配分だけ価格と結合する。最大3 simulator呼出し。複数metric/target証拠にも必要な同時会計を行う。
2. **screen＋独立確認**：screen最大3ベクトルで選ぶ配分を1つに固定し、独立seedの確認を1回だけ行う。合計最大4 simulator呼出し。screenは選抜用で、確認riskの精度/方法は事前固定。確認失敗時に別の配分へ乗換えて同confidenceを繰返さない。
3. **非MC全risk法**：MCのsupport/rare制御が非実用的な場合、A97146481にあるzero式・逐次branch mass/tail法の技術検証を先に行う別scope。完成solverとSOURCEが得られた後の科学案は1 candidate×指定有限N familyの1回評価。現`exact_counts`を全13round直積へ投入しない。

これは選択肢であり、今回はどれも採択・実装していない。risk精度が満たせないならINCONCLUSIVEを有効結果とし、無制限にNやn_MCを増やさない。target/raw/law包含を必要とする場合はA/Cの既存有限案へ戻し、Bで重複開発しない。

既存S3の保存43,819trialを新上側risk法へ適用するなら、それはA97146481のB1に相当する**別の保存数値再解析**。今回の設計作業や今後の新candidate screenに黙って混ぜない。全mean再計算や新reference取得を一律の前提にもしない。

## 3. 統計規模と実行効率

n_MCは量子shot Nではない。例えば有効な全loss support UとMSE絶対幅η・failure αが採択された場合だけ、既存`trial_budget`等から有限nを決める。特定の精度が非実用的なら、その精度未達を報告する。SE/CPだけでrare branch lossの期待寄与を認定しない。

元の高速count/decoder/CIを再利用し、同candidateのsignalは再利用。独立candidate/固定allocationの実験は依存・seed・同時confidenceを保った上で必要に応じて計算processで並列化できる。共有資源設定・他担当processに干渉せず、過去runのworker capを恒常的な制限にしない。科学入力やdecoderを速度のために変更しない。

## 4. 出力・STOP

新run directoryにconfiguration/mean/target/shot vectorと元SOURCE参照、claim/statistical contract、全trial又は解析massの検証可能な記録、summaryとN資格・MC誤差/系統誤差の範囲を保持。入力meanや過去N packetへPASSを上書きしない。条件付き診断ならN_X/Y sufficient欄・global boundはnull、evaluated_Nを別欄とする。

SOURCE/candidate/law/target混在、欠測、raw範囲外/非有限、N非整数/ backend domain外、loss/support超過、既存decoderからの逸脱、保存不能などはSTOP。zero/tieそのものは通常の観測として全件会計し、捨てて再sampleしない。CIが目標に届かないことを理由に同run内で反復追加・再seed・shot変更をしない。

同精度のNと同候補価格が揃った後の**実G比較/最適化も別の明示許可**で行う。Bは既存exact会計を使い、N総数だけの最小を目的にしない。候補物理生成・C kernel/残差model・新compileは他担当のscopeで、Bが自動開始させない。

次：**ユーザー＋B/A**がclaim/loss/有限familyと精度を選び、**signal担当**が全roundの正本入力を供給する。**B**は必要な最小adapter/確認を同じscopeで準備し、明示許可後だけ評価する。今回の技術成果をcommit/push・remote照合してmandatory STOP。
