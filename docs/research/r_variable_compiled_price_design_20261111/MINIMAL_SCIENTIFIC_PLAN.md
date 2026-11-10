# 最小科学検証案（未承認・未実行）

この文書は次の実行範囲を具体化するもので、実行許可ではない。新しいcompiled価格、sampling、構造価格の再計数、fit/予測/残差評価は、ユーザーの具体的な明示承認後にまとめて実行する。N/signal/Gや候補最適化は含めない。既存の消費済みgrant/契約/STOPを再利用して起動しない。

## 保存値で現在判定できること

同一科学条件の保存Cは、H4 PR p1/S2/δ.1/K2がR4・q8（探索9/確認33）とq16（独立33）、H6 PR p7/m5_best/δ1.6/K2がR16・q1（探索9/確認33）、q2（校正8）、q4（独立8）。それぞれの役割・mean・path・validationは元SOURCEを保つ。H6 p5/R32/T.8等をR16との対応比較に使わない。

既存価格表6種類の候補条件照合は[SAVED_PRICE_COVERAGE](SAVED_PRICE_COVERAGE.json)。確認範囲で新Rの同条件compiled比較点はない。単にsignalの候補gridにRがあること、screeningのevent数proxyがあることはcompiled価格の取得証拠ではない。

今回は保存mean・残差・分散・cohortの再集計をしていない。次の校正でも保存されたCと構造Sのpath対応を優先し、旧Sを再計数しない。H4は独立qのPREDICTIONS/RESULT/PATH_RESIDUALS、H6はQ2_DIRECT/Q_PATH_RESIDUALS等のM0と直接値を対応させる。meanだけを無関係なpathと組み合わせない。必要な対応が不足した場合はそのセルを未資格とし、追加descriptor計算を明示scopeに含める。

## Phase 1：q×R相互作用を反証する小さいpilot

科学条件・compilerは[MODEL_DESIGN](MODEL_DESIGN.md)どおり。以下は候補計画で、A/Bの候補menuと照合してからユーザーが承認する。

| system | qL / qH | R0 / 新R1 | 保存校正2セル | 新校正1セル | 独立検証1セル |
|---|---|---|---|---|---|
| H4 DF10 | 8 / 16 | 4 / 8 | (q8,R4)探索n9、(q16,R4)既存n33 | (q8,R8,T.8) n8 | (q16,R8,T1.6) 別n8 |
| H6 DF19 | 1 / 2 | 16 / 32 | (q1,R16)探索n9、(q2,R16)既存n8 | (q1,R32,T1.6) n8 | (q2,R32,T3.2) 別n8 |

新規は各system 2セル×8 whole-path、両system計32 whole-path、同じpathでX/Yを共有して**最大64完成wrapper compile**。PFの新compileは不要。n8は少数反証pilotの規模で、平均価格の所望精度や全law tail保証を満たす標本数と認定しない。

1. sampling前にSOURCE・candidate ID（Rを含む）・11piece allocation・cohort/seed規則・compiler・price unitを固定する。新cohort間および旧cohortとのseed重複を照合する。X/Y以外を独立標本として複製しない。
2. 保存R0の2セルと新R1/qLの1セルで、同一pathの残差C−Sを使い3係数を校正する。systemを分け、X/Yの対応を保つ。既存paired分析/汎用multiresponse fitを再利用できるが、旧q weightingや等分散仮定を自動継承せず、校正cohortの名目的共分散を明示する。
3. qH/R1を見る前にモデル・係数・評価指標・domain/不確かさをcommitで凍結する。校正3セルは飽和しており、systematicの検証は外部セルが担う。
4. qH/R1は新しい独立cohort。価格平均のbias/SEとpath価格のSD/RMSEを分離し、X/Y covariance・共有校正依存を保持する。holdoutによる再fitや係数/特徴の後出し選択はしない。旧R0の独立q成果は、今回使えば既知の校正/診断であり新しいholdoutではない。
5. actual wrapper fingerprint・compiler hashが一致する旧価格のみ読取再利用する。Rが違うcandidateや相似したevent数だけでcache hitへしない。元builder/sampler/native lowerer/full wrapper・元の正しさ検査を再利用し、phase/identity/scalarを保つ。

## Phase 1の希少event診断（別unit、上記cohortへpoolしない）

各systemでlegal order0/order2の**対応2組**を事前固定する案。H4はhead/tail境界と異basisを含むtail内の2 context、H6は正piece/負pieceの2 contextを優先する。元component/application順・有限K2 lawのsupportを守り、order2の2 productsとrotation、identity/relative phaseを保持する。

各組は基準/変更pathの2本、各X/Y。両system合計最大**16追加wrapper compile**で、Phase 1上限は**80**。同じ基準path/fingerprintの価格が保存済みなら再利用できる。対応差は局所メカニズムの診断であり、forced pathを無条件lawからの標本やcohort平均へ混ぜない。

2 contextでは全rare配置・複数order2・全basis組合せを網羅できない。層重み/未観測層の価格を資格できなければ全law期待価格は未資格のまま。観測order2ゼロや有限最大値で未取得寄与を0/上界にしない。追加K1–K3のcompiled校正が必要なら、Phase 1の結果を基に別scopeを具体化する。

## Phase 2以降：必要な方向だけを追加

Phase 1で新R×qの整合性を確認できても、新しいRへの移送や長qを検証済みとはしない。

- 未使用Rの検証候補：H4(q16,R16)、H6(q2,R64)。各system別n8・X/Yなら最大32 compile。新R点を学習に使わず、凍結モデルを検証する。Phase 1とは別の実行承認とする。
- q方向が支配的な不足なら、H4(q32,R8)またはH6(q4,R32)等の、既存qHより長い一点を優先する。全13roundのqへ一括compileしない。費用寄与が大きい実使用領域をA/Bの公開成果から選ぶ。
- 中間RもH6 allocationの段差で未検証となる。二つのRを検証しただけで整数R全域の補間を採用しない。必要なら代表的なallocation変化点を追加する。

追加点・標本数・局所補正の増加は科学scopeの変更なので、必要量と判断内容を報告してから明示承認を受ける。CPU/RAM/workerに恒常的な独自上限は設けず、許可された独立処理を効率化し、共有環境や他担当を変更しない。

## 全RPE比較に必要な価格精度

まず有限cohort scenarioの反証pilotと全law期待価格の資格を分ける。BがAのN(T,R)を固定して価格区間/系統誤差を伝播する際、ΣN|ΔC|が比較対象の費用差を覆うかを判断基準にする。差が区別できない場合だけ、支配寄与のセルへ追加標本/直接価格/モデル検証を割り当てる。共通θ・shared sourceとX/Yを保ったΣCが未取得なら総CIはnull。

一律の「数%以内」で全RPE採用を決めない。Nの条件付き資格、finite CDF law/ideal lawの差、rare寄与、長q/R systematicが残れば、そのまま未資格としてBへ返す。今回、費用差・寄与重み・必要相対精度の実数値は求めていない。

次の判断担当はユーザー（上記Phase 1の対象・80上限等を承認するか）とA/B（公開されたR menu・N条件・採用するprice unitの照合）。Codex Cはその許可があるまで科学実行しない。
