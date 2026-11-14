# Full-H P-2：採択前の有限実行案

**提案の対応能力を実装しただけで、D6/D7/H2のいずれも未採択・未実行。mandatory STOP。** 旧q1のみ18 compile案はrank不足のため実行可能案に含めない。

1. A/C/ユーザーがD6（22）、D7（24）、またはD7＋m6 n2（26）の一つと目的を決める。数はX/Y両軸の最大compile数であり、pathは独立/pairedを区別する。
2. 科学SOURCE、全cellのq/R/T・cohort・index・seed表、axes、fresh public/private、科学call範囲を固定する。既存record/alias/receipt/running jobを§10.5で再照合し、再利用可能なら新取得を減らす。P-1配線標本を較正へ転用しない。
3. 資源を観測する順序を決める。AのG566→P2/Q8→m5→大R→m6という案を保持するが、m5がHOLDOUTである条件と矛盾しないよう採択前に解決する。全CALIBRATION後にCのfit・係数凍結を行うならm5はその後となる。この順序変更を今回採択したわけではない。
4. 各cellを個別に許可された範囲で取得し、gate失敗時は部分receiptを保全してSTOPする。pairedは親G566 path0/1の既存eventとseedを使う。無断の追加path・retry・seed/round変更をしない。
5. 較正後のC fit、係数凍結、独立holdout、予測/不確かさの評価は、それぞれ許可範囲に明記する。現callerはfitや価格予測を実装せず、HOLDOUT前に公開凍結モデルmetadataを要求する。

大R cell、m5/m6の実メモリ・wall timeは未測定。P-1性能を新cellの測定値として扱わない。既存native/cacheと独立cell間並列化を利用できるが、親→paired、較正→凍結→holdoutの依存は保つ。CPU/RAM/workerの恣意的な上限や共有設定変更は行わない。科学実行時に実負荷を観測し、one-shot/既存STOPを維持する。

必要価格精度、confidence、価格区間のcontract、holdout許容誤差、law距離/rare寄与、長時間へ許す外挿範囲はUNKNOWN。採択前にC/Bの実際の比較要求から決める。rank3や有限holdoutだけでm12の平均価格やG比較の適格性を認定しない。追加roundをm7→m8→m9と逐次compileする計画、全候補の高精度取得も前提にしない。

今回の成果はcaller・metadata・人工回帰のみ。P-2科学取得、実cohort平均、実fit、m5/m6 holdout、E[C]/N/Gへ進むには別途明示許可が必要である。
