# 初回LD1/LD2評価：今回追加した統計機能と未採択条件

2026-11-13 JST。正本はB91d730の[FINITE_SCIENCE_PLAN](../pr_ld1_ld2_rpe_risk_shot_plan_20261113/FINITE_SCIENCE_PLAN.md)、B2906の[NEXT_SCIENCE_PLAN](../pr_ld1_ld2_statistical_caller_20261113/NEXT_SCIENCE_PLAN.md)。これらのenergy task/元Minkowski/identityと、LD1/LD2×共通whole-shot vector1本＝2 MC call又は2本＝4 MC callの未採択案を維持する。今回はいずれも実行していない。

今回、元callerにfixed-family risk intervalを任意追加できるようにした。元target・mean・decoder・全trial/statusは変更しない。新しい科学実行には、ユーザーが従来のtarget/risk水準/全事象loss/全13round X-Y vector/seed/反復・precision/confidence/有限scopeを採択し、明示実行許可する必要がある。

区間を要求する場合に追加で必要な判断：

1. **条件付き区間**：全解決outcome用の一様loss上界とSOURCE。保存priorや経験的最大値では代替しない。p_L=0/区間が広い場合はUNKNOWN/未決着を受け入れる。
2. **全事象model risk**：未解決loss familyを採択するか、その意味と全事象supportを定める。元energy出力の補完契約やMinkowski資格とは区別する。未採択なら正式model/energy riskはUNKNOWN。
3. **同時confidence**：LD×whole配分×metricの有限familyを事前固定し、全α_p/α_v/α_modelの和をfamily failure probability以内へ明示配分する。旧95%や旧反復数を自動採用しない。
4. **MC精度**：分子MSE寄与の半幅η_vなら既存trial_budgetのHoeffding式で有限nを設計できる。conditional比率の精度にはpの下限・CP幅も影響し、pの根拠なしに同じnで所望RMSE幅を必ず達成するとは保証しない。全事象側も採択supportと必要な識別幅から決める。

新しい区間は固定nのIID modelに条件付く。repeatを増やしてPASSまで回す、旧trialへ新metric/別lossを適用する、最小点推定だけを選んで未調整CIを付ける処理は今回の計画に含めない。追加範囲が必要ならその科学scopeとfailure配分を改めて定める。

元energy条件・十分Nへ進むには、採択する全事象energy task、元Minkowski各項又は承認された接続根拠、physical/raw/target数値・state・actual lawの資格が必要である。今回の統計区間だけではこれらを埋めない。Nの数値探索・十分N producer・正式G最適化は未実装/未許可のまま。E[C]もC/Dの同domain資格付き入力を必要とする。

ユーザーが条件・有限実行範囲を採択した後、Bが既存MC/R3/新interval hookで必要範囲だけ評価し、Cへ同signal binding付きの証拠を渡す。Dの価格研究を待って技術作業を止めたり、その実行へ介入したりしない。計算時は既存batchや独立callの並列化を利用し、共有環境を変更せず、過去worker数を新しい恒常capにしない。

STOP：入力/identity/law/SOURCE不一致、support違反、未知status、数値domain外等では未確認結果をPASSにしない。support不足やp_L=0はUNKNOWNで保持し、無断でloss/target/N/反復数を変更しない。十分N/global bound/G/winnerは未資格のまま保全する。

今回は人工・回帰検証と公開まで。**mandatory STOP。ユーザーの別明示承認まで新MC/保存数値再解析/実RMSE/N/Gへ進まない。**
