# 最小の独立価格検証：有限scopeの決定表

2026-11-13 JST。**未承認・未実行の提案。** 保存pilotの再集計/fitを今回行わず、D/Cが方法と誤差根拠を供給した段階で、下の記号を埋めた1枚の有限実行案をユーザーへ提示する。既存の実行契約やSTOPは変更しない。承認証明書/新grant発行は不要。

## 0. D/Cから受け取る最小入力

1. **方法とSOURCE**：paired direct、導出済み解析補正、又は事前固定family。新旧lowering・compiler・全wrapper/P0、物理V2とC/S path定義。
2. **使用domain**：対象候補とround/axis、最大m、q/R/K/allocation/preparation/law。初回は固定PR1個を優先する案。11候補全てを一度に価格検証しない。
3. **free parameterとfeature**：p、既存calibrationのroleとrank、feature期待値を得る既存API、全domainに及ぶ誤差移送の根拠。
4. **law・rare/tail**：target P/Q、normal sampler/PRNGのjoint仮定、TV又は必要な適用条件、stratum確率と価格/残差のpopulation support。
5. **精度と統計法の入力**：セル別t/b又は相対誤差案、variance/rangeの根拠、同時α配分。pilotのSDだけで確定しない。
6. **実行可能性**：最大roundのpath生成・S・full-wrapper compile・保存の見積り/実装準備。現在m6 timingからの未検証比例外挿を上限にしない。

**A/B**はNの資格・許容集合𝒩・比較で区別したい資源差を供給する。未確定なら、Nを必要としないセル別精度の確認に限定する。保存nominalでGを試算しない。

## 1. 方法別の最小回数の形

| 方法 | 新calibrationの最小対象 | 新独立実測対象 | full-wrapper compile数の有限式 |
|---|---|---|---|
| 直接paired mean | fitなし。pilotは設計用に留める | 必要な各round jに、先に決めたn_jのnormal path。1cellの成功を他roundへ流用しない | 両軸を測る場合`2Σ_(j∈J_direct)n_j` |
| 係数導出済み解析補正（p=0） | 新fit0。導出/全domain証拠と期待featureをC/Dから受ける | 最大m12と内部challenge1roundの2点を最小案として、各n_hの独立path | `2(n_12+n_int)`。これはspot-check予算であり未測定roundを認定する数ではない |
| p parameter model | 既存校正rank rから少なくともmax(0,p−r)方向を補う。実際のcell数はrankとlack-of-fit根拠により固定 | 凍結後m12＋内部challengeの2roundが最小案。必要regimeが別にあれば、その数を事前に加える | `2(Σ n_cal,new + Σ n_holdout)`。p自体も未確定のため現時点で整数予算を捏造しない |

X/Yの全domain同値性が立証される場合だけ、その条件内で軸の実測数削減を検討する。2%/95%達成には最低4pathといったfloorだけでは足りない。各nは[方法別計画第5節](PRICE_VALIDATION_PLAN.md)で選んだ統計法・t/α・range/varianceの根拠から計算し、予定予算内で達成不能なら「未達」を許す。

固定nで直接meanを取得する場合、通常それ自身が最終推定cohortであり、同じmeanをもう一つのcohortで再測定することを一律必須にはしない。ただしpilotから方法/予算を決めた場合は、最終cohortを独立に新規取得する設計を推奨する。方法間の選抜・refitに使ったdataをholdoutと呼ばない。

## 2. 最初に選ぶべき分岐

- **Dが直接法を推す場合**：まず必要domainを限定し、cellごとのmean精度を確認する案。既存m0/m6だけを追加測定しても全13Cは完成しない。最大round又は高影響セルを直接測る有限案を別途選ぶ。
- **C/Dが全域の解析補正を提供する場合**：その全域条件を人工/静的に確認してから、最大m12＋内部1roundだけの独立challengeを検討。解析法が未測定roundを覆う理由を明示する。
- **少数parameter modelの場合**：P1/P2はdevelopment/calibrationとして使う案。保存dataへ新fitすることも新科学実行なので、対象record/cohortと1つのfamilyを固定して別許可を得る。p/feature/校正点/holdout/nを同じ有限案へ記す。
- **m12の実compileが不能又は全域誤差根拠なしの場合**：対象を実測/証明できるroundに限り、他をnullで返す。全13G/winnerを出さない。m7→m8→m9の無条件の連続実験を始めない。

初回の2holdout案を通過しても、全域model discrepancyが未取得ならまだ全13認定はできない。必要な未被覆regimeのchallengeか直接meanを追加する**次の有限案**を作る。小さなholdoutで任意の長時間誤差を否定できるとは主張しない。

## 3. 事前固定するrun内容

採択method、candidate/round/axis、law/strata、校正とholdout役割、固定n・seed生成法・出力path、最大compile/path数、精度判定式、同時confidence、support/数値精度、欠測/STOPを明記する。既存D runner・paired残差adapter・C構造packetを再利用し、Bでcompilerモデル/kernelを実装しない。

同じpathのS・C_X・C_Y・event digest・law/seed・native wrapper fingerprint・compiled SOURCEを対応させる。forced-rareは別cohort。複数modelを試すならその有限familyとselection/holdout契約が別途必要で、今回は1つに凍結する案を優先する。

完了条件は全指定path/軸の会計、凍結した誤差判定とdomain・CI前提の保持。精度が目標に届くことを「正常終了」の条件にせず、未達も保存する。source/law/loweringの不一致、値域/support超過、未定義価格、保存不能、未許可のdomain外が必要になった場合は部分結果を保全してSTOP。pathの棄却・再seed・holdoutでrefit・自動追加をしない。

## 4. 後続の分担

**D**：残差方法・p/特徴・新loweringの全wrapper実測・実行可能性。**C**：E[S]・期待feature・全domainの解析条件・law/rare数値根拠。**A/B**：N/RMSE資格と比較誤差予算。**ユーザー**：最小限区別したい資源差、confidence、価格domainとこの有限実行を採択/許可。

結果は既存B価格schemaで返し、total price intervalが未完成ならmodel false/finite cohort/nullを維持する。全13価格・適格Nが揃っても実G計算・正式4方式比較は別の明示許可後だけ。本計画公開後Bはmandatory STOP。
