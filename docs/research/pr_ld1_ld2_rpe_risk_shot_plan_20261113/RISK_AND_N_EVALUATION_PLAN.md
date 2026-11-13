# 共通risk契約とN評価水準

2026-11-13 JST。**設計候補でありA〜Cの科学的主張水準・loss・confidenceを採択していない。** LD1/LD2の元full DF-H target、h=.1/M12、prior/decoder/alias/shift0、εE=.00159360010199040 Ha、元等分budgetとMinkowskiを保持する。[有限案](FINITE_SCIENCE_PLAN.md)／[SOURCE](SOURCE_AND_READINESS.json)。

## 1. 混同しない5層

| 層 | 入力/評価対象 | 現状と接続条件 |
|---|---|---|
| 固定ideal finite-K raw model | Aの全13rawを±1 outcomeのmeanとして使うplug-in joint model | raw値と選んだIID仮定に条件付き。corrected/logBから確率を作り直さず、Bを再乗除しない |
| finite-shot error | 各全round X/Y countを元逐次decoderへ渡す出力分布 | whole-vector Nに依存。round単独の局所精度やΣN最小では代替できない |
| 物理近似 | PF・finite-K・normalization・保存stateがrawへ及ぼす効果 | rawの直接energy riskにはこれらが既に現れる。同じbiasを後で二重加算しない。個別budgetへのL2分解は別証拠 |
| 数値不確かさ | raw/corrected/normalizer、matrix power、referenceの数値差とempirical allowance | gate PASSやuは認証raw誤差ではない。rigorous enclosure/感度scope/unknownを保持 |
| target/state/actual law | 保存solver proxyと同sector ground、idealと実samplerのjoint law | 厳密ground・actual law・実装誤差は未資格。量子化学DF前Hへtargetを変更しない |

LD1/LD2は同じenergy taskだが、prefix/tail λ・R・raw・candidate/law bindingは別。lawのクラスが同じでも全bindingを跨いでN証拠を共有しない。

## 2. 全decoder outcomeとloss

元scalar互換R3 decoderを維持する。前round推定が次のbranchを決めるため、全13round/26軸を一組で評価する。countはexact整数差/Fraction→binary64、rawからのPCG64 binomial順序も元SOURCEのまま。priorはdiagnosticで、decoderのclip・projection・正解branch選択に使わない。

- **DECODED**：zero/tieで未解決ではないというstatus。branch正解oracleではない。energy差を実直線上で評価し、circular wrapでaliasの大lossを消さない。
- **UNRESOLVED_ZERO_COORDINATES / BRANCH_TIE等**：energy=Noneと全trialを保持し、捨てて再sampleしない。
- **入力欠測/数値失敗**：未解決trialとは別。NOT_EVALUATEDで止め、成功試行に混ぜない。

解決集合D、未解決U、評価target E*とする。元実装のUNDEFINEDなら解決条件付きRMSE、P(U)、解決分MSE寄与を報告できるが、全population RMSEは未定義。未解決0観測でも真のP(U)=0や全branch正解を意味しない。

全事象riskを定義するには、(i) 元decoder出力は変えず、別途指定したenergy補完を持つ評価器、又は(ii) 明示的な補完family/loss区間のrisk envelope、の契約が必要。BOUNDED_INTERVALは(ii)であり、単一energy推定器を実装した意味ではない。任意loss区間を一つの確率変数Zに見立ててMinkowskiを適用しない。

## 3. 評価水準A〜C

| 水準 | 既に可能 | 最小追加・必要な科学根拠 | 利点・限界/出力 |
|---|---|---|---|
| A：固定raw条件付き経験的診断 | S.simulate_prepared、R3、summarize(UNDEFINED)、必要ならCP未解決率。W.prepareで全Nを管理 | 今回のtarget欠落を解消する別evaluation_target、有限N/seed/MC精度の採択。原meanは変更しない | 2候補の同配分比較、shot感度を低開発量で調べる。conditional標本RMSEは母集団上界・十分Nでない。N packet=null |
| B1：モデル内の全outcome感度envelope | BOUNDED_INTERVAL＋global loss support＋IID反復のHoeffding端点/CP | loss family/supportの根拠、failure budgetと精度、binary64の数値scopeを採択 | rare未解決を0扱いせず統計的不確かさを示せる。広い区間ならINCONCLUSIVE。特定estimatorや物理global保証ではない |
| B2：特定estimatorのモデルrisk/conditional CI | binomial/decoder/summary部品は既存 | 全事象energy評価契約又はconditional分子/分母CIの小追加、support/同時被覆。rare branchを含むrisk証明が必要 | 所望modelの上側riskで配分を限定評価可能。ただし入力・target・law移送と元予算は未達。既存controllerの正式Nへ自動昇格しない |
| C：元energy条件/Minkowskiへの接続 | canonical/assess/coverage/error-contract gate | 同targetのZ,Z0・coupling又は有効energy-bias理論、各L2 budget・raw/target/law誤差の包含、finite-K/signed条件、受入資格付きproducer | 元精度を満たす配分の科学的資格に進める。現在未実装/未資格。局所gate、A/B、旧S3だけでは不足 |

初回にAを選ぶ案と、B1の感度まで同じtrialで併記する案を提示する。正式にどれを研究主張とするかはユーザーが採択する。Cが完成するまで科学的に無意味ということではなく、各出力のscopeを限定する。

## 4. 統計精度と有限MC規模の決め方

n_MCは量子shot Nとは別。先にmetric、loss/support、MSE許容半幅η、confidence/familyを選び、既存trial_budgetへ渡す。lossが0≤L≤U、two-sided failure αなら保守的な既存設計は

`n_MC ≥ ceil[ U² log(2/α) / (2 η²) ]`。

RMSEの差を見たい場合も、sqrtの単調変換を通してMSE区間で比較する。点RMSEの近さから必要試行数を決め打ちせず、目標ε²又は事前指定した識別余裕との関係でηをユーザーが選ぶ。未知のrisk gapを0又は保存S3の見かけの余裕へ置かない。

未解決0観測時にP(U)≤p_tolというCP two-sided上端を目指す設計なら、選んだα_Uに対して`n_MC ≥ ceil[log(α_U/2)/log(1−p_tol)]`が目安となる。これは未解決率だけのprecisionで、energy riskの精度保証ではない。非零観測時は元CP区間を報告し、達成するまで無断で反復を増やさない。

**conditional riskのCIを要求する場合の追加案**：Y=1_D(Z−E*)²、p=P(D)、v=E[Y]を別に囲み、同時区間[v_L,v_U]と[p_L,p_U]からv/pを囲む。p_L>0のとき`[v_L/p_U, v_U/p_L]`等、p_L=0なら上端unknown。Yの未解決部分を0とするのは解決分寄与の定義で、全事象loss=0という採択ではない。現summarizeにこのconditional CIはなく、必要なら小さな追加と人工検証を行う。

loss supportにprior幅を使ってはいけない。元decoderはpriorにclipしないため、principal energy出力範囲（概ね(−π/h,π/h]）と採択target域、binary64境界を含む根拠が必要。前S3の[0,4096]や43,819反復、95%等を新契約へ自動採用しない。保守的supportで必要nが実用的でなければ、根拠付きのより強いtail/risk法を別技術scopeで検証する。小分布用exact_countsの全13round直積列挙は代案にしない。

LD×配分×metricの同時confidenceを事前会計する。独立seedでも候補選択による多重性は消えない。全menu同時上界又はscreen後の独立確認を設計し、同dataの最小点推定へ未調整marginal CIを付けて十分Nとしない。

## 5. 元Minkowski予算への接続

元contractのεqpe=εtrot=εE/2を維持する。単にrawモデル全riskがεE以下という結果は、その二項の個別成立証明ではない。

共通確率空間に全事象出力Z_θと基準Z0を定義できれば、

`||Z_θ−E0,DF||₂ ≤ ||Z_θ−Z0||₂ + ||Z0−E0,DF||₂`。

Z0を保存stateのfull-H reference modelにするか、単一位相モデルにするか、normalizationを同Bにするか単位振幅にするかで項の意味が変わる。N/decoder/未解決補完・state/target/law/couplingを先に固定し、各項を元budget内で資格付けする。couplingはMinkowskiを定義するために必要で、独立性自体は同不等式の必須条件ではない。現IID simulatorは各周辺を生成する部品であり、coupled差のL2 producerではない。

別途共通有効H/正しいenergy branchとそのbiasを証明できれば、統計RMSE＋energy biasという分離も可能だが、finite-K/round依存モデルから現在取得されていない。局所signal差、phase/T、normalizer単独をenergy biasへ変換しない。全事象Zが定義された場合の`MSE=Var(Z)+(E[Z]−E*)²`は恒等式だが、そのbiasにはfinite-shot/branch/量子化も含まれ得る。Varだけをfinite-shot RMSE、残りをPF/finite-Kの物理biasと自動同定しない。

参考の**直接risk移送**として、同一loss≤U、全RPEデータlawのTV≤Δ、proxy-ground距離≤dが有効なら`RMSE_target ≤ sqrt(R_model^+ + UΔ)+d`という既存A97146481の経路がある。これは元個別予算を自動で満たさず、今回その代替契約を採択しない。rawに現れたPF/finite-K/state効果を二重加算せず、数値/raw/law/targetの適用誤差を別に扱う。

独立shotモデルと認証raw座標誤差η_jがあれば`Δ≤min(1,Σ_j N_j η_j/2)`の保守的couplingを検討できる。Aのempirical uを認証ηに代用しない。このΔは**energy推定1回の全データlaw**であり、MC反復数を掛けない。samplerのcohort使い回し・shot/軸/round依存がある場合、meanだけではjoint lawを決められない。

保存stateの効果はrawに含まれるが、保存solverが同sector厳密groundという資格や狭いtarget包含は別。今回のoriginal Hは保存full DF-Hで、DF前分子Hへ主張を伸ばさない。target/raw/law/数値不確かさを元budgetへ配分する場合も二重使用せず、現予算のまま成立する根拠が要る。

## 6. 必要Nの判定と出力

Nは配分全体のexact整数ベクトル。AはDIAGNOSTIC_ONLY、Bは適用model/仮定を名前付きで限定、Cは採択contractと全証拠が揃う範囲でのみ適格Nへ進める。上側riskが閾値を跨ぐ場合はINCONCLUSIVEで、物理candidate全体の不可能性としない。下側riskが閾値を超えても排除できるのはその配分/contractだけ。

現BのSCIENTIFIC N受入にはEXTERNALLY_QUALIFIED policy・全round/error contract・finite-K/signed条件等が必要。A configのPROPOSED_NOT_ADOPTEDをin-placeで変更するとcanonical identityを壊す。将来の正式採択は元生成規則と、必要なら新policy identity/物理同一性bridgeを管理し、過去packetを再ラベルしない。

価格未取得中はrisk対Nとcoverageまで。N総数最小をG最小へ読み替えない。資格付きE[C]と適格Nが同candidate/law/profile/loweringで揃い、実Gの許可もある場合だけ、W.compare→C外側selectの既存exact会計へ進む。
