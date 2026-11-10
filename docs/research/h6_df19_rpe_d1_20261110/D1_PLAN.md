# H6 D1：固定4候補のscalar評価計画

**未実行。現指示書§8による実装準備と、次の科学評価の実行許可は別。** [REPORT](REPORT.md)の技術結果と前回保存4点を用いる。本書は許可の発行・証明ではない。

## 固定scopeと必要入力

- Input carrier：`99ad77ef949b70192918a8700b092dd33c17af8c`の`docs/research/h6_df19_rpe_preparation_20261110/SAVED_POINTS.json`。元保存データは`155529e20a8fac39e00786c40b57830b71203c4f`の長時間study。元H/state/scalar/fragment順・completion・SOURCE来歴を維持する。
- 対象：T=.8のFull19 `76601611270ed9f0677d`とPR5/R32/K2 `2606efdab20e3b8522e5`、T1.6のFull19 `a4f5ffc0f320644dadfa`とPR7/R16/K2 `2d2af6ca830dff14a695`。全てm5_best/q_PF1。候補変更・探索なし。
- Target：保存DF-H ground、1 kcal/mol（既存CA）、各algorithmのαtotal=.05、P=0。数値stateの既存証拠水準を保持し、元HへのDF誤差は追加しない。
- 初期β候補：total=.4/stat=.2。h=.8を第一候補、.1を代替。**採否は外部prior/alias条件に従う**。priorの出典なしなら未資格とし、保存E0から窓を狭めない。不成立時のh/β/閾値の事後最適化は行わない。
- 必要な誤差入力：同じtarget位相に対するcorrected complex disk、reference/stateの資格、normalization marginを含むraw誤差、implementation/law誤差、fresh IIDの条件・出典。未知はNone。数値allowanceのため主張は条件付きpilotに限定する。

## 承認後に同じcampaignで行う処理

1. `inspect_saved_point`で4候補と保存価格のbinding/compiler/scopeを照合する。保存log B、u/margin、reference/raw/correctedの出典を保持する。
2. 採択したhと外部prior、β、εE、αtotalを`prepare_conditions`へ明示的に渡す。prior未確定／alias超過なら数値入口を進めない。成立時だけ既存horizon関数からMを得て、全round/全軸の一様αを割り当てる。Mを旧13/18roundに固定しない。
3. 保存referenceと数値E0の位相対応、raw/correctedのnormalization closure、bias/u/margin、正半径・位相余裕をscalarで評価する。E0はreference診断にだけ使い、decoderのbranchを合わせるoracleへ渡さない。観測差やresidualだけから未知state/reference誤差を0にしない。
4. ground targetへのdiskが資格付く場合だけ`assess_supplied_round`から旧geometry/Hoeffdingへ渡す。βdet+βstat≤βtotal・rho>0と軸別Nを報告し、未資格／不成立はN=null。B、event数、q_PF、q_RPEを重ねてNへ掛けない。
5. `supplied_price_axes`で価格cohortを一つ明示選択する。PRは探索n9と独立確認n33を別表とし、poolしない。対応・N条件が成立する場合だけ、PF/PR各々の保存2round部分RZ会計を`aggregate_partial`で補助的に評価できる。h=.8ではm0/1、h=.1ではm3/4をそのまま使用し、m0/1へ改名しない。included/missing roundsを明示し、未評価m0..2を含む全horizon成功は認定しない。
6. 4 candidate×phase/radius/disk/β/α/N/資格と、価格出典／任意の部分費用をfresh outputへ保存する。全energy成功・全G・新しい最良候補を認定せず、公開照合後mandatory STOP。

実行条件が不足しているときは、その条件を未資格として同じcampaignで報告する。追加eigenproblem・state作成・signal・sampling・compileを自動で始めない。単なる条件付き前提の記述を厳密証明とは扱わない。

## 結果形式と次の境界

H4と共通の`target_id/candidate_id/m/q_PF/q_RPE/base_time/binding/axes/N_X/N_Y/alpha_X/alpha_Y/evidence_kind/qualification/full_rpe_energy_success`を使う。H6側のtargetは固定DF-H groundと明記し、欠測・未知費用区間／covariance／幅を0にしない。副資源は保存された同wrapperの値だけを対応付ける。

D2に必要な追加signalは、採択時間列で保存値がないT_m／構成に限る。D3に必要な価格は、その構成・時間域で資格付けたH6直接値又は明示したmodelだけ。いずれも今回未実行・未許可。H4 D1の科学実行には介入しない。
