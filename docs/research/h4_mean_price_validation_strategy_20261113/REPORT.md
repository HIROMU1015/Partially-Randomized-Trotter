# H4：全13round平均compiled価格の精度と比較可能性

2026-11-13 JST。**Nの十分性が未資格の現段階では、保存nominalをGの正式重みにせず、価格の同時区間・Nに一様な相対精度・条件付き誤差予算を先に整える。** Dの暫定2%/95%を採択しない。独立実測は凍結した方法の最大使用roundと構造上のchallenge点を優先し、m7→m8→m9の逐次compileを前提にしない。

[方法別検証計画](PRICE_VALIDATION_PLAN.md)／[有限科学scopeの決め方](FINITE_SCIENTIFIC_PLAN.md)／[D/CへのHANDOFF](HANDOFF.md)／[人工式確認](ARTIFICIAL_CHECKS.txt)／[既存回帰](REGRESSION_TESTS.txt)／[SOURCE・文書確認](TECHNICAL_CHECKS.txt)。今回は調査・設計・人工回帰のみ。実H4保存価格/N/S3の数値再解析、残差fit、simulation、compile、実G/winner算出は0。production codeは変更していない。

## 1. 固定SOURCEと現状

| SOURCE | 実際に確認・再利用した範囲 |
|---|---|
| B `51500b724a75f9535a2c831d50bcfe09eabb9326` | 4方式11案・N/C資格・旧新lowering・保存receiptの境界。menuは暫定のまま |
| B `2084791b1b09cff65e07fa8566bb734e1e6bff8e` | S3 HANDOFFとnull資格。固定trial原本/保存Nを開いて新解析していない |
| A `97146481f7b05e9418d8fc016f133c7b78d93d08` | RMSE接続HANDOFFの全事象loss/population risk/target/raw/law不足。価格精度だけで十分Nを作らない |
| D `39ddd59edf54a3a901e5302207b2765ddbd03fbb` | P2 REPORT/HANDOFF/PLAN、`p2_direct_pilot.fresh_summary/planning`本文。pilotを再集計・fitせず、m0/m6の開発証拠として参照 |
| C `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` | 全13round/26構造recordのHANDOFF、ideal finite-K E[S]とcompiled E[C]の区別 |
| 現B実装 | `pf_pr_rpe_pipeline.joint_uncertainty`、V2 `assess/cost/select`、`compiled_residual`、`bv2_mean_price`と関連テスト |

対象はLinear H4/1Å/STO-3G・保存tol-only DF10/weighted-Hermitian completion・sector(2,2)、元H/state/snapshot、PR S2/L_D1/K2/ΔPF=h=.1/M12・exactκ・元prior/decoder/budget/P0。価格は測定付きcontrolled full Hadamard wrapperのRZ/shot。Full PF/truncated/Full-H RTEにはそれぞれの物理recipeを用い、PRモデルの資格を移植しない。

D P2はP1 replay2 path後のfresh8 path（m0/m6各4）を含む完了pilotで、全law E[C]、rare/law距離、長round、2%/95%達成は未資格。freshも既に方法・精度の設計に見たdataなので、今後の方法選択後の最終holdoutとして使い直さない。P1/replayをfresh cohortへpoolしない。

新`CONTROLLED_ZZ_CX_CRZ_CX_V1`と旧generic RZZ-control loweringは別の価格context。compiler hashが同じでも混同しない。CのE[S]にDの平均残差を足す資格は、同じS定義・全wrapper・candidate・law・loweringと誤差契約が揃った場合だけである。

## 2. 固定した適格Nに対する誤差伝播

j=(candidate,m,axis)とし、μ_j=E[C_j]、点予測c_j、非負整数重みN_jを考える。価格が同時に|μ_j−c_j|≤b_jを満たす事象上では、

\[
|G-\widehat G|\le B(N):=\sum_j N_jb_j,\qquad
\widehat G=\sum_j N_jc_j.
\]

同時価格区間0≤L_j≤μ_j≤U_jなら、`G∈[Σ N_j L_j, Σ N_j U_j]`。セル間独立性はこの区間伝播に不要。ただしmarginal95%区間26本を、そのままjoint95%と呼べない。全使用domain・model選択を含む同時被覆が前提。

価格のconfidenceは研究全体の成功confidenceと別。N/RMSEやtarget/law証拠にも統計的failure budgetがある場合、それらとの同時会計が必要で、price95%だけで全比較95%としない。

2方式A/Bの適格Nと同じ価格policyが固定され、比較上意味のある差をγ>0と定めるなら、価格による曖昧さの目安は`B_A(N_A)+B_B(N_B)`。これがγより十分小さい精度を目指す。実判定は、例えば`G_A^U + decision_margin < G_B^L`のような区間関係で行う。区間が重なる場合は「未識別」であり、点価格の順位で勝者を決めない。γやmarginは研究上どの差を区別したいかというユーザー判断で、今回は数値採択していない。

Nが有限menu選択やpriceを見て変わる場合も、使用全セルに同時成立する価格区間なら、その同じ事象で選択後のNに伝播できる。後から選んだセルだけのmarginal CIでは同じ主張ができない。RMSE資格・N推定誤差・選択による不確かさは別に必要で、価格区間はそれを補わない。

## 3. N未確定でも使える3つの契約

| 契約 | 今できる設計 | 限界 |
|---|---|---|
| セル別の絶対誤差vector | b_jと対象domain/law/confidenceを供給し、B(N)=ΣN_jb_jを式のまま保存 | Nに上限がなければ一様な有限絶対G誤差は一般に出せない。nominalを代入して確定扱いしない |
| セル別の相対誤差 | c_j>0でb_j≤r_j c_j、全使用セルに同時成立させる | 下記のN一様性が得られるが、r_jの根拠にmodel/law/rare誤差も必要 |
| 将来の適格N集合𝒩 | Aが根拠付き有限範囲・候補集合を供給した後、sup_(N∈𝒩)ΣN_jb_jで比較する | 保存nominalを集合の中心・上限と無断でみなさない。集合が未取得なら数値化しない |

相対契約では、`w_j(N)=N_jc_j/ΣN_kc_k`（分母>0）とすると

\[
{|G-\widehat G|\over\widehat G}
\le\sum_jw_j(N)\,{b_j\over c_j}
\le\max_j r_j.
\]

したがって**全セルが同時に同じrの価格誤差内**なら、任意の非負Nにも同じrの相対伝播が成立する。これは予測cを分母とする契約で、真のGを分母にする率とは違う。c_j=0や未取得セルは絶対区間で扱い、b=0・N=0と仮定しない。N未資格でもこの関係は条件付き数学として使えるが、実資源比較の資格を与えない。

適格なbox N_j∈[N_j^−,N_j^+]が将来供給されれば、`sup B≤ΣN_j^+ b_j`。非負価格のG域も`[ΣN_j^−L_j, ΣN_j^+U_j]`と保守的に囲める。Nの依存制約があればその集合を使う。現段階でN上限を新設せず、最適N・winnerは計算しない。

**推奨**：まずNに依存しない価格domainとセル別区間/相対誤差profileを定義し、Nが資格付いた後に比較目的の精度へ絞る。m12は重みが大きいと仮定して選ぶのではなく、最大使用domainを覆うため検証する。重みの小さいセルの精度を緩める判断は、N又は𝒩が供給されてから行う。

## 4. 精度の中身を分ける

同じ対象lawに対し、必要に応じて`b_j = b_num + b_sampling + b_fit + b_model + b_rare + b_law`という上界会計を作る。ただし各成分の包含範囲を明示し、同じ寄与を二重に足さない。例えば全law通常samplingの有効MC区間が既にrareを覆うならb_rareをもう一度加えない。独立holdoutで得た真のmean対予測の全誤差区間に、同じparameter誤差を再加算しない。

- **sampling**：宣言したpath lawの平均推定の不確かさ。nはpath数でX/Yを2nとしない。
- **fit**：校正meanのMC誤差から生じる係数の不確かさ・共有covariance。
- **model**：特定familyがdomain内の真のmeanを表せない差。残差SEや係数CIだけでは覆わない。
- **num**：Cの構造期待値enclosure、数値期待feature等。構造intervalをcompiled価格CIとしない。
- **rare/law**：未観測stratum、idealとsamplerの分布差。標本追加だけではlaw biasは消えない。

Dの旧候補「MCを1.2%×E[S]以内」は、E[C]に対する1.2%でも全Gの1.2%でもない。D=C−Sの符号に一般保証はなく、μ_C≥μ_Sとも仮定できない。Sとの比を使う場合は、適格なμ_C下界又は全errorを含む比の証拠で変換する。暫定2%/95%・26軸Bonferroni・pilot分散plug-inのnは候補設計であり、今回新nの計算・正式採択はしない。

平均価格推定量の共分散Σが根拠付きで供給される場合、固定Nに対する統計分散は`Var(Ĝ)=NᵀΣN`。同pathのX/Y、round間共有係数、候補間共通校正も含める。path共分散と平均推定量共分散を混同せず、iid path平均なら前者/n。未知covarianceを0にしない。ΣだけではsystematicやCIを認定できない。CIを同時boxで渡す経路は、共分散を推測せず利用できる。

## 5. 現コードで満たせる範囲

| 既存関数 | 再利用と限界 |
|---|---|
| `compiled_residual.paired_residuals/cohort_summary/combine/b_price_record` | 同path C−S、cohort/role/axisとlaw/full wrapperを保持。有限cohort/modelの和は未資格のままで、confidenceや新modelを生成しない |
| `bv2_mean_price.V2MeanPriceLookup`、既存価格受入 | identity/law/SOURCE/cohort/domainとcompiled mean単位を照合。構造Sを拒否、null/model falseを維持 |
| `pf_pr_rpe_pipeline.joint_uncertainty` | supplied mean-estimator covarianceのWΣWᵀ、supplied simultaneous_price_boundsを固定Nへ伝播。既存PF/PRの2arm APIで、4方式joint CIを自動生成する機能ではない |
| V2 controller `assess/cost/select`、`evidence.plan` | 4方式の資格/不足管理とexact rational N×C。`cost`はjoint_covariance/total_CI=nullを保持するため、点費用だけでstrict dominanceを認定しない |

4方式の比較には、既存4方式のexact点会計へ、供給済み同時価格区間と固定Nを用いる上記の伝播を必要時だけ接続すればよい。旧2arm関数を4方式へ無理に流用せず、今は実N/C・完成不確かさpacketがないためコード拡張しない。巨大整数Nの点会計は既存exact経路を維持し、binary64の共分散/区間経路で整数を丸めた値を正式資源にしない。必要なら将来その入力範囲の精度検査を行う。

## 6. 結論・STOP

推奨は**直接paired residualを第一候補、解析式は全domainの根拠がある場合、少数parameter modelは直接法の予算/実行可能性が不足する場合**という条件付き順序。どれも未採択。全13roundへ使うなら最大m12を検証又は全域の妥当な解析証拠で覆う。m0/m6からの無資格外挿、2holdoutだけからの全domain認定、観測rare0の平均寄与0化はしない。

既存回帰37件と、実数値を使わないexact rationalの人工誤差伝播確認がPASS。最初は全suiteを同processに置いたため、先行suiteがimportしたライブラリを残差suiteの受動import検査が検出して1件FAILした。[初回記録](INITIAL_TEST_IMPORT_CONFLICT.txt)を保持し、元testを変更せず別processに分けて37件PASSを確認した。これはimport状態の干渉で、科学計算/compileが起動した失敗ではない。

実価格/残差の再解析・fit・全13C認定は0。11menuは提案のまま、十分N/global RMSE・G/winnerは未資格/null。A/C/Dの作業・SOURCE/raw/結果/KEEP/STOPへ非介入。

**次：D/C**が方法の自由度・domain・全域誤差根拠・独立path計画を供給し、**A/B**が比較のN/claimを整える。**ユーザー**が必要価格精度・confidence・有限科学scopeを選んだ後だけ、担当がその範囲を実行する。今回はcommit/push・remote照合後mandatory STOP。
