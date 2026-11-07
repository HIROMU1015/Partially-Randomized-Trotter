# Signed support：fresh学習・全5層paired診断の科学設計案

2026-11-07 JST / SERVER1。**未採択の設計。今回、新科学計算・保存値の数値再解析・実装・人工テスト・benchmarkを行っていない。**
第一候補の有限条件は[実行草案](SIGNED_SUPPORT_FRESH_PILOT_EXECUTION_DRAFT.md)、接続上の不足は[再利用計画](SIGNED_SUPPORT_REUSE_AND_ACTIVATION_PLAN.md)に具体化する。

## 1. 出発点と問い

結果 `b4512675ae1d650858256547d359210361d77f68`、科学SOURCE `71778dae6954ed9a7a6821062a2a6e6be17672c4` のmetadata-only結果は、h1でsigned supportが非定数であることを示した。保存値は

\[
\Sigma_1=69532480438749347841126879238803/73245175157838542641857534620449>0.
\]

これは固定REPORT/JSONの引用で、今回の再計算ではない。metadata identityは `c5e0c5c26b926eb4a73d1808a1c61222dc45934248c30d8aea4f8fc32c44b7e2`、control identityは `5e252ff9fa5682850683ced97d007b59fecbf6d7617c561adb4a34d4cc7c7f89`。

次の問いは、**このmetadataが測定信号の層内変動を説明し、fresh trainで凍結した係数が独立評価で追加分散削減を示すか、その点差に学習費回収の余地が見えるか**である。Σ1>0から相関・良い係数・総費用改善を推論しない。

## 2. 固定対象

| 項目 | 維持する条件 |
|---|---|
| モデル/geometry/basis | 保存linear H4 / 1.00 Å / STO-3G |
| Hamiltonian/system | legacy DF12 / 8 system qubits / 全4電子sectorのN2保証ground |
| split/PR | L_D3、m0、q1、δ=.02、r1、K2、coefficient_atol0 |
| law/層 | N1 actual finite-grid Q、既存B1-OL全5正層、固定a,b・products→rotation |
| policy/回路 | selected full、状態準備のみ除外、ordinary controlled diag(I,U)、scalar/identity/event phase保持 |
| 測定/engine | XとY、理想gate、認証済みexact Fraction engine、独立uniform53-bitモデル |
| 規約 | X=実部、Y=虚部、YはSdg→H、bit0→+1・bit1→−1、weight1。event phaseをbitへ二重に掛けない |

固定参照はN1 `744e1a09661ebba6841fe5faf456f88c41c47780`、N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`、engine認証 `fdf00cee3c87d804845497afc89ac6a71f3b3f59`。保存DFモデルの同じfinite targetに限る。H_D ground/PF有効状態への変更、他round、再preparation/CDF/state solve/認証は範囲外。

## 3. 推定器と追加分散削減の条件

u(ω)=s_rotation|S_rotation|/2∈[−1,1]、h0/1ではL0、h2–4ではL2がrotation。保存componentの同じ行のsign/supportだけを使い、絶対値・角度・Born・state・真のpath信号は特徴にしない。

各軸b∈{X,Y}で独立なh1 trainからintercept付き既存OLSを一回fitし、binary64 slopeをexact Fractionへ凍結する。λ_b,0=λ_b,2=λ_b,3=λ_b,4=0を取得前に固定する。

\[
\widehat\mu_b=\sum_{h=0}^4p_h\{\overline B_{b,h}-\widehat\lambda_{b,h}(\overline u_{b,h}-\nu_h)\}.
\]

p_h・ν_h・m₂,h・Σ_hは取得済みJSONのexact値を再利用し、新しいSignedControlのmoment有限和を実行しない。独立評価と固定係数の下で、E_Q[u|h]=ν_hにより元のfinite targetの平均を保持する。評価bitで係数・quotaを再選択しない。停止による完走者選別まで含めた無条件性能を、この条件付き不偏性だけで保証しない。

以下は既存SCV理論の整理で、未知のpopulation値を今回取得したものではない。

\[
\tau_{b,1}^2=\sigma_{b,1}^2-2\widehat\lambda_{b,1}c_b+\widehat\lambda_{b,1}^2\Sigma_1,
\quad c_b=\operatorname{Cov}(B_b,u\mid h1),\quad\lambda^*_{b,1}=c_b/\Sigma_1.
\]

\[
\tau_{b,1}^2=\tau_{b,1,*}^2+\Sigma_1(\widehat\lambda_{b,1}-\lambda^*_{b,1})^2.
\]

非零のc_bと適切な係数が必要で、非定数性だけでは十分でない。同quotaなら全推定量のpopulation分散差はp1²(σ²_b,1−τ²_b,1)/n1で、inactive4層は変わらない。既存3-controlは5層内で定数なので、その追加による層内改善を期待しない。本候補に3-controlを追加しない。

path固定時のBornノイズはmetadata補正で消えない。b_b(ω)=E[B_b|ω]と書いた分解

\[
\operatorname{Var}(Z_b\mid h)=E[\operatorname{Var}(B_b\mid\omega)\mid h]
+\operatorname{Var}(b_b(\omega)-\lambda(u-\nu_h)\mid h)
\]

は理論説明だけで、b_b(ω)を学習器に渡す設計ではない。真のvariance・E0・gap・state vectorもCV側へ公開しない。

## 4. 第一候補と、少数trainとの比較

第一候補は**各軸train64＋共通評価160、両軸合計448 physical shots**。評価quotaは各軸(8,128,8,8,8)。統計的に一意な最小必要量や最適量とは呼ばない。

| 候補 | train/軸 | eval/軸 | 総physical | 主なトレードオフ |
|---|---:|---:|---:|---|
| 第一候補 | 64 | (8,128,8,8,8)=160 | 448 | scalar slope学習とh1診断に余裕を増やし、他4層も診断する。学習費128 shotsが加わる |
| 費用を優先する代替 | 32 | 同じ160 | 384 | 64 shots分のtrainを省くが、係数不確かさ・rank/guard失敗に対して弱い。第一候補失敗後に実施する案ではない |

train64は、未知応答に対するscalar OLSの探索的な情報量と全full価格の負担を両立させる候補。非定数性があってもfinite trainで同じfeatureばかりならrank不足になる。保存momentだけから係数精度や検出力に必要なtrain数は決まらない。真の相関・価格・分散を未承認集計して最適化していない。

h1 eval128は学習対象の追加効果を診断する中心配分。他4層の8はsample varianceを残し、固定二分診断でも各4を保持するための探索的最低取得である。質量による最適Neyman配分ではない。全正層を残し、希少層のvariance点推定が不安定なことを明示する。旧(4,64,4,4,4)を全評価quotaとして流用していない。

## 5. Range guardの候補

実際の凍結係数から既存signed enclosureを計算する。B,u∈[−1,1]なのでD_b,1=2+2|λ_b,1|。第一候補は**D_b,1≤6、すなわち|λ_b,1|≤2**。inactive4層は幅2。

特徴幅2に対する補正幅を最大4へ制限する、データを見る前の探索的な安定性guardである。最適値・信頼保証・oracle係数を必ず含む上界ではない。旧indicator式2+|λ|や旧width4をsignedへ適用しない。signedでD≤4なら|λ|≤1となり、同じ数値でも別の制限になる。

guardなしは失敗による選別を減らす一方、極端なfinite-train slopeのrange拡大を無制限に許す。初回候補では制限を設けるが、実際の失敗率はUNKNOWN。厳しいguardは有用な係数も除外し得るため、guard失敗を「CV無効」や「安く完了した比較」と扱わない。失敗した全train費・係数・rank・rangeを保存し、clip/refit/追加shotで救済しない。

## 6. Paired診断と不確かさ

full評価の各軸で、既存paired_evaluateにより

\[
\widehat V_{OL,b}=\sum_hp_h^2s_{B,b,h}^2/n_h,\quad
\widehat V_{SCV,b}=\sum_hp_h^2s_{Z,b,h}^2/n_h
\]

とpaired covariance、平均差のsample-based varianceを保存する。平均差のvarianceはvariance削減量のCIではない。

追加shotを使わず、各層を事前指定したtrial ID前半/後半へ二分し、同じ凍結係数で2つの補助paired診断も行う案とする。全quotaが偶数なので半分は(4,64,4,4,4)。これは旧科学quotaの再採択ではなく、新しい全評価quotaの固定二分である。worker完了順では分けない。full＋half A＋half Bを各軸で計算し、paired_evaluateは計6回、量子データと費用は追加しない。

δ_full=V̂_OL−V̂_SCV、半分の同じ差をδ_A,δ_Bとし、fullと同じ分母scaleにそろえたd_A=δ_A/2,d_B=δ_B/2とH=|d_A−d_B|/2を補助表示する。Hは二分間の経験的ばらつきであり、標準誤差・CI・有意差判定ではない。δ_full>0でも片方のhalfが非正、またはδ_full≤Hなら探索的方向判定を保留する。安定した両halfの正方向も母分散低下の証明ではない。

この規則は未知の統計的不確かさを数値保証するものではなく、小さい/不安定な点差を過大評価しないための事前screen。正式な有意差・coverageを主張するなら独立データと追加の統計設計が必要で、今回の一pilotへ自動要求/追加しない。

## 7. 費用と結論の限界

同じ評価bitを両推定器へ使うため、固定quotaではSCVの量子回路は短くならず、追加train費が増える。両軸それぞれC_train,b、C_eval,bを保存し、baseline参照=C_eval,b、SCV参照=C_train,b+C_eval,b、全campaign実支出=Σ_b(C_train,b+C_eval,b)。2つの参照費は足し合わせない。

V̂_OL,b C_eval,bとV̂_SCV,b(C_train,b+C_eval,b)は探索的point scoreとしてのみ比較する。小さいscoreは精度を保証するshot削減、真の期待単価、学習費回収、期待総RZ改善の証明ではない。固定quota B1-OLを最適baselineとは呼ばない。

範囲・calibration・整数quota・同精度/返却条件を含む保証付き本検証では、B1-OLにも同じ校正・配分最適化機会を与える必要がある。既存[af47cf9の有限confidence/費用整理][finite]を後段の課題として保持し、今回の初回pilotに十分N・期待RZ・全round RPE保証を要求しない。

[finite]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107
