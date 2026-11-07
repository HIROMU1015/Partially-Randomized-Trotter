# 学習費込み改善が確認されなかった要因：証拠と仮説

2026-11-07 JST。既存理論と固定pilotの引用。**原因を数値的に確定する新解析は行わない。**
対象は保存linear H4／1.00 Å／STO-3G／legacy DF12／8qubits・全4電子sector／L_D3／m0,q1,δ=.02,r1,K2／coefficient_atol0、N1 actual Q・selected full、N2保証ground・認証済みexact Fraction engine、理想gate・X/Y。[統合証拠表](SCV_TWO_PILOT_EVIDENCE.md)を数値の出典とする。

## 1. 非定数性から効果までは何段階あるか

以下は[既存層別CV設計§1–4][design]のscalar版。独立な学習履歴Tからλ_hを凍結し、fresh Q_h評価を行い、ν_h=E_Q[u|h]をexactに中心化すると

\[
Z=B-\widehat\lambda_h(u-\nu_h),\quad E[Z\mid T,h]=E[B\mid h],
\]
\[
\tau_h^2(\lambda)=\sigma_h^2-2\lambda\gamma_h+\lambda^2\Sigma_h,
\quad \gamma_h=\operatorname{Cov}(B,u\mid h),\quad\Sigma_h=\operatorname{Var}(u\mid h).
\]

Σ_h>0ならλ_h*=γ_h/Σ_h、oracle削減はγ_h²/Σ_h、有限学習損失は

\[
\tau_h^2(\widehat\lambda_h)=\tau_{h,*}^2+\Sigma_h(\widehat\lambda_h-\lambda_h^*)^2.
\]

これらの分解は同じbit lawに対する式。実装engineのbit lawと理想finite signalの差は既存engine allowanceで別に扱い、式の不偏性から数値bias=0とはしない。評価途中のSTOP/返却選別もこの固定quota完了モデルとは別条件である。

| 段階 | 区分と現状 | そこからは言えないこと |
|---|---|---|
| 特徴が変化 | CONFIRMED：indicatorもsignedも活動h1で非定数。signed Σ1>0はmetadata結果で既得 | γ_h≠0、説明力、利益 |
| 実際のbitとの関係 | UNKNOWN：真のγ_h・相関・oracle λ*は未取得 | 非定数性やbalanceから相関が大きいと置けない |
| fitできる | CONFIRMED：両pilot/両軸rank2 | 係数の精度・oracleへの近さ |
| finite係数のfresh効果 | SUPPORTED_BUT_LIMITED：224 Yは微小点低下、448 Yは点増加、Xは実質改善なし | 母分散の改善/悪化、有意差、同control族の無効性 |
| 全費用 | CONFIRMED：独立trainをfull価格で追加。保存scoreは両件/両軸で不利 | より厳しい精度や別の配分でも費用回収不能という一般命題 |

Signed SupportのΣ1>0は「特徴の分母が正」という入口を満たす。γ=0ならoracle λ*=0で、非零の有限係数はvarianceを増やし得る。一方、γ≠0でも学習損失がoracle利益を上回り得る。**どちらが実H4で起きたかはUNKNOWN**。旧bitへの再fitや新しい共分散算出はしていない。

## 2. Path平均変動とBornノイズ

m(ω)=E[B|ω]を解析用の記号とする。実数値をCVへ渡すoracleではない。B∈{−1,+1}について既存設計の全分散分解は

\[
\sigma_h^2=E[1-m(\omega)^2\mid h]+\operatorname{Var}(m(\omega)\mid h),
\]
\[
\tau_h^2(\lambda)=E[1-m(\omega)^2\mid h]
+\operatorname{Var}(m(\omega)-\lambda(u-\nu_h)\mid h).
\]

path-only metadata controlが説明できるのは後者の平均変動の線形射影部分で、固定pathのBornノイズ項は残る。Cov(u,B|h)=Cov(u,m|h)。h0はpath固定でpath平均変動そのものがない。[既存理論][design]

この制約は理論として確認できるが、**今回のH4がBornノイズ支配か、説明可能なpath変動が小さいかはUNKNOWN**。Yのsample varianceが大きいことだけでは二つの項を分離できない。Xのall-equal標本から母分散0やBornノイズ0とも言えない。

## 3. 有限学習と特徴の選び方

- **CONFIRMED**：Fragment群はQ質量balanceで選ばれ、bit相関を最適化していない。224 Y trainのu=0は2件で、少数feature出現という制約がある。
- **CONFIRMED**：Signedはrank2、X係数がほぼ0、Y係数は有限非零。符号/supportを使えてもstate・rotation・Dブロック・wrapperの応答を完全には表さない。
- **HYPOTHESIS**：metadataの粗い集約で信号との関係が弱い、有限trainで係数がずれた、fresh評価の変動が点差を反転させた、という説明はいずれも可能。
- **UNKNOWN**：各説明の寄与。448はcontrolもquotaも変わるため、224から学習量を増やした効果を隔離した対照実験ではない。

rank2は実行可能性の検査であり、係数の標準誤差や誤差保証ではない。Xの微小な非零λは凍結したOLS数値を保持したもの。これを0へ修正してpilotを再評価することはしない。Yで有限λが得られたことも真のγ≠0の保証にはならない。

## 4. Rangeはどこへ効くか

indicatorはu∈[0,1]なので保守的box幅D=2+|λ|、Signedはu∈[−1,1]なのでD=2+2|λ|。実際の活動幅の表示値とguardは[統合表](SCV_TWO_PILOT_EVIDENCE.md)を参照する。幅は特徴のscaleと係数を組で見て、式の係数だけでcontrolを順位付けしない。

両pilotはguardを通過し、独立calによる分散上界やBernstein/Hoeffdingの十分Nを取得していない。**今回のscoreはV̂×実現RZで、range penaltyをshot式へ代入した結果ではない。** Rangeが今回のscore悪化を直接引き起こしたという説明は支持されない。

将来の保証付き比較では、range増がcalの上界・Bernstein線形項・Hoeffding二乗項に効く。Hoeffdingだけならvariance低下はshot式に入らない。sample variance0も保証上界0ではない。[有限confidence整理][confidence]はこの接続を扱う設計案で、現在のpilotの精度保証ではない。

## 5. 学習費を回収するには

同じQ_h・selected policy・quotaのpaired評価では、metadata補正自体は量子回路や1shot価格を変えない。共通評価費を一度だけ支払い、SCVの参照費には独立trainを追加する。2件とも保存された追加の点改善はその負担に見合う材料を示さなかった。これは**SUPPORTED_BUT_LIMITED**な費用診断である。

既存[総費用設計][cost]の記号式を再掲する。凍結係数、正の期待full単価c_h、連続quota、母分散目標v>0を仮定して

\[
K_a=\left(\sum_h p_h t_{a,h}\sqrt{c_h}\right)^2,\quad
C_a(v)=F_a+K_a/v,
\]
\[
C_{\rm SCV}<C_{\rm OL}\iff K_{\rm OL}-K_{\rm SCV}>v(F_{\rm SCV}-F_{\rm OL}).
\]

これはoracle会計で、整数N、cal誤差、range、floor、STOP/返却を含むε/α保証ではない。本書で実H4のbreak-even値を作らない。現実には、評価のfull RZ削減が追加train/probe/calの全費用を上回る必要がある。固定quota評価だけでは量子費は節約されない。

一campaignの実現支出、係数・quotaを固定した条件付き期待費用、学習から反復するpolicyの期待費用を区別する。学習を償却するには利用回数・scope・独立性と費用境界を別に定める。都合よく過去trainを無料化しない。B1-OL側にも同じcal/価格情報/配分最適化機会が必要で、現在の固定quota参照は最適B1-OLではない。

## 6. 何を確認すれば説明・判断が変わるか

| 問い | 将来得る証拠 | 変わり得る判断／残る限界 |
|---|---|---|
| 現凍結λの効果は少数評価の偶然か | 同じcontrol/λへの独立fresh評価を事前固定し、variance差そのものの不確かさを扱う | 現モデルの追加評価価値。oracle係数や再学習policyは保証しない |
| 係数学習が主な損失か | 独立trainを含む別の有限設計。学習量と目的を固定し、全train費を記録 | 同control再学習の価値。単一再fitからpolicyの期待性能は決まらない |
| 別構造の情報なら学べるか | Product一致のknown mean/活動層確認後、必要と判断したfresh train/paired評価 | その候補の局所説明力。h2/h3の利益が全体費用へ十分かは別 |
| 保証付き費用を減らせるか | 独立cal、actual残差のrange/variance上界、対称な整数配分、期待単価/返却の保証 | 同finite target・ε/α下の費用比較。全round RPEは別接続 |

今回これらを実施せず、追加解析・実装・科学実行・重要レビューの正式採択も行わない。未知の原因を0や「Born支配」で埋めず、[次研究方向](SCV_RESEARCH_DIRECTION_OPTIONS.md)へ引き渡す。

[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107/STRATIFIED_CV_DESIGN.md
[confidence]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107/SCV_FINITE_CONFIDENCE_MAP.md
[cost]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107/SCV_TOTAL_COST_COMPARISON.md
