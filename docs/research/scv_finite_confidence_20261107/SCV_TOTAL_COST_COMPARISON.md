# SCVとB1-OLの総compiled RZ比較

2026-11-07 JST。**既存式・SOURCEの静的整理。実RZ、分散、shot数の新規算出は行っていない。**
対象は保存linear H4 / 1.00 Å / STO-3G / legacy DF12 / 8 qubit / 全4電子sector /
$L_D=3$ / $m0,q1,\delta=.02,r1,K2$ / coefficient_atol=0 / actual finite-grid $Q$ /
selected full / N2保証stateと認証済みexact engine / X,Y。
状態準備を除き、理想gate・固定保存DFモデルに限定する。
精度・range・失敗予算は [有限confidence文書](SCV_FINITE_CONFIDENCE_MAP.md)、
SOURCE版とAPIは [再利用表](SCV_REUSE_AND_VALIDATION_MAP.md)。

## 1. 比較対象を二段階にする

| 比較 | Baseline | 判断できる内容 |
|---|---|---|
| 同quotaのpaired診断 | 各軸$(4,64,4,4,4)$のraw B1-OL | 同じfresh評価記録での層内追加効果。最適B1-OLへの優位ではない |
| 保証付き総費用比較 | B1-OLにもcalibration・価格情報・誤差配分・quota最適化を認める | 同じfinite signal、精度、成功・返却条件での総費用差 |

224-shotの[固定草案][pilotdraft]は、h1 train32/軸、paired評価80/軸の**予備診断**である。
今回その実行・結果を取得していない。この設計では
$\widehat V_a=\sum_h p_h^2s_{a,h}^2/n_h$を比べるが、保証付きvarianceや十分Nではない。
差の不確かさには既存paired covarianceを使う必要がある。
$\widehat V_{\rm OL}+\widehat V_{\rm SCV}-2\widehat{\operatorname{Cov}}$は推定平均の差のsample-based分散であり、
「variance削減量そのもの」のconfidence intervalではない。

同じ160評価shotを両推定器が読むことによる物理支出は一回だけ。
B1-OL単独の参照費は評価分、SCV単独の参照費はtrain64＋評価160。
二つの参照費を足して実支出と呼ばず、baselineの測定費を0にも置かない。
固定quotaのままならSCVの評価回路は短くならず、trainの追加費用がある。
分散低下を実際の費用節約へ使うには、将来の十分shot数を減らす経路が必要である。

## 2. 母分散目標に対する連続oracle

以下は[既存SCV設計 §6][design]の式であり、有限confidenceの実用shot数ではない。
軸$b$は省略し、固定係数、正の期待単価$c_h=E_{Q_h}[C_h^{\rm full,no-prep,RZ}]$、
連続quotaと正の目標分散$v$を仮定する。

$$
V_a=\sum_h\frac{p_h^2t_{a,h}^2}{n_{a,h}},\quad
t_{{\rm SCV},h}=\tau_h,\quad t_{{\rm OL},h}=\sigma_h .
$$

同総shot数なら$n_h\propto p_ht_{a,h}$、同期待評価RZ予算なら

$$
n_{a,h}\propto\frac{p_ht_{a,h}}{\sqrt{c_h}},\qquad
K_a=\left(\sum_h p_ht_{a,h}\sqrt{c_h}\right)^2,\qquad
C_{a,\rm eval}(v)=K_a/v .
$$

oracle係数が利用できる緩和では$\tau_{h,*}\leq\sigma_h$ゆえ$K_{\rm SCV,*}\leq K_{\rm OL}$。
最初の候補で変わるのはh1だけである。
学習誤差、guard、整数化、minimum shot、未知単価はこの緩和に含まれない。
zero-variance層のoracle quota0を、未知平均を無料で知る実装へ読み替えない。

pre費用を$F_a$とする理想完了モデルでは

$$
C_a(v)=F_a+K_a/v,\qquad
C_{\rm SCV}(v)<C_{\rm OL}(v)
\iff K_{\rm OL}-K_{\rm SCV}>v(F_{\rm SCV}-F_{\rm OL}).
$$

$K$は既に平方された定義である。さらに二乗しない。
$F_{\rm SCV}>F_{\rm OL}$かつ$K_{\rm OL}>K_{\rm SCV}$なら、
小さい$v$で固定費を回収できる可能性がある。ただしrange・cal上界を入れた実装で
同じbreak-evenが成立するとは限らない。
これは有望性の数値閾値やGOを新たに採択するものではない。

finite trainでは
$\tau_h^2=\tau_{h,*}^2+(\widehat\lambda_h-\lambda_h^*)^\top\Sigma_h(\widehat\lambda_h-\lambda_h^*)$。
oracle改善をそのまま実現改善と置かない。独立calでactual frozen残差を囲えば、
係数損失を別の推定量で重ねて加算する必要はない。

## 3. 有限confidenceで実際に比較する量

同じ許容座標誤差$\epsilon_b$、信号失敗予算、bias allowance、返却条件を両方式へ課す。
各方式のcal上界$U_{a,b,h}$、range $D_{a,b,h}$、層別統計headroom $e_{a,b,h}$を用い、
既存sufficient_shotsの$\min(N_B,N_H)$を外向きceilで評価した**整数**quotaを使う。
全部の正質量層を残し、必要なminimum shotを同じ規約で適用する。
sample varianceを使う診断には2以上が必要だが、保証用既存shot関数の最小返値は1である。
旧pilotのfloor4を一般の保証要件として強制しない。

全段階を完了する凍結計画について、各軸のpre履歴を$H$、preの実現RZを$A_a(H)$と書けば

$$
T_a(H)=A_a(H)+\sum_{b,h}n_{a,b,h}(H)c_{b,h}.
$$

同じ履歴条件の比較でSCVが安いのは

$$
\sum_{b,h}\{n_{{\rm OL},b,h}c_{b,h}-n_{{\rm SCV},b,h}c_{b,h}\}
> A_{\rm SCV}-A_{\rm OL}
$$

の場合。左辺は実際に減る評価RZ、右辺は追加pre費用である。
平均化するpolicy比較では全項を該当履歴分布について期待化し、両方式の返却率も同時に扱う。
この不等式へ224-shotのplug-in分散と実現価格を代入して、保証済みと呼ぶことはできない。

次の効果は異なる。

| 効果・制約 | 有利になり得る条件 | 利益を失う条件 |
|---|---|---|
| 母分散の低下 | fresh train係数がbit/path変動を説明する | Bornノイズ主体、係数誤差、相関が乏しい |
| variance上界の低下 | cal件数が補正項を抑え、$U_{\rm SCV}$が小さい | 少数cal、all-equalを0認定、range capに張り付く |
| range | actual係数が小さく幅の増加が限定的 | Bernstein線形項・Hoeffding二乗項が増える |
| quota | 重要層で保証付き必要数が実際に減る | ceil、floor、rare層の必要最低取得が支配 |
| 固定費 | 保存済み係数を適法に固定利用する、または十分な評価削減 | 毎round再学習・cal費が大きい |
| 価格 | full条件付き期待価格が正しく評価される | 別scope価格、cacheを無料化、欠測を0扱い |
| Return | 同等の精度・返却サービス | 安い早期abortだけで費用差を作る |

Hoeffdingだけならvariance低下はshot式へ入らない。
Bernsteinが有利になっても、raw側は同じHoeffdingとの最小値を選べる。
同じ$e,\delta$でのrange対上界差の代数条件は[有限confidence §8](SCV_FINITE_CONFIDENCE_MAP.md)参照。
十分上界同士の改善は、その会計法での改善であり、情報論的な最小必要shot数の証明ではない。

## 4. B1-OL側にも同じ最適化機会を与える

既存[a1_n3_connection.choose_allocation][n3]は、
等しい層別誤差と、正整数$j_h$の有限compositionから
$e_h=e_{\rm total}j_h/(Jp_h)$を候補化する。
各候補の十分Nを作り、cap内で$\sum_h n_h\max(1,\widetilde c_h)$が最小のものを選ぶ。
これは有限候補内の価格proxy選択であり、大域最適化・期待費用保証ではない。
両方式へ同じ候補集合、floor/cap、価格規約、失敗予算の自由度を認める。
calによって選ぶe/quotaはfresh production前に固定する。

旧[allocate_evaluation][pilotdesign]の
$(1+\min(s_h^2,1))/2$ shrinkage＋largest-remainder配分は**pilot heuristic**。
そのproxyはconfidence boundではなく、SCV残差ではraw同様の分散cap1も無条件には使えない。
旧752-shot quotaや224-shot固定quotaを保証付き最適baselineと呼ばない。
旧N3のcal512、cap8192、J8等は当時の契約であり、今回の将来SCV契約へ移植しない。

片方だけにcal/probeを無料で与えたり、片方だけを固定quotaに閉じ込めたりしない。
共有取得を用いる設計では各方式に必要な参照費と、共同研究campaignの実支出を両方保存する。
同じraw calからrawとfrozen residualの上界を作る案は考えられるが、
係数学習からの独立性と全familyの同時保証を持つ**将来の別設計**であり、現224-shotの変更ではない。

## 5. 実価格と全段階会計

再利用候補は[SavedPrices.price][n3]。保存selected full列、ancilla/readout/measurement、
ordinary diag(I,U)、固定compilerを使う。
[selected_full_prices][prices]はlive request用であり、m0/q1の固定guardを持つ。
保存payloadはそのrequestと異なるため、後者へ無理に再preparationして接続しない。
今回はどちらも呼び出していない。

| 費用 | 量子RZへの計上 | 別に記録するもの |
|---|---|---|
| train | 全取得full interrogation | OLS、metadata setup、失敗学習も含む |
| allocation probe | 実bitを取るなら全shot | 配分の古典時間 |
| residual calibration | 全cal bitのfull価格 | 分散上界・interval処理 |
| production | 全整数quotaのfull価格 | 合成処理 |
| cache hit | 毎shotの量子RZを全額計上 | compile節約・cache lookup時間 |
| 評価用price-only cal | 量子bitを実行しない取得なら量子shot支出ではない | path生成、build/compile/cache、研究評価費 |
| abort/in-flight tail | 取得済み既知費を残す。未価格部分はUNKNOWN | 最終回収・分類・未返却 |

SCVのmetadata補正自体は量子回路を変えない。同じ$Q_h$/profileなら$c_{b,h}$はbaselineと共通。
価格とbitの層内相関があっても、fresh fixed-n計画の期待費用には$c_{b,h}$を使える。
実現RZを見てhard-budget停止するpolicyにはこのfixed-n式を使わない。

account_costs、summarize_costs、SCV cost_summaryは実現会計であり、期待価格のcertificateを返さない。
TrialCostのclassical=0 placeholderを実測0と報告しない。
setup、path生成、exact-engine simulation、fit、cal、compile/cache、親wall、worker区間和、RSSを別記録する。
worker wall区間和をCPU時間・elapsed wallと混同しない。未記録値はUNKNOWN。
exact-engine simulationの古典費と、仮想量子deploymentのRZも別の資源vectorである。

## 6. 条件付き計画、policy期待費用、実現支出

[既存の期待RZ保証計画 a53c1d6][expected]の区別をSCVへ対応させる。

| 量 | 意味 | 一回pilotで得られるか |
|---|---|---|
| $C^{obs}_a$ | 一attemptの全実現支出 | 価格・tail完備なら取得可能 |
| $T_a(H)$ | 係数・quotaを凍結した計画の条件付き期待費用 | 真の$c_h$または有効区間が別途必要 |
| $\Theta_a^{plan}=E_H[T_a(H)]$ | train/calから繰り返すpolicyの仮想完了仕事量 | 一履歴の$T_a$では代替不可 |
| $\Theta_a^{stop}$ | abortも含む一attemptの期待全支出 | STOP確率・到達履歴・未価格tailのモデルが必要 |
| $E[C_a\mid return]$ | 返却者に選別した費用 | 無条件期待費用の代用不可 |

calから決まるNとcal価格は依存し得る。
$E[N\,\widehat c_{\rm cal}]=E[N]E[\widehat c_{\rm cal}]$とはしない。
future価格lawが履歴から独立で固定なら、$E_H[\sum n_h(H)c_h]$という形を使える。
中断したpre履歴にも既知支出を残し、失敗者を捨てて固定費を推定しない。
成功までのretry費を$\Theta/p_{\rm return}$で勝手に追加定義しない。現契約は科学retryなし。

価格期待値を保証するには、同じcompilerの全support上界と独立価格cal、または十分な厳密有限和等が必要。
観測最大価格や旧affine/connected proxyを全support上界にしない。
既存計画には独立価格pool・必要時のみpre履歴反復・STOP補正・理想bit lawへの移送案があるが、
**未採択の設計**であり、SCVで実装済みの保証pipelineではない。今回その計算も実行しない。

engineの座標biasが小さくても、fit/ceil/guardを含むpolicyの期待費用が理想量子bit law上で
同じとはいえない。必要なら既存計画の履歴couplingと有界費用による移送幅まで含める。
signalの$b_{\rm engine}$を費用へそのまま足す方法ではない。

## 7. 期待総費用改善を判断する不足情報

第一の有限signal比較には、frozen actual係数・全range、独立cal上界、
両方式の整数N、全段階価格、返却失敗、同じ精度familyが必要。
期待費用まで保証するには、さらに全support価格根拠・条件付き期待単価の区間、
学習/cal履歴の変動、STOP、理想bit lawへの適用範囲が必要である。
費用証明の失敗予算$\gamma_{\rm cost}$を精度$\alpha_{\rm signal}$から無料流用しない。
両者の同時claimにはその和、必要ならabort分も明示する。

224-shotで確認するのは、その追加取得に意味がありそうかを判断する材料である。
sample scoreの好転だけで期待費用改善を採択せず、[最小判断表](SCV_REUSE_AND_VALIDATION_MAP.md)に沿って
calのrange penaltyと固定費を回収できる余地を検討する。今回break-evenの数値は算出していない。

[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107/STRATIFIED_CV_DESIGN.md
[pilotdraft]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/research/scv224_preparation_20261107/SCV_224_SHOT_EXECUTION_DRAFT.md
[n3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_connection.py
[pilotdesign]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_pilot_design.py
[prices]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_connections.py
[expected]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a53c1d6eedea0d85fb484425db9012aa89738129/docs/research/o2_a1_n3_decision_expected_rz_20261106/EXPECTED_RZ_COST_GUARANTEE_PLAN.md
