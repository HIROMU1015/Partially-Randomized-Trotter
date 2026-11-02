# 利用できる構造と反証条件

以下は既存SOURCE・式の照合結果で、新しい科学仮説の正式採択・新定理ではない。参照IDはsource_catalog.json、関連結果はcrosswalkへ戻る。

## H1：qとRTE内部反復の分離

PR2_SIGNAL_CODEの実装はdelta=T/q、tau=lambda_R*delta/r、B=b_K(tau)^(qr)を使う。したがって**同じS2配置・tail/lambda/p・T・K、整数R=qr、同じmicrostep law**ならtau=lambda_R*T/Rとraw法の反復数はqによらない。PM0_CODE.same_R_groupsとPM0の56 groupが既に確認した範囲である。

持ち込めるのは同じ分布・normalization計算の再利用class。deterministic half-sweepの挿入頻度・順序、scalar/control、境界fusionはqで変わり、非可換なのでsignalが同じとは言えない。compiled C、bias、headroom、N、Gの不変性やqの単調則はない。PM1/PM2ではdiscard総biasがqとともに悪化する保存例もあり、純PF/discard分解は欠測のままである。

高次PFではpiece係数a_jと整数r_jに対しtau_j=lambda_R*T*a_j/(q*r_j)。同じR総量だけでは、(signed tau_j, cutoff, q*r_j, label law)の対応は保証されない。配分のfloor/最低1/remainder処理、family変更、prefixによるlambda/p変更、pieceごとのK変更は同値classから外れる。

人工検査では既存allocation関数だけをAST抽出し、H2/H4/H6とは無関係なtoy係数(2,-3,2)、同じ総R=12のq1/q2で整数配分が比例しないことを確認する。これは新しい実分子結果やYoshida係数の検査ではない。法の対応が取れた場合も、qを跨いで費用receiptを再利用する根拠にはしない。

## H2：finite-RTE固有の低情報量

|取得可能な構造|必要入力と費用|選択へ使える範囲・未確立事項|
|---|---|---|
|finite paired order weights/probabilities/normalizer|RTE_CODE.finite_rte_distribution、固定tau/K。入力lambdaは既存DF抽出が必要|sampling/状態不要。normalizationが総Gを順位付けするとは言えない|
|signed scheduleと整数r配分|SCHEDULE/ALLOCATION、係数・q・T・law|metadataから可能。signed順序を絶対時間へ潰して誤差を推定しない|
|label power sums、隣接同label確率chi|与えられた全pからの和。p取得・保存精度の費用は別|形式returnの情報であり、実際のbasis/Pauliの追加関係・native価格を尽くさない|
|G6 Green局所query|G6_NOTE Eqs.8–11、G6_CORE.ReturnKernel、正有理p、odd m、short step等|全word辞書不要で係数を取得。全label取得やbit精度は無料でない。既存Green原理自体は既知|
|U upper、B_ord、moment/range予算|G6_NOTE Eqs.12–18。root interval・raw-reduced massの計算。finite-bit補正も必要|全signal/referenceなしで使える条件付き保証情報。Uは真のB_fullでなく、全対照に対する優位はP_Uで反証済み|
|P5 root/pair/first-label群|G6_NOTE Eqs.20–27、P5_CODE。power sumsを共有し群構築O(L²)算術|全child表を避ける具体構造。現group selectorはO(L²)になり得る。word別native価格や認証期待費用は未取得|
|support/basis/短窓cluster・q affine価格proxy|CC_CODEのcalibration→predict、RTE component構築と短回路compile|全候補長回路より軽い可能性はあるが、Tier0ではない。prefix/snapshot/compiler/cost scopeの移送を検査する必要|

native側のmetricは独立に扱う。D3は保証だけの変更で十分N、合成価格だけの変更で価格寄与を分離し、T/CX/1Qの選択が異なる保存結果を持つ。G6のdegree0はempty parentでありno-opではなく、C1ではchild rotationが残る。zero/rejectionとtechnical failureも区別し、全trial分母を保持する。

## H3：誤差と資源を結ぶときに足りない情報

- FR_AVAILABLEは粗い利用可能半径下界で有用性gateを通らず、真のrho/参照固有状態を使うmechanism結果だけが良かった。Oracleの利用で安価な予測を主張しない。
- BFは固定5-stage/family/budgetでfinite-task目的F固有の追加decision valueを得なかった。別geometryやfamilyを増やして同じ仮説を救済しない。
- BMのfloor＋internal/m²は同じnested列のcompact BCH再帰から得られる。相殺を取る前後のnorm policyでscoreが違っても、新しいoperatorや固有誤差情報ではない。
- PR2の保存signed bias分解は事後診断。正解signal/reference、N、Gを構成前の説明変数へ入れない。M2の予測にもheld-out shotが使われており、完全な事前総G予測ではない。
- D3の誤差保証改善は特定Hermitian contraction/Taylor taskの既存結果。同じ保証・native誤差・cost boundaryへ戻す必要があり、H4全PR信号へ自動移送しない。

これらから、現在の保存資産だけで「独立して安価に得た特徴から適格性/headroomを保証する新しい関係」が得られたとは言えない。

## H4：再検討範囲を狭める制約

|既存の根拠|避ける再探索|残る問い|
|---|---|---|
|R3 method-delta STOP|一般MFに特徴を渡すだけのcertified selectionの再命名|quantum固有のbias envelope/情報取得差が具体的にあるか|
|BM/Gaussian同値・方法差監査|既知compact/cache/Gaussianの再発明|既知baselineにも同じ最適化を認めた具体追加手順があるか|
|P_U analytic counterexample|Uが全closed対照を支配するという全称命題|より狭い前提で別の命題を置く必要が本当にあるか。未採択|
|G6固定辞書IS下界|同じ辞書/価格/残余幅でqだけを変えれば必ず勝てるという期待|辞書/合成/誤差を変えるなら、その取得費用と新機構を先に定義|
|Parent S4/D＋E|係数L1低下だけで選択、同じtoy配分の追加探索|native角度辞書・N・価格・全trial/準備費用を含める具体差|
|C1の正式pause|度数0・zeroなしの同じ短時間条件を自動増量|再開には既存pause条件を満たす具体機構と別承認が必要|

現段階の反証可能な結論は「law-level同値性をG不変性へ拡大しない」「上界/normalizer/momentの一因子だけで順位を決めない」という制約である。これらも既存結果の帰結で、新しい選択理論の成立を意味しない。
