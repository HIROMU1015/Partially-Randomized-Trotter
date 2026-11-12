# 固定H4のS observableに限った状態縮約の十分条件

これはSOURCEと今回の有限局所検査からの構成的な論証である。独立科学レビュー、実physical kernelの実装検証、実平均S/Cの取得ではない。run02の`state_equivalence=UNKNOWN`は、同runが全continuationの証明・到達graphを構成しなかったことを示す。以下は同じ承認scope内でその局所結果から進めた追加解析であり、run02を上書きしない。

## 対象observableと状態

Sは`Accounting.finish()['structural_rz']`。`NativeCounter`のRZ countは単に入力RZ命令数で、depth/role/features/global_phaseは将来のS分岐へ戻らない。従って各wireのZ-foldが発行するRZ数をrewardとし、その和だけを保存すればよい。全Accounting reportやcompiled Cを同じ縮約で再現するとはしない。

元`basis`の遅延・literal inverse cancellationはfull operations/inverseで判断する。basis hashだけに置換しない。tail内ではactive basisを共有し、各applicationの非identity controlled primitiveを発行した直後はpending_basisが空になる。tail開始/終了と固定headでは、元のqueued basisを保持した別の決定論mapを使う。

今回9 basisの正逆18wordは、それぞれsystem全8wireに非RZを含む。従って一つを実際にemitすると、各wireの出口pending Zは入力値に依存せず、wordの最後の非RZ以後のRZ列の逐次binary64和になる。basisがliteral逆対として消える場合は、pending Zを変更しない。

全合法componentのsupportは1又は2wireの非identity。全local product/rotation wordについて、各wireは「非RZでresetし、その後の固定suffix値へ移る」か「完全にuntouched」のどちらかである。resetなしのZ加算wordはない。rotationの角度は各law/order/signごとに元float64値をそのまま使用する。

このためtail内のapplication境界で、system wire jのpending Zは次の有限集合V_jに入る。

1. 固定forward head終了時の値。
2. 9 basisのinverse wordの出口値。
3. touched applicationの出口値（None又は元CZ等のsuffix値）。

Noneと数値0、各binary64値は区別する。basis切替は(2)へ、applicationは(3)又は値の維持へ遷移するので、この集合は任意長の合法application列について帰納的に閉じる。basis emit中・application内の加算は元の順序で実行し、丸めやmodulo/非ゼロflagへ置換しない。9つのbasisの実operations・qubit順・inverseを区別した有限stateとして残す。

## phaseとcontrol

identity extraction後、K2 eventの相対phase加算はorder2の+πと負productの+πだけである。order0はrotationのみでphase0。固定R≤128では全和は有限の非負binary64値で、0であることと「order2がまだ一度もない」ことが同値。単なるmod 2π同一視ではない。

各rotation終了はcontrolをresetして出口pending ZをNoneにする。tail終了時、phaseが正なら元`gate(p)`がqueued forward basisをemitしてからcontrolへphaseを置く。phase0ならqueued basisはそのままである。この分岐はsystemの価格にも影響するため、**order2-seen flagを全wire marginalに含める**。

その後のreverse headは、control RZを発行する前にcontrolへ非RZを発行する。basis操作もcontrolに触れない。このbarrierでtail phaseのpending Zは、正ならRZ一つ、0なら発行なしとなり、その正確な大きさは以後のSに影響しない。途中で別のscalar/wrapperを挿入する任意suffixにはこの論証を適用しない。認めるのは元の合法tail→reverse headだけである。

## outer・terminalと全continuation

固定S2/prefix1のforward最後のDF diagonalはsystem全wireをresetする。従ってtail entryのsystem Zは固定である。reverseのDF/one-body directional diagonalはいずれも全9wireをresetする。最後のone-body directional diagonal後のZとqueued basisは、tail履歴から独立した固定値になる。よって次outerへ未知の連続角履歴は持ち越さない。

scalar −cTはq outer終了後の一回、Yのsdg、H/measure、finishも元順で一回だけ。これらは固定outer出口の値とqueued basisへ適用する決定論suffixであり、新たな無限履歴を作らない。q1/64のfirst/steady outer entryとaxis suffixを同一視しない。途中finishは認めない。

以上により、同じfull active basis、同じphase flag、同じwire jの正確なpending Zを持つ二つの合法prefixは、そのwireに関して、任意の合法continuationに同じ遷移・RZ増分・terminal rewardを返す。他wireのZはbasis/phase分岐を変えない。全Sの期待値は各wire rewardの期待値の和なので、**wire間の統計的独立性を仮定せず**、共有basis/phaseを持つ各wireのmarginal kernelで扱える。全wireの直積stateやcovarianceを同じ小ささで表せるとは言わない。

tail application境界の保守的enclosureは各system wireで、最大(9 basis＋entry)×|V_j|×2 phase flag。entry/close/head/outer/wrapperの有限mapは別に必要で、これは到達state数の実測でも完成したkernelサイズでもない。

## 判定と残る条件

**確認したのは、固定supportと2lawの元S observableについて、有限なwire-marginal価格状態を構成する上記十分条件である。** 有限float registerという一般論ではなく、元wordのreset/untouchedと決定論boundaryによる帰納である。

一方、実physical transition/reward行列・entry/close/suffix mapはまだ構成しておらず、元Accountingとの局所/境界parity、probability massとbinary64伝播誤差を検証していない。効率的な実平均S評価器の成立はUNKNOWNのまま。今回のgeneric `projected_step`とMeanKernel接続は人工相関eventだけで確認する。実H4 meanは計算しない。

ideal有限lawなら、event kernelはorder0のrotationと、order2のphase-positive→product→product→rotationの混合として、元componentのIID確率から組める候補である。これは全324³ order2組合せ/全event列を列挙する必要を減らす設計で、今回実matrixや期待値を計算した結果ではない。実PCG64/CDF lawとideal lawの距離・共有covariance・compiled残差は未資格を維持する。
