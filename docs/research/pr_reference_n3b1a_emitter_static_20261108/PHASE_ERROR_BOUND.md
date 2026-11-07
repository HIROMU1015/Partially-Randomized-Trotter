# N3-B1a：W位相とemitter誤差の記号的十分上界

2026-11-08 JST。基準`9c60c43135666a9ab5497a43a4b1695ab371650f`。[SOURCE監査](EMITTER_SOURCE_AUDIT.md)、[REPORT](REPORT.md)。以下はCodexの**静的な代数・IEEE条件付き導出**であり、独立科学レビュー、位相ループの実行、区間再評価、Operator診断の結果ではない。

## 1. 距離と比較対象

exact wordのW=exp(i*pi/4)Iを保持し、emitterの保存位相との差には**真の2piを法とする円周距離**を用いる。これは角度keyを2piで同一視する操作ではない。位相wrapの前後を単純な実数差で比較すると、微小な誤差を約2piと数えることがある。

同じunitary Vに対して

\[
\|e^{i\delta}V-V\|_2=|e^{i\delta}-1|\le|\delta|,
\qquad \|e^{i\delta}V-V\|_F=\sqrt2|e^{i\delta}-1|
\]

である。後者のsqrt2は**1-qubit wordの2×2行列**についての式。joint全体のFrobenius normへ無条件に流用しない。composeに伴う全体scalarの増分は、置換中の1-qubit wordにscalarを付ける形で帰属させられる。その局所Frobenius上界から、埋込み後はspectral normのtelescopingへ接続する。

## 2. 適用条件C_phi

本節の短い上界は、次をすべて条件とする。

1. 既存のIEEE binary64 round-to-nearestモデル。有限の通常演算、correctな有界remainder、fast-math等による意味変更なし。
2. CPython math.piとQiskit Rust側のPIが同じ保存binary64定数Pi0である。`P=Pi0/4`、`tau=2*Pi0=8P`。Qiskitの数値setterは`rem_euclid(tau)`を使う。
3. local circuitはglobal_phase=0から始め、H/S/T/Tdag/Xのnamed gateを元の順序で追加する。これらを別の浮動角gateへdecompose／再compileしない。named Clifford+T gateは既存研究のideal論理gateとして解釈する。
4. Wの処理は既存lower_nativeの1文字ごとの加算か、それと同じ位相を持つことを別途証明した処理。任意のfloat(K)*Pへ書き換えない。
5. composeのglobal phaseは数値であり、同setter由来の範囲[0,tau]にある。negativeには連結済みpositive word全体のinverseを1回使う。free ParameterExpression／control-flow／後置controlは本証明に含めない。
6. 実行するphase操作・QPY再読込の経路と回数を固定する。任意回数の後続処理まで保証しない。

Pi0の保存keyはCT-0の`0x1.921fb54442d18p+1`。従ってPはそのexact binary scalingである`0x1.921fb54442d18p-1`。CPythonの定数定義とRustのPI定義は[SOURCE監査](EMITTER_SOURCE_AUDIT.md)に対応付けた。実runtimeのbit／rounding条件を新しい数値検査で確認したわけではなく、その適合確認は後続の実装検査に残す。

以下のremainder評価は、有界な本入力に対する正しいf64 remainderと、Rust SOURCEに記載された負剰余への加算を条件とする。SOURCE参照だけから、wheelの機械語／C runtimeまでを形式検証したことにはしない。

## 3. W加算の8状態閉包

Pのbinary64 significandは、上記hex末尾の3つのzero bitを除くと50 bitである。k=0…8についてkPは高々53 significant bitなので、すべて正確にbinary64へ表現できる。したがってlocal phaseがkPなら、次のP加算もk+1<=8の範囲で正確である。

0…7のkPはtau未満なので数値setterのremainderはそのまま返り、8P=tauでは0へ戻る。よって、Wが任意の有限K個あってもlocalに保存される位相は

\[
\phi_K=(K\bmod8)P
\]

となる。H/S/T/Xのappendはこのglobal phaseを変えない。**この固定定数・演算経路ではK回の丸め誤差を単純にK倍する必要がない。** 未取得W数を仮定したのではなく、状態が閉じる条件を証明したものである。

負角ではglobal phaseを否定して同setterへ渡す。kPの否定とtau加算も上記有限lattice内で正確に表現でき、(-k mod8)Pへ移る。gate列は逆順／各gate inverseなので、exact wordのactual adjointへ対応する。ただし、float位相のQiskit circuit行列がpositive circuitの厳密なcomplex adjointと完全一致するとは置かない。binary tauと真の2piの差は次節で数える。

u=2^-53、Pi0がpiの通常の正しく丸められた定数である条件の下で、pi<4から

\[
d_P:=|P-\pi/4|<u,\qquad d_\tau:=|\tau-2\pi|<8u.
\]

従ってpositive／negativeのlocal word位相誤差は円周距離で7u未満、局所Frobeniusでは14u未満である。zero local wordは恒等だが、周囲のphase正規化の誤差まで0とは置かない。

## 4. compose・inverse・再読込

localの位相kPを既存carrier位相phiへcomposeすると、SOURCEは`RN(phi+kP)`をsetterへ渡す。0<=phi<=tau、k<=7なので加算丸めは15u未満。正の和は2tau未満で、remainderは0又は1回のtau除去。除去する差はSterbenzの範囲にあり、表現可能な差へのremainderがSOURCEの意味どおり返る条件で追加の丸めを要しない。真の2piとの周期差は別にd_tauを数える。

\[
|\delta_{phase}|<7u+15u+8u=30u,
\qquad e_{F,step}<64u.
\]

これはlocal Wの定数差と、そのcarrierへのcompose増分を合わせた保守的上界。先行gateから既に累積した誤差の全量を、このvacuum wordへ付け替えた式ではない。

二つの一般のcarrier位相が[0,tau]にあるcomposeも、和が2tau以下、除去周期数が高々2であることから、既存誤差に追加される局所scalarのFrobenius上界を64u未満で覆える。これは既にある両carrierの誤差を消す主張ではない。

一般のcarrier全体をinverseする場合は、-phiのremainderとtau加算により、新しいFrobenius増分を32u未満で覆う条件を使える。negativeのごく小さい位相で結果がtauに丸められる境界を除外せず、円周距離で評価する。Rustのrem_euclidにもこの境界の留保がある。

QPYのfloat payload自体はdoubleのpack/unpackで保存されるが、読込時のQuantumCircuit constructorはsetterを再度通る。保存phi=tauが0へ移る場合もあるため、payloadのbit保存だけでoperator誤差0とはしない。今回の数値phase経路で、再設定／再読込1回の追加Frobenius上界を16u未満で覆う条件を使う。QASM／JSONや任意の再compileへ自動移送しない。

## 5. 固定43-word構成への十分条件

推奨する最小経路は、固定順の43基本wordのうち使用するものを**1つのwritten wordとして渡し、既存lower_nativeで1回emit**する形。既存emitterに新しいphase機構を入れる必要は、このSOURCE証明だけからは認めない。

別の明示的なcarrier経路を含めて保守的に覆うなら、次の操作bookkeepingに限れる。

- 基本word carrierは高々43個。各local位相とjoinの増分を64uで覆う。
- 最後のjoint carrierへのjoinは1回、追加64u。
- 連結済みword全体のinverseは高々1回、追加32u。
- 初期phase再設定／QPY再読込は合計高々2回、追加各16u。

この経路なら

\[
e_{emit}<\bigl[(43+1)64+32+2\cdot16\bigr]u
<2^{12}u=2^{-41}.
\]

zero／負角／上端近傍でも、localのresidueは同じ8状態に属し、bit分解は43 chunk以下であるため、この条件式の対象に含まれる。近似wordの並べ替えは認めず、固定指数順のwritten productをそのままemitする。これは端点が実samplingで到達したという主張ではない。

N3-B0の保存外向き下端は5e-13より大きい。また2^10>10^3から2^41>2*10^12なので

\[
2^{-41}<\frac1{2\cdot10^{12}}<R_{emit}^{lo}.
\]

従ってC_phiと上の有限経路を**実際のSOURCEへ対応させられれば**、残余条件を満たす記号的十分上界が得られる。この比較は保存済み下端と整数不等式による静的導出で、N3-B0残余や新しい区間値を数値再計算したものではない。任意のphase操作回数や未実装pipelineをPASSにしていない。

## 6. 閉包条件が使えない場合

別定数／別emitter／一般の浮動phaseがlocal loopへ入る経路には、8状態証明を無条件に使わない。W数K、compose数C、inverse数I、再設定数S、各演算の安全な誤差rhoを用い、例えば

\[
e_{emit}\le\sqrt2\{K(d_P+\rho_++\rho_{rem}+d_\tau)
+C\rho_{compose}+I\rho_{inverse}+S\rho_{reload}\}
\]

という一般の保守的条件へ戻す。この場合は実word metadata／具体的phase経路が必要。Kや長さを取得前に代入しない。20,000文字の取得後拒否条件を未取得全wordの上界へ使わない。

## 7. 保証の限界

ここで小さくできるのは**数学的wordと具象化の差**。基本wordのRZ近似誤差と量子化誤差はN3-B0予算に含めたまま。上のe_emitをその内部へ一度だけ加える。他の既存C0 wordについても、実guardとemitter増分を合わせて各局所許容へ入ることの確認が別途要る。

浮動Operatorの行列積誤差、物理noise、gateを浮動角へ再分解する誤差は、本証明のnamed ideal Clifford+T gate意味論とは異なる。CT-1/CT-4Bの有限診断を任意wordの証明へ格上げしない。全supportのword取得／実装適合、全CT RPE成功、総Tは未資格のままである。
