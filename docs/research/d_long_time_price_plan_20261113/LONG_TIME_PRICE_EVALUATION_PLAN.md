# 長時間の価格評価方式・平均資格

2026-11-13 JST。[REPORT](REPORT.md)に固定物理・SOURCE・実装調査を示す。本書は理論設計で、実価格への代入、平均・区間・fit計算は行っていない。

## quantityとlawを先に固定する

| quantity | 定義 | 言えること／言えないこと |
|---|---|---|
| logical event数 | signed allocationを全outerへ展開した位置数 | compile済gate数・時間・RAMの測定でない |
| S_a(ω) | 元Accountingを同event path、scalar一回、測定付きX/Yへ適用したRZ数 | compiler前の別functional。Cの上下界でない |
| C_a(ω) | 固定native/control/ZZでfull measured wrapperを固定compilerへ通したRZ数 | そのpath・軸の整数価格。標本分散/母平均ではない |
| finite-cohort mean | 明示した正常終了path集合上の算術平均 | development/replay/forcedを混ぜず、全law meanへ自動昇格しない |
| μ_a=E_P C_a | 明示したtarget path law Pの期待値 | law・統計・数値/モデルの必要根拠を含めて資格を判定 |
| mean interval | μを覆う証拠付き区間 | path分布の範囲・標本SD・予測区間とは別。複数セルの利用には同時被覆が必要 |

PはA signalで使うideal finite-K lawか、実装sampler law QかをB/Aと一致させる。PRNG固定seedからの有限列はそれ自体IID全lawの証明ではない。normal sampling、order条件付きforced sampling、保存path replayはroleを明示する。replayは同値性検査でありfresh nに数えない。

## A：選択pathの直接full compile

既存 `request_for → NativeRuntime.context/sample → Adapter.acquire` を唯一の現行参照経路とする。実行には別途有限承認を要する。canonical configuration/physical ID、binary64 signed time、q/R/K/piece順、prefix別head/tail/λ/scalar、measurement/axes、全SOURCEを固定する。wrapper fingerprintは**transpile前回路全体**を同定し、compiled価格の同値性にはcompiler profileも必要である。

各成功pathのCは固定compilerの操作的定義に対し直接得られる。これは理想unitaryへの数学的厳密性、任意環境でのbitwise再現、全law平均資格を証明するものではない。既存の人工同値性・SOURCE/runtime検査を再利用し、tolerance/compile設定を速度のために変えない。新LD2の実preparation適合性は後続の許可範囲内で初めて確認する。

現経路はfull native evolution、選択軸wrapper、transpiled circuit/全DAGを要する。L入力に少なくともO(L)の生成/走査/保持があるが、全passの実時間・peak記憶をO(L)と保証していない。最大適用規模はUNKNOWN。m6実績からm12のtime/RSSを線形に予報しない。両軸は同pathを共有して順次処理し、独立pathは許された範囲でprocess並列化する。

preparation/component/basis・同じpiece角度のhead記述、共有event path、Sの共通本体、call内fingerprint intern、厳密に同じwrapper/同compilerの既存cacheを再利用する。異なるpathの同event数、同じunitaryだけ、角度を落としたkey、X/Yの有限観測同値をcache共有の根拠にしない。旧finite cohortのnをcache hitで増やさない。

## B：構造的・解析的な評価

Sの元AccountingとCの有限状態reward kernelは、証明されたstate/境界/lawの範囲でSを評価できる。既存LD1全26 E[S]は元構成・ideal law・数値enclosureに限定し、新canonicalとのbridgeを確認して使う。LD2では値の流用をせず、Cがprefix別kernel/closure・law・scalarを確認する必要がある。これらの新科学評価は今回未実行。

compiled Cを解析的に計数できるのは、(i)固定compilerの出力wordと同じstreamを得る同値性証明がある、又は(ii)compilerに対して十分な状態・interface閉包を証明し、その状態遷移と数値簡約を再現できる場合である。既存native emitter/ZSXX/CX bridgeは部品として再利用可能だが、別CDF8/P0限定の証拠を新DF10/K2/opt1へ移送できない。unbounded frontierを残す既存bridgeは最悪O(L²)時間/O(L)記憶で、大規模の解決を保証しない。

元Sのbasis相殺とwire別pending-Z閉包は、Qiskitの1q行列再合成・CX逆対消去・fixed-point loopの閉包ではない。Qiskit opt1はRZ数を最小化せず、1q gate数を減らしながらRZを増やす人工反例が既にある。離れたCX相殺が1q runを連結する反例もある。したがって局所blockの価格和、raw RZ数、S、native instruction上限をCの厳密な上下界にしない。

安い非自明なcompiled上界は研究候補だが、translation/UnitarySynthesis/1q再合成と全反復の増加を囲む証明が必要。現時点でその証明はなく、普遍的な非負性による下界0以外の未取得情報はUNKNOWNである。巨大raw size上界を仮に証明しても、比較やrare平均を実用精度で囲めるかを別に問う。

## C：残差・モデルを使う場合

同lawでSの期待値が分かれば

\[
\mu_a=E_P S_a+E_P D_a,\quad D_a=C_a-S_a.
\]

直接paired residual法のcompile費は直接C法とほぼ同じ。分散低減は相関次第で、必ず改善するわけではない。適合したSの数値区間とDのvalid mean intervalをMinkowski和で結合し、負側端点は非負性で0へ切ることはできる。S区間をcompiled区間と呼ばない。

target Pとsampler Qが異なるときは、少なくとも次のlaw項が残る。

\[
 E_P C=E_P S+E_Q D+(E_P D-E_Q D),\qquad
 E_Q C=E_P S+E_Q D+(E_Q S-E_P S).
\]

最後の項をSEやfit残差で代用しない。同じQで直接C平均を取ればP/Q混合は避けられるが、A/BがPのsignalを使う場合のjoint task qualificationは別に残る。CDF丸めの距離だけでPRNG独立性は証明されない。TV距離×price oscillationによる補正は、距離とC又はDの範囲がともに証明された場合だけ使える。

### outer-windowの条件

target candidateのR、piece/time/angle、law、head/tailとcapsを固定し、n outerだけを同じtarget scalar p(−scalar T)一回・full X/Yで囲った残差の平均をμ_nとする。μ_0も同じcapsの空windowとする。window lawの定常性がある場合、δ_1=μ_1−μ_0、δ_k=μ_k−2μ_(k−1)+μ_(k−2)（k≥2）から

\[
 E D_q=\mu_0+\sum_{k=1}^q(q-k+1)\delta_k,
 \quad d_q^{(h)}=(q-h+1)\mu_h-(q-h)\mu_{h-1},
\]

\[
 |E D_q-d_q^{(h)}|\le\sum_{k>h}(q-k+1)|\delta_k|
\]

となる。これは厳密恒等式と条件付きの誤差式であり、tailが小さい証明ではない。小さいδ_3、Sのouter出口、有限個のholdout合格から残り全tailを0にしない。target scalarをwindow時間へ縮めたりouterごとに配らない。windowを本来のcandidate価格recordとして登録しない。

使用条件は、対象lawにおける長距離compiler影響のtail/有限閉包の証明、又は**使用する長時間domainそのもの**の独立full参照と妥当なmodel-discrepancy契約である。後者でも一pathの誤差はmean biasの区間ではなく、m12の合格はm10/11又は他prefixを自動保証しない。unmeasured domainにuniform envelopeを主張するには追加の理論又は明示された統計的仮定が要る。最大domain参照もtail根拠も取れなければm12はUNKNOWN。

coupled prefixのstencil Z=(q−h+1)F_h−(q−h)F_(h−1)の分散には共分散とq増幅がある。小回路化だけで総費用を節約できるとは限らない。将来、同じMC幅に対するvariance×実測path費用を直接D法と比較し、tail/model誤差の予算も含めて採否を決める。今回その実数値は求めない。

### 少数feature・その他のmodel

C37cbcの単純Z/ZZ/order/basis-transition特徴とMeanKernelを優先し、取得困難なcompiler状態の新kernelを先行実装しない。q/R・prefix・K・signed allocation・角度/小角・scalar/wrapper/lawごとに適用域を記録する。feature meanが同じでもcompiler出力平均が同じとは限らない。

独立calibrationとholdoutを分け、モデル選択後のholdout再利用で検証資格を作らない。過去P1/P2はdevelopment evidenceで、長時間独立holdoutではない。予測区間には係数共分散、共有校正、target-domain discrepancy、law/rare、数値誤差を含める。高いR²・短域の残差SD・経験的最大誤差だけを長域のbias boundにしない。

もしmodel期待値を同lawで安く正しく求められるなら、独立に固定したmodelに**target-domainの直接paired residual**を足すmodel-assisted control variateも候補になる。これは外挿biasの放置を避けるが、targetでのdirect compileとmean interval根拠はなお必要である。新fit/実装は今回行わない。

## 平均と不確かさの資格

| 成分 | 必要な根拠 | 不足時 |
|---|---|---|
| MC mean | 明示したpath ensembleと独立性/依存構造、全pathの成功/欠測、interval法の適用条件 | finite-cohort診断のみ |
| rare | normal法の有効全law CIに含まれるか、未観測stratum確率とconditional price影響の囲い | 0代入しない。未資格 |
| sampler law | idealと実samplerのbinding、距離/期待値差又は対象law変更を含むA/B契約 | PのE[S]とQのDを無補正で足さない |
| compiler/systematic | 同SOURCE/control/wrapper/lowering/profile、canonical入力、既存同値性検査。近似counterはoracleとの誤差根拠 | 別quantity/UNKNOWN |
| model/window | 使用domainでのbias/tail、校正誤差・外挿・共有dependence | model varianceだけのCIを出さない |
| numerical S / coefficients | 元enclosureの条件とround/candidate適合性、binary64順序 | 未検証精度を0扱いしない |
| X/Y | 同pathの二応答。標本数はnで2nでない | 軸別取得、共分散unknownを0扱いしない |
| 同時被覆 | 全使用候補/round/axes・選択/停止規則を覆う契約 | 点wise CIでwinnerを認定しない |

normal sampleにorder2が出なくても、その確率・conditional価格は残る。全law範囲を使うvalid CIならrareは既に含まれ、別に二重加算しない。forced層別法が必要ならnormal meanとpoolせず、元のjoint stratum確率・importance weight・各層誤差・未取得層を明示した別有限scopeとする。今回は追加forced samplingを提案の既定にしない。

有限分散だけを観測したsmall-nのt/normal/bootstrap幅は、未知rare tailを含む分布に対する保証ではない。IIDかつ真の範囲[A,B]が既知ならHoeffding等が選択可能だが、現compilerの有用な全path範囲は未取得。分布仮定付き近似CIを採るなら、その主張水準をユーザー/Bが選び、rigorous/full-law qualificationと区別する。範囲も妥当な分布仮定もなければfinite-cohort mean/SEの診断までに留める。

価格のconfidence/許容幅は今回は未採択。後続ではBの比較差/許容regret・N集合に応じてMC、law、model、numericalの幅を分け、未知系統項をn増加で解決したことにしない。α_priceはRPE物理測定のrisk予算とは別に記録し、両方の保証を主張するなら全体failure budgetとの関係を明示する。固定n CIを見て無制限に追加せず、有限batch family＋同時budget又は有効な逐次区間を使う。

## 技術点と正式比較点の選び方

技術点は最大使用q/R・prefixの適合性、native/hash/compiler peak、modelの長距離反例を問う。m12-LD2からの一path診断は候補になるが、LD1のdomainや平均を代行しない。m7→m8→m9のladderは既定にしない。

正式価格点はBの同target/decoder/claimで適格な全N又は集合𝒩、競合候補・比較margin、必要な同時被覆familyで決める。同時に0≤L_j≤μ_j≤U_jが成立するなら、固定非負NではG∈[ΣN_jL_j,ΣN_jU_j]。独立性は不要である。未知上端は+∞として残し、Sやmodelで埋めない。

候補xの適格な部分セル下界だけでも、実行可能incumbent bの全正重みセルを覆う有限上界より大きければxを除外できる。bの正重み未知セルが残れば有限上界は作れない。未知priceが比較差で消えるのは、同じlaw/profileの**同一価格quantityと同じN係数**を共有する証拠がある場合だけ。LD1/LD2のevent数の類似では消えない。

精密化の目安は比較係数の絶対値×同時価格幅を取得費用で割ったものとし、nominal proxyから新数値を作らない。N集合なら全許容Nを覆う最悪影響で判断する。未資格Nを集合の端点にしない。m10〜m12の構造集中だけでnや2%等の精度を割り当てない。未資格価格はBへpendingとして渡し、正式G/winnerはUNKNOWNを維持する。
