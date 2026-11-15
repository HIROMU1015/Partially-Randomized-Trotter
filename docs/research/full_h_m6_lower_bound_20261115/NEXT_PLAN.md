# 次のm6限定検証案（未採択・未実行）

現在の正当化済み下界0はp7基準に届かない。新規compileが必須とはまだ言えない。まず以下Aを優先し、取得を検討する場合はBのように「取得後に何を片側評価するか」を先に固定する。対象はFull-H κ2・m6・X/Yのみ。他round、再fit、Q3は含めない。

## A：0-compileの証拠回収と全support不等式

1. D7の保存preparation hash `92325333…8aa100` に対応する完全descriptorを、元producerの保存済みarchive/receiptから回収・公開する。component順、符号・係数・確率、basis ID/hash/runtime operations、binary64角度・phase、K2 law、H/state/snapshotを含める。旧0e3989…の368成分表を代用しない。保存descriptorが存在しない場合は欠測としてSTOPし、prepare/contextやstateを新しく再生成しない。
2. 同じdescriptorと固定compiler SOURCEで、局所patternからpointwise報酬h(ω)を定め、**全合法外部continuation**を含めて `C_X(ω)+C_Y(ω)≥h(ω)` を示せるか調べる。basis共有・Z/phase合成、product/CZ、order2、outer/scalar/wrapper端を含む。RZを複数のpatternへ二重計上しない。
3. 不等式が成立した部分だけ、元ideal lawでE_I[h]を評価する有限scopeを別途固定する。全event列を列挙しない。`57 E_I[h]>593555352` なら当該条件で片側比較、弱い/未証明ならUNKNOWN。

Aの新規path/compileは0。元m6 QPYを見る場合も、観測した1 pathの構造診断までで、全supportへの証明の代用にしない。現段階ではprivate QPYを読んでいない。原本・他担当branchへの変更はしない。

**前進させる一つの証拠**は、上記のall-support生存不等式。完全descriptorの公開はその検証に必要な入力であり、それだけで下界が成立したとは認定しない。

## B：価格閾値の事象を使う有限取得案

これは**条件付きの統計的片側主張の提案**であり、無条件な全support証明ではない。現在のPCG64 joint law・独立性は未資格。price confidenceも未採択。前提が解決される前のsampling/compileは0件とする。

一つの共有pathについて `W=C_6X+C_6Y`、事前閾値 `H=11,000,000 RZ`、成功事象 `A={W≥H}` とする。各価格はfull measured wrapper・元new ZZ/compiler/P0。Wは非負なので `E_I[W]≥H P_I(A)`。価格の全support上限やvarianceを仮定せず、0/1の事象で評価できる。

### 事前に採択・解決が必要な条件

- 新price検定の有意水準 `α_price=1/20` を提案。RPE/Q3の失敗確率と別の判断であり、自動的に転用・消費しない。
- 実取得law Qについて、**当該事象の片側差** `P_Q(A)−P_I(A)≤δ_A` を正当化し、提案値 `δ_A≤1/200` を満たすこと。これは仮定する予算であって、今回測定/証明した値ではない。全path TVの同上界は十分条件だが必須ではなく、価格変動幅も不要。PR-DF F0・Full-Hの1event CDF誤差だけをPCG64 joint lawの証明にしない。
- 将来のpath間で独立なQ draw、又は各過去履歴の下で同じ成功確率上界が成立する条件を固定する。固定seed列・異なるseed名はその証明ではない。I/Qの定義・乱数モデルを明示して、条件付きの主張かどうかを採択する。既存samplerを無断変更しない。
- H・最大件数・成功/失敗/欠測規則、cohort・seed/source・compiler・metadata gateを**取得前に公開**。元D7の1 pathは設計の参考としてのみ用い、今回の新標本へ数えない。元の独立holdout roleは保全する。

研究上IID近似を採択しただけでは、実PCG64の無条件保証へ昇格しない。厳密なI母期待値への保証が目的なら、独立性/条件付き確率上界と事象law差を正当化する証拠が必要。証明できなければ条件付きモデルの主張に留めるか、取得前にSTOPする。

### 有限件数とサイズ証明

帰無仮説を `57 E_I[W]≤K`、K=593,555,352とする。すると

`P_Q(A)≤K/(57H)+δ_A≤12431049/13062500 = p_*`。

将来の61個の独立pathが全て成功する確率は、帰無仮説下で `≤p_*^61<1/20`。上記条件の下で61/61成功すれば、有意水準5%以下で帰無仮説を棄却する片側主張が可能である。これは新しいCIを保存データに当てたものではない。exact算術で `p_*^60≥1/20` も確認し、**この提案パラメータと全成功規則では61が最小**。

- 対象：固定m6、q64/R566/K2、同一physical ID、X/Y。
- 最大新規fresh path：**61**。各pathでX/Yを取得、最大新規full-wrapper compile：**122**。
- 1pathでもW<Hなら、この全成功証明は不成立。そこでfutility STOPしてよい。欠測・identity/実装gate不成立もUNKNOWN/STOP。失敗pathを入れ替えず、件数やH/α/δを事後変更しない。
- 早期の成功認定はせず、成功主張は61件完了時の一回だけ。失敗時は順位未判定であり、Full-Hが安いとも主張しない。
- 同じevent列が偶然再出現しても強制除去しない。lawを変えるためである。同一compile入力の完全一致が確認された場合のcache再利用と、独立なdrawの標本roleは区別する。
- D1の元実装・sampler/lowering・compile/cache・軸並列・phase/identity gateを再利用。元のindex0数値gateと全pathのparity/scalar/event/XY検査範囲を維持し、実行前に有限callerを固定する。新optimizer/fitは不要。

D1の保存m6は1 path/2 compileでwall1:26:15、GNU最大RSS16,338,552 KiB。これは61件の時間/同時RAMの上界ではない。後続path・worker数・cache効果を未測定で比例見積りしない。負荷が大きいためAの証明を先に検討し、Bの前提が未解決のまま試験的に122 compileを発注しない。資源へ恣意的な低い上限は設けず、共有環境への変更はしない。

## 実行前§10.5・STOP

同じpreparation hash・m6 physical/axis・compiler・ideal/Q law・事象Hの保存下界、進行中の全support診断又はprospective cohortを照合する。D7/P1や旧K4を新独立標本へ数えない。条件を満たす既存証明・取得があれば再利用し、不足分だけに限定する。

今回A/Bは未実行。AのSOURCE/入力回収、又は前提を解決したBの科学callに進むには、具体scopeの明示指示が必要。A不成立を理由にBへ自動fallbackしない。完了/不一致/欠測は保存してmandatory STOPし、m0修正・他round・fit・全G・winnerへ進まない。
