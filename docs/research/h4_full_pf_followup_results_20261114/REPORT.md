# Full PF追加5点のH_eff科学実行結果

承認された5点を一回評価し、全数値gateを通過した。既存の局所εtrot=.00079680005099520 Haに対し、S2/16h/7、Y4/16h/3、M5/16hは予算内、S2/8h/3とY4/32h/5は予算超過だった。保存M5/.8のENERGY_WINDOW整合検査もPASSした。これは局所spectral診断であり、全RPE RMSE・十分N・compiled総費用の認定ではない。

[HANDOFF](HANDOFF.md)／[5点CSV](CELL_SUMMARY.csv)／[65round metadata](ROUND_SCHEDULE.csv)／[C packet](C_HANDOFF.json)／[SOURCE索引](SOURCE_INDEX.json)／[raw結果](run01/result.json)／[人工テスト](TESTS.txt)。

## 固定SOURCE・対象

固定計画は `d48d8732ee7ef521aab5d67a0945439364f5a2f8`、科学caller SOURCEは `a3005d72349e9a91c3229215b189e6cf8bed0000`。既存A branchを継続した。保存Linear H4/1.00Å/STO-3G/tol-only1e−8 DF10、weighted-Hermitian completion、sector(2,2)/36次元、Full PF/prefix10、保存H/state/snapshot、元fragment順とscalar2.29310124732の一回会計を保持した。

hは保存binary64 .1のexact有理数、M12、P0、targetは元の保存DF-H、energy shift0。元prior/decoder/profileとεE=.00159360010199040、εqpe=εtrot=.00079680005099520 Ha/Minkowskiを変更していない。元Hの固有系はcommit69ee0ff8のSOURCE付き完全cacheで、新しい元H eighは0。入力receipt/snapshot/contextの3 SHA256は実行後も正本と一致した。

## 固有値シフトと構造cost

δEは元Hの最低energy groupに対し最大trace overlapを示すeffective groupのenergy差。全5点でそのgroupはeffective最低groupの0番で、非縮退だった。最低値を無条件に対応付けたものではない。元H最低値は保存固有系の−2.1663874486347643 Haで、保存solver proxyとの差と区別する。

| 公式・actual幅 | δE（Ha） | 局所εtrot screen | 元最低groupとのtrace overlap | 保存stateのeffective group重み | m12 qPF / signed S2 pieces |
|---|---:|---|---:|---:|---|
| S2・16h/7 ≈ .22857143 | +6.996149842586341e−4 | 予算内 | .9999979251020927 | .9999979251020923 | 1792 / 1792 |
| S2・8h/3 ≈ .26666667 | +9.528866685140080e−4 | 予算超過 | .9999961486023305 | .9999961486023301 | 1536 / 1536 |
| 標準Y4・16h/3 ≈ .53333333 | −5.907266327462857e−4 | 予算内 | .9999999386599634 | .9999999386599627 | 768 / 2304 |
| 標準Y4・32h/5 ≈ .64 | −1.200673137023767e−3 | 予算超過 | .9999997398078293 | .9999997398078289 | 640 / 1920 |
| M5 best・16h = 1.6 | −1.049216226967076e−5 | 予算内 | .9999998038354189 | .9999998038354186 | 256 / 2816 |

予算超過2点も数値的には正常に完了した有効な結果である。閾値緩和、別候補、追加探索、科学retryは行っていない。新しい境界fit・最適幅探索もしていない。

比較へ渡す3候補はS2/16h/7、Y4/16h/3、M5/16h。旧局所候補S2/.2、Y4/.4、M5/.8に対し、late qはそれぞれ12.5%、25%、50%減る。ただしこれは整数scheduleの構造比較で、同じNやq比例のcompiled RZを仮定していない。3候補のm12 piece数は1792/2304/2816であり、δEが最も小さいM5を総費用winnerとはしない。全13roundの価格・shot/riskとの接続が残る。

## M5の窓と固有状態対応

承認済みENERGY_WINDOW [-2.5,1.0] Ha、purpose=EFFECTIVE_SPECTRUM_WINDOWを使用した。Hamiltonian/scalar/decoder priorの変更ではない。分枝は保存solver energyを参照せず、指定窓から選択した。

まず旧commit3ae055faのM5/.8全スペクトルを1回検査した。同じ窓でのenergy表現差は最大4.440892098500626e−16 Ha、group重み差0、group indexとstate weightsは不変、保存数値gateも成立。旧raw/signal/固有系原本は変更していない。[検査結果](run01/SAVED_M5_08_WINDOW_CHECK.json)。

新M5/1.6では36固有モードのうち1モードにbranch integer −1、他35モードに0が付いた。selected cut clearance=.7400404439996042 rad、窓端clearance=.3984477904098114 radで、元guard1e−8 radを通過。元最低groupとの最大overlapと保存state支持は上表のとおり。指定窓のもとでの整合を数値確認したのであり、physical alias・厳密ground同定の資格falseは維持する。M5/3.2は未実行。

## 二時点照合・数値gate・early round

各新stepは初使用時点とm12を照合した。S2/16h/7はm4/q7とm12/q1792、S2/8h/3はm3/q3とm12/q1536、Y4/16h/3はm4/q3とm12/q768、Y4/32h/5はm5/q5とm12/q640、M5/16hはm4/q1とm12/q256。直接PFと全spectral再構成のcomplex mean各10を保存した。

最大mean差は2.8106753943936025e−12、最大power unitarity誤差は1.2706239085083123e−11で、元1e−9 gate内。state norm・Schur・再構成・元固有系cache residual等も元policy内だった。再正規化・unitary修復・fallbackは0。有限binary64の数値診断であり、rigorousなenclosureではない。

全13roundのactual幅とearly例外は固定PLANどおり。早期の旧9actual cellはSOURCE参照のみで、新しいstep/meanを取得していない。Y4/32h/5のm4はY4/16h/3と同一physical step・同一T/qなので今回のm4 meanを共有できる。65配置roundの表は全roundを新計算した記録ではなく、新meanは10 distinct physical roundだけである。

## 実行件数・テスト・残る資格

context load1、新PF step5、Schur評価5、直接/spectral mean各10、保存M5/.8スペクトル窓検査1。元H新eigh、新reference、旧保存signal再解析、normalizer/N/MC/価格/G、M5/3.2、追加幅、retryは全て0。実行時間7.923秒。[実行記録](run01/EXECUTION.json)。

新caller5＋既存plan4＋旧caller4の計13人工/metadata test PASS。旧callerへの技術差分はoptional branch引数だけで、未指定時のPRINCIPALは保持。PF式・直接expm・H_eff数値評価器は再実装していない。[caller差分](CALLER.md)。

ground/physical alias/rigorous numerical errorは未資格。rpe_bias上界・十分N・global RMSE・G・winnerはnullを保持する。新しい価格証拠も取得していない。今回の結果をC/Bへ渡し、次の価格・risk評価をどの有限範囲で行うかはユーザーが決定する。commit/push・remote照合後mandatory STOP。
