# M1-S有限信号：意味論・条件付き誤差予算の技術設計

本書は未採択SOURCE候補の設計説明。独立数学的受理・platform認証・新数値結果ではない。**mandatory STOP**。

## 平均演算の意味論

T1b原本§2/§10のD Phi^r(DXD)D、ordinary O、independent VXVを同じ相対時間・独立label総数Rへ合わせる。
P_jの符号はhに残し、twirl P_j X P_jでは二回現れるので打ち消す。負枝はlabelの同順sign reversal。
角thetaはshared lambda*delta/(2R)、ordinary/independent lambda*delta/R。independentの各枝はR/2。
XからXへの作用は同符号の左右積であり、PXPやDXDをunitary channelのdagger配置に変えない。

## 小さい行列のままtwirlを適用する

Pauliを(x,z) bitmaskで表すと (PXP)[u,v]=(-1)^(z dot(u xor v)) X[u xor x,v xor x]。
Y[d,u]=X[u,u xor d]へ並べ替えれば、twirlはu方向のXOR convolutionとなる。
正のweights p[x,z]をz方向Walsh変換、転置してx方向Walsh変換し、spectrum[d,k]を一度だけ用意する。
Yをu方向Walsh変換しspectrumを掛けて逆変換（同じ変換を1/256で割る）すればtwirlを得る。
保存するのは256×256の行列・spectrum・indicesだけであり、65536×65536 superoperatorを構成しない。
future correctnessでは別のliteral I/X/Y/Z tensorとmonomial matrix actionを使う。これは演算経路の照合であって別hardwareの独立証明ではない。

## 条件付きFrobenius台帳

Ballはmatrix_hat、Frobenius誤差radius e、数学的targetのoperator norm上界bを持つ。
行列積ABの半径候補は

    e_AB = b_A e_B + b_B e_A + e_A e_B
           + gamma_(8d+32) ||A_hat||F ||B_hat||F.

exact targetがunitaryなDやcoherent contractionの平均写像ではb<=1を使い、反復ごとにFrobenius norm16を不用意に増幅しない。
和・scalar積にはoperand半径・scalar値のexact-to-float差・gammaモデルを加える。正のbudget式は64演算未満に512u relative headroomとnextafterを付ける。
norm reductionはgamma_(8*n_cells+64)でinflateし、複素abs/hypotはrelative2uという条件を採用する候補。
twirlのWalsh2軸によるspectrum誤差はprobability conversion L1差＋gamma32、適用側は2回Walshと積/scaleをgamma128で会計する。
これらはbinary64 round-to-nearest、想定BLAS演算モデル、unaccounted overflow/underflow/FTZなし等を仮定したengineeringモデルであり、検証済みcertificateではない。
観測subnormal/nonfinite・exact Hermiticity不成立はtechnical STOP。ゼロを生むunderflow等をruntime観測だけで網羅したとは主張しない。

## 行列指数とscalar library

Hermitian Hの数学的exp(i t H)はunitary。scaled generatorのoperator boundをb<=1/8へ下げ、degree24のTaylorを評価し、その後squaringする。
解析的tail boundはFrobeniusで

    sqrt(d) * (8/7) * (1/8)^25 / 25!.

sqrt(d)は整数の上側値、各squaringで入力半径・matrix product丸めを会計する。これはTaylor打切りの項でありBLAS丸めとは別。
sin/cosはexact引数をbinary64へ丸めた差＋絶対2^-51、受理域のasin/atan2/logも絶対2^-51というlibrary仮定を候補とする。これらの実環境でのcoverageは未認証。
旧v5の17精度条件を新しい有限step結果へそのまま保証として転用せず、この新しいsignal budgetを別採択対象とする。
exact入力行列→binary64変換は各entryのexact差から、16*max row L1というFrobenius上界を作り最初のBallに入れる。
signal期待値は保存stateのexact norm²で割り、matvec/dot/divisionの丸めを追加する。定義上のstate誤差や真の化学Hへの誤差保証ではない。

## 比較の誤差と限界

visibilityはsignal球のradiusにabs丸めを加え、差には双方のradiusを足す。主信号の複素誤差とideal stateでないことによる参照visibilityも保存する。
phaseはq=z conj(z_ref)の球がzero/negative-real branch cut/floorへ接触しない場合だけ主値を出す。角誤差はasin(e_q/abs(q))＋library予算、unwrapしない。
precision不足は不一致・利益なしと同義ではなくUNDETERMINED/technical STOPとして保存する。G9のような極小tailをlambdaで時間rescaleして救済せず、lambda²の絶対先導尺度を並記する。
相対ratio・科学GO・single-rate domain・native/RPE資源を認定しない。採択はユーザー、科学実行はさらに別の明示承認。**mandatory STOP**。
