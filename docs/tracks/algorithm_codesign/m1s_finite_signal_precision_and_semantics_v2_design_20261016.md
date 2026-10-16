# M1-S有限信号：意味論・条件付き誤差予算のv2技術設計

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

## v1からの変更と独立性の限界


各 j=G3/G6/G9 に対し、H_full はraw保存binary64 DF入力をexact dyadicとみなして構成した理想参照、H_tilde_j=HD_j+HR_saved_j は3方式共通の実際のsplit-targetである。
HDはraw DF prefix＋保存HR identity scalar、HR_savedは保存nonidentity Pauli lawのまま。identity移動・符号・samplingを調整して両targetを一致させない。

DeltaH_j=H_tilde_j−H_full。exact算術でfull、HD、HR、DeltaHのHermiticityを検査し、不成立ならtechnical STOP。
B_j=max_i sum_k (|Re DeltaH_ik|+|Im DeltaH_ik|) [Ha] は、Hermitian性により ||DeltaH||_2 <= ||DeltaH||_infinity <= B_j を満たす。
有限次元Hermitian両HのunitarityとDuhamel積分から

    ||exp(i T H_tilde_j)−exp(i T H_full)||_2 <= |T| B_j
    |z_tilde_ideal_j(T)−z_full(T)| <= eta_j(T)=|T| B_j.

Tの単位はHa^-1、etaと複素信号は無次元。保存stateの元norm²で割った同じ期待値なのでstateが厳密固有状態である必要はない。真の化学Hamiltonianへの誤差保証ではない。
exact整数・有理数実装の正しさも今回数値検証していない。将来実行でB/etaを初めて評価し、全3分割のT=0,1,4でexact numerator/denominatorを保存する。今回は数値代入0。

結果schemaでは各完了行に次を保存する。

|台帳|意味・解釈|
|---|---|
|numerical_rounding|方式信号・raw理想参照・その差の条件付き数値半径。signal精度2^-24のgateにetaを足さない|
|Hamiltonian_model_difference|B_jとeta_j(T)、単位、両target、モデルのみの位相上界・未解決理由|
|finite_step_algorithm_error|H_tilde基準の純アルゴリズム差の**大きさの区間のみ**。point_value=null、algorithm_only_error_measured=false|
|total_complex_discrepancy_to_raw_H|方式−raw理想参照の総複素差。純PF／乱択／有限stepアルゴリズム誤差とは呼ばない|
|phase_interpretation|TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H。既存phase_bias_radはこの意味だけ|

数値差の中心 d_hat と数値半径 r_num に対し、|d_hat|の条件付き下上界をl,uとすれば、純アルゴリズム差の大きさは [max(0,l−r_num−eta), u+r_num+eta] に含まれる。
これは三角不等式によるenclosureであり、モデル差の実ベクトルやアルゴリズム差を厳密に分離して測定した値ではない。z_tilde_idealは数学的に定義するだけで**評価しない**。将来one-shotにも追加のsplit ideal指数評価は含めない。
eta>0ならMODEL_TARGET_CONFOUNDEDを表示する（原因帰属の制限であり数値FAILや科学NO-GOではない）。eta=0では当該時刻のモデル差は0。モデル差の大きさに応じたgrid・threshold調整は行わない。
shared−ordinary/independentのvisibility差は同一H_tilde内の直接対照である。モデル差をその数値半径へ加えず、公平比較の意味を維持する。

モデルのみの位相上界は、raw信号の条件付きvisibility下界 L を用いる。L>1/4、L−eta>1/4、eta<L、かつ数値raw信号を中心とする半径 r_ref+eta のdiskがzero・負実軸branch cutから分離することが十分条件。
条件成立時のみ asin(eta/L) の上側評価＋既存library予算を保存し、対象はArg(z_tilde_ideal)とArg(z_full)の差に限る。判定にはdiskを覆う長方形を使い保守的にbranch接触を排除する。
条件不成立・ratioの上側丸めが1以上ならmodel_phase_status=UNDETERMINED、bound=null。unwrapしない。モデル位相上界はモデルの大きさであり、測定位相の数値半径2^-18 radを緩和する項ではない。
旧v5のfull-H/HD未保存によるUNDETERMINEDは維持し、新しい再構成で直接一致を確認したと主張しない。


新namespace m1s_finite_signal_v2 はv1のnumerics.py、word_reference.py、__init__.pyをbytes不変でコピーした。共有・ordinary・independent平均写像の本体、指数・Walsh・correctness設計は不変。旧独立referenceの15 helperもAST一致を検査する。
exact_input.targetsにはHD/HR/DeltaHのexact Hermiticity STOPを追加。pilotには数値参照計算前の全split/T台帳、各行の三分割会計、総bias名称、same-target対照注記を追加した。
新scalar区間はbinary64 absのrelative2uという既存の未認証仮定からexact Fractionで導き、上下端を有理数保存する。モデル位相のasinは既存の絶対2^-51仮定と上向きscalar予算を使用する。
IEEE RN、complex BLAS、math.fsumを使う継承経路、abs/hypot、trig/log、underflow/FTZなどの仮定は静的検査では実証できない。CONDITIONAL_ENGINEERING_NOT_CERTIFIEDのままである。
v2専用schema/authority/result namespace、SOURCE/hash/採択契約/別実施承認の順序guardを使う。採択契約だけでは計算しない。guard閉鎖、authorization/grant/marker0。mandatory STOP。
