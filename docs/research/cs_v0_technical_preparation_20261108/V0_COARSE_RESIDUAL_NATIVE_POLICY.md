# C-S V0：単一coarse・finite residual・strict native政策案

2026-11-08 JST。**一組の提案、未採択。実H4への適用/合成/価格0。**

## Coarse候補を一つに固定

`Angle(π成分=a, radian成分=b)` に対して、π成分aをそのまま保持し、bだけを最寄りの `2^-8 rad` へ丸める。丁度中間は絶対値を大きくする側。coarseは `Angle(a,k/256)`、residualはexact差 `Angle(0,b−k/256)`。

D/R共通、S0/S1/S2共通表、1slot1候補、価格・signal不使用。πを数値化せず、sign・2πを捨てず、`|θ_res|≤2^-9 rad`。既存PAIのπ/4 notchはchannel用であり、その選択器は使わない。既存にこの有理radian丸めAPIがなかったため、小さいexact規則だけ追加した。

粗角が0、residualが0、通常角とkeyが重複する場合も、その結果を受け入れ、別coarse/slot/格子への探索で救済しない。合成後に安いcoarseを選ぶ政策ではない。有理dyadic角だからTが安いとは仮定しない。

## Finite residual：r_res=1, K_res=2（第一候補）

`R_P(θ_res)=exp(−i σ x P)`、`x=|θ_res|/2≤2^-10`、`σ=sign θ_res`、`P²=I`。既存CanonicalGeneratorのordinary degree3（K_res2のpaired n=0,2）、p=(1)、root_bits256、proposal_bits160、既存eta=rho=10^-12をそのまま使う。短時間domainを全政策で満たし、二つのorderでfinite/normalization/価格接続を識別できるため、V0ではこの一組を提案する。Parentや新law探索ではない。

- event `V_n = i^n exp(−i σ atan(x/(n+1))P) P^n`。実装は既存reduce_wordとoperator_targetを使い、n=2の `i²=−1` を保持する。
- 理想係数 `a_n=sqrt((x^n/n!)²+(x^(n+1)/(n+1)!)²)`。
- 実有限係数c_nは既存root intervalのmidpoint。実proposal q_nは既存dyadic law。`w_n=c_n/q_n`、`Q_res=sum q_n w_n V_n=sum c_n V_n`。
- 実normalization相当の係数和 `B_d=sum c_n`、corrected half-range `max |w_n|`、二次moment `sum q_n w_n²` はevent_statisticsから保存する。B_dとrangeを同一視しない。未補正meanの減衰を重ねて課金しない。
- 正のeventが有限bit格子で取得不能、coefficient precision不十分なら既存generatorがSTOP。bit精度/r/Kを増やして救済しない。
- residual=0はexact I、B=range=1、追加event/keyなし。異なる非zero residualを勝手にzeroへclipしない。

既存Taylor展開から、絶対tailを

`b_trunc ≤ x^4 / [24(1−x/5)]`

で上から抑える。係数midpoint誤差 `b_coef ≤ Σ(hi_n−lo_n)/2` を別に加え、`b_finite=b_trunc+b_coef`。これは単一involutionの有限operatorに対する既知Taylor boundで、新しい分子理論・経験的allowanceではない。人工入力で包含を確認し、H4値は未取得。政策domainではfinite上界は候補cap `10^-12` 内に収まるが、実入力でも必ず保存値を検査する。

**独立性**：V0は二eventを決定論的に列挙し、元N1 pathもresidualもsampleしない。元Qとの確率積・normalization積・moment積を計算しない。将来V1ではpath依存角の条件付きkernel `q_res(e|ω,site)` とfresh独立drawを仕様化し、jointを `Q_original(ω) × Π q_res(e|ω,site)` とする必要がある。現SOURCEは独立性を認証しておらず、cache共有は独立drawの証拠にならない。

## Strict providerと位相

`cs_v0_native.Expression` は必要な二classだけ受け付ける：exact `aπ+b` とexact `s atan(q)`（q>0）。新汎用synthesizerは作らず、旧pygridsynth endpointを呼ぶ小adapter。符号はintervalで確定し、positive keyのみ取得、負角は同じ保存列の実adjoint。sign不明・未知class・float入力はSTOP。

既存rte_reallocation.numericのatan/pi Angleでは混合角を受け取れないため、angle_value境界と同じFrobenius interval guardだけを追加した。sequence_matrix/enclosure/ivf/check_cache_metadata/adjoint_plan/native_accountingは再利用。backendのmpmath点丸め・合成差も、**元exact式に対するstrict guard**に包含する。global-phase最適化やchannel PAIは使わない。

- strict fast pathは位相込みで表せる `RZ(kπ/2)=W^(−k) S^k`（W=exp(iπ/4)I）。identity、Cliffordをexactに扱う。key/targetを周期化せず、2πでは−Iを残す。
- **strict RZ(π/4)はbare Tではない**。必要なexp(−iπ/8) scalarを落とす旧projective T経路は拒否し、通常strict synthesisへ送る。T/T†のexact文字列を保持・計数することと、任意RZのexact T fast pathを持つことを区別する。
- positive stringはwritten-product、実回路は逆順。負角は時間列全体のactual adjoint（S†、T†、W†を含む）。controlled diag(I,U)へ渡す前のphaseを落とさない。
- selected joint IZ Pauliには全arm同じexact parity CX ladderを使う。basis/partner/wrapperは変更しない。局所literal T/T†・CX・Clifford・serial depth・scalar W・古典取得費を別に保存し、full optimized価格と呼ばない。

## 環境・precision・error budgetの第一候補

[NATIVE_PROFILE.json](NATIVE_PROFILE.json)に固定。pygridsynth2.0.0／mpmath1.3.0、strict up_to_phase=false、seed0、dps=interval_dps=100、旧endpointと同じepsilon/4のbackend headroom。**各RZ strict ε_native=10^-6**を共通に提案する。旧合成研究と同じ精度は再利用可能な政策の出発点で、H4全interrogationの正しい予算と認定した値ではない。

SERVER1の既存venvにpygridsynthがないため、`/tmp/cs-v0-provider-20261108`へ専用overlayを構築した。pygridsynth wheel SHAとPython source treeは旧固定toolと一致。importに必要なcvxpy/clarabel/osqp/scsも専用directoryだけへ配置。A1共有venv・OS・共有cacheは無変更。依存版と実executable hashはprofile/TOOL_IDENTITYに固定し、旧Python3.10環境との同一性は主張しない。provider importだけ確認、backend acquisitionは人工を含め0。

実局所corrected meanに対して

`b_local ≤ b_finite + B_d b_coarse + Σ c_n δ_native,n`

を既存error_accountingへ接続。coarseと各eventがε_native以内なら `≤b_finite+2 B_d ε_native`。native評価誤差はstrict guard内、独立角近似は導入しない。missing termを0にしない。候補として **b_finite≤10^-12、各guard≤10^-6、選択joint局所specの合成bound≤3×10^-6** を事前capとする。小xによるB_dの包絡と二回転の誤差を十分覆う局所技術capで、energy精度・全回路予算ではない。S0も同じcapで記録する。V1のfull予算はUNKNOWN、旧engine allowanceは転用不可。

## Positive keysの有限上限

max2 selected joint primitive ×（S0通常1＋coarse1＋residual n0/n2各1）＝**最大8**。residual primitiveのRZ角は **2 atan(x/(n+1))**。旧controlled-helper用atan/scale1 keyへ偽装しない。既にjointへlowerしたため、residualにもう一つcontrolを付けない。

同keyは一度だけ取得・guard・保存。exact pathはbackend0回、negative/adjointは新key0。zero/重複は減数するだけ。fixed partnerやbasis gateの合成はこのscopeにない。別K・全factor・追加phase gate classを要求する場合は8を流用せずSTOPする。primitive局所費とB/rangeの交換を記録するだけで、資源削減の採択はしない。
