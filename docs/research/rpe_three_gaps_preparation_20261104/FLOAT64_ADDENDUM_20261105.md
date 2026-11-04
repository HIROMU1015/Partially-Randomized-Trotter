# B追補：Float64 decoderの一様誤差保証・技術準備

2026-11-05 JST。親SOURCEは`61f569c32a94c2b698df10a62a51e515b12160ed`。
専用branch `codex/rpe-float64-decoder-preparation-20261104`（着手日名）、独立cloneで作業した。
これは凍結済み[PROOF_AND_SCOPE.md](PROOF_AND_SCOPE.md)のBへの**新規追補**であり、旧ファイルは変更しない。

**結論：一部の一様上界は構成可能だが、全18roundのFloat64 branch保証は未解決。
H4科学実行準備③には達しておらず、科学runner・実行計画は新設しない。**
距離関数全域の誤差証拠と、全18round測定からscalar入力への具体的な接続SOURCEが不足している。
これは不可能性の証明ではない。以下の条件付き補題と有限格子法が見通しを与える。

固定研究条件はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12／Partial L_D=3／δ=.02、
保存physical state normalized ray／4電子sector70、m0..17、q=2^m（最大131072）、保存N/α/β/ε/μ₀/E_ref。
これらの**新しい数値評価・成功格子の計数・列挙は行っていない**。
sampling、RNG replay、回路build/compile、費用評価、旧float guard/BLAS系列の再開もない。

## 達成段階と必要な証拠

| 対象 | 到達点 | 残る条件 |
|---|---|---|
| SOURCEの演算・既存呼出経路 | ①確認済み | 全18roundの実測定→scalar callerは未固定。異なる経路を統合しない |
| runtime | ①version・実体・静的call edgeを確認 | 動的dispatchの完全な同定や精度契約は未取得 |
| mean変換 | ②RN条件付き一様補題 | 保証対象のmean経路・FP条件との接続 |
| a_m | ②供給された有限round集合では構成可能、人工検証済み | 全成功count pairの被覆と実際のscalar呼出形状・runtimeの固定。H4未適用 |
| rootのwrap前 | ②全kに一様な基本演算補題 | これはc_m全体ではない |
| c_m | ②exact-fmod条件付き補題／実runtimeへの適用は④ | 対象剰余kernelとの対応・補題の前提証明 |
| d_m・全候補順序 | ④未解決 | 全引数域のcomplex exp・atan2精度、または到達域の完全被覆 |
| H4実装branch・energy | ③未到達、④UNKNOWN/STOP | 上記を揃えてから限定科学SOURCEを作成・公開し、別途実行承認 |

## 既存成果の保全・固定GitHub照合

GitHub上の次の固定commitを実際に確認した。親commitのREPORT、RESULT、decoder、両証明文書、
`decoder_finite.py`は公開Git blobとlocal bytesの一致を照合した。
[照合hash・runtime・人工結果](DECODER_PREPARATION_VALIDATION.json)を併記する。

- [A/C結果61f569c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61f569c32a94c2b698df10a62a51e515b12160ed/docs/research/rpe_three_gaps_execution_20261104/REPORT.md)、科学SOURCE `d29a86963f91044a95217d7d2e2ee9703d2f1dc1`。
- [branch/energy証明SOURCE6942e51](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6942e5176d2dec115ac3c271e48f84377abda915/docs/research/rpe_branch_energy_preparation_20261104/PROOF.md)、結果 `89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f`。
- [全18round包含結果d2e7a7d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2e7a7db683e116243eae061c0e4c87865054e9e/docs/research/rpe_18round_ideal_enclosure_execution_20261104/REPORT.md)。再計算しない。

A/Cの成立値は**既存結果の引用**：ideal IIDの成功下界≥.950307030063252182、
保存DF sector ground reference誤差上界約1e−8 Ha、reference込み理想energy上界≤.000138689200360119854 Ha。
この追補では再計算も判定変更もしない。実装energy上界はnullのまま。
全分子、他sector、upstream積分/DF誤差、実state preparation、PF有効固有stateとの同一性、実測定bridgeは未保証。

正式共通運用指示`research_project_instructions_unified_2026-10-25.md`の§8.1、§8.4、§10は
共有checkoutにある文書をread-onlyで確認した。今回の技術作業をまとめて実施し、新しい承認証明機構は作らない。
既存decoder、凍結SOURCE/raw/grant/ledger/marker/STOP、他branch/worktree、Codex C、SERVER2 H8は変更しない。

## SOURCEとruntimeの対応

| 処理・呼出元 | 確認したSOURCEと適用範囲 |
|---|---|
| 旧explicit測定→scalar | `rpe_hadamard_failure_validation.py`のint8 outcomes→`float(np.mean(outcomes))`→`rpe_four_round_phase_validation._explicit_reconstruction`のcomplex→scalar `np.angle`→`reconstruct_rpe_phase`。旧4round経路 |
| count格子→vector | 同phase moduleの`_marginal_branch_monte_carlo`は`2.0*k/N−1.0`→配列angle→**別の**`_vectorized_reconstruction`。SOURCE閲覧だけ、RNGは実行していない |
| fresh-IID診断 | `fresh_iid_adapter.summarize_axis`の`math.fsum(outcomes)/N`。q1/q8の測定診断で、全18round scalar decoderへの接続ではない |
| V2全18round scalar | `run_saved_v2.reconstruct_saved_means`は保存されたanalytic signal→scalar angle→scalar decoder。測定countを入力していない |
| scalar decoder | phasesをfloat化、初回wrap、以後principal wrap→全kのroot生成→配列distance→`np.argmin`。q₀=1かつ連続倍増を要求 |
| distance | `abs(angle(exp(1j*(candidates−previous))))`。scalar decoderの内部でも、distanceは長さqの**配列ufunc** |
| energy | V2は`-final_phase/delta`。oracle/E_refは選択後の診断にのみ使用。np.partitionもrunner-up診断のみ |

`rg`による直接呼出と、辞書経由のV2呼出を確認した範囲で、全18roundの実測定をscalar decoderへ渡す
production callerは一つに確定できなかった。既存scalar本体を保証対象とすることと、mean生成callerの確定は別の要件。
本追補は両mean経路を独立の補題として扱い、count-gridをproduction経路として自動採択しない。

同じ既存venvのPython3.11.0rc1／NumPy1.26.4／python-flint0.9.0（既存A/C報告と同version）、
x86_64／glibc2.35で人工検証した。radix2、mantissa53、観測時`fegetround()==0`。
NumPy拡張とmapped libmのpath/hashを保存したが、それだけを精度証明にはしない。

ローカルNumPy Python SOURCEで`np.angle`は実部・虚部をarctan2へ渡し、整数入力の`np.mean`は
既定f8 accumulatorでsum後に除算する。ELF/disassemblyには次のcall edgeが存在した：

- `DOUBLE_arctan2 → atan2@plt`。
- `npy_cexp → cexp@plt`。
- `DOUBLE_remainder → fmod@plt`。
- 通常arctan2と`DOUBLE_arctan2_AVX512_SKX`、SVML atan2 symbolが共存。観測CPUではAVX/AVX2=true、AVX512=false。

upstreamの[NumPy dispatch SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/loops_umath_fp.dispatch.c.src)も
通常経路とAVX512/SVML経路を区別する。これらは通常経路を示唆する**静的証拠**であり、
全ufunc呼出の動的dispatch証明ではない。自分の人工childに対するgdb追跡は`ptrace: Operation not permitted`で取得できず、
sudo・OS設定変更等で回避しなかった。共有環境・venvへの書込みは行っていない。

[NumPy arctan2仕様](https://numpy.org/doc/1.26/reference/generated/numpy.arctan2.html)のrange・C library・特殊値の説明は確認した。
ただし今回取得した資料には、対象binaryへ適用できる全域の検証済みulp契約はない。
version名、symbolの存在、人工残差の最大値をその代わりにしない。

## meanからa_mへ：条件付き解析と有限被覆

u=2^−53、η=2^−1074とする。以下の解析はbinary64 RN（nearest/ties-even）、通常の演算順序、
gradual underflow、overflowなしを前提とする。観測したFP modeだけで将来process全域の前提を証明したとはしない。
1≤N≤2^52、0≤k≤Nなら、h=(2k−N)/Nに対し：

- integer-sum：2k−Nはexact整数。1回のRN除算の誤差は≤u。
  int8 ±1のnp.meanも全中間整数和の絶対値≤Nなのでf8加算はexact、同じ除算になる。
  math.fsumもこの整数域ではexact sumを作る。非zero平均は≥1/Nなのでsubnormalではない。
- count-grid：2kはexact。RN(2k/N)の誤差≤2u、その後の−1の丸め≤u、合計≤3u。
  N3・k1などでinteger-sumとbit値が異なる。二つの上界を混ぜない。

成功域のexact signalの半径下界ρ_G>0が別途あり、座標丸め上界r_c,r_sに対して
D_mean=sqrt(r_c²+r_s²)<ρ_Gなら、平均変換の円周誤差は≤asin(D_mean/ρ_G)。
ρ_Gには例えば既存中心半径下界−座標disk半径上界を使えるが、正値の保証が必要。
保存μ₀をPF単位半径とせず、今回H4のρ_Gを評価しない。
解析でa_mを完成させるには、その後のatan2とwrapの上界も必要。

代替として、各roundのexact strict成功集合

    S_m={(k,l): |(2k−N_c)/N_c−μ₀,c|<ε_c,
                  |(2l−N_s)/N_s−μ₀,s|<ε_s}

を完全に被覆すれば、実装のmean→scalar angle→最初のwrapを各点で実行し、
exact rational signalのAcb偏角との円周誤差をArbで包含してmaxを取れる。
これは**列挙した有限集合に対する証明**であり、未知のlibm精度を仮定するspot-checkではない。
μ₀にはbinary64のexact rational、εには元契約のexact値を使う。origin、非有限、空集合、未確定pathはUNKNOWN。
全履歴の積を作らず、angle評価はΣ|S_m|回で足りる（現helperの集合抽出は各roundで(N_c+1)(N_s+1)のexact比較）。
scalarとvector化したangleは同一とは仮定せず、呼出shapeを変える高速化には別の適合根拠が必要。

新規[`finite_input_envelope`](decoder_numeric_preparation.py)は、この一点の拡張だけを行う。
旧`feasible_pairs`、`represented_mean`、D/Arb helper、既存SOURCEのwrapを再利用する。
既存finite checkerはangleまでで最初のwrapを含まず、全履歴も列挙するため、a_m専用の最小拡張が必要だった。
m0はこのa₀に実際の最初のwrapを含め、c₀=0という会計にできる。同じwrapを二度誤差加算しない。

## 全rootの基本演算とwrap

p64=math.pi、D64=2p64。Arbで|p64−π|<2uを確認でき、D64の2倍はexact。
q=2^m、2≤q≤2^17、principal p∈[−p64,p64]、0≤k<qに対し、実装のwrap前を

    t=RN(D64 k), s=RN(p+t), v=RN(s/q)
    v*=(p+2πk)/q

とする。|p|<4、D64<8、q≤2^17でoverflowはない。
|t−D64k|≤8uk（k0はexact）、|t|≤8k(1+u)、
|s−(p+t)|≤u[4+8k(1+u)]+η。
qによる2進scaleはnormal結果ならexact、underflow部分も誤差≤η。
π64との差も含めると

    |v−v*| ≤ [8uk+4uk+u(4+8k(1+u))+η]/q+η < 64u.

これは**全kに一様なwrap前の条件付き補題**で、巨大q列挙を必要としない。
helper`prewrap_bound_under_rn`はこの値だけを返し、c_mを返さない。
人工テストのroot比較は実装との整合確認であり、この補題の根拠を経験的最大値に置き換えない。

wrapについては次の補題を構成できる：x∈[−4,8]でy=RN(x+p64)、
**fmod(y,D64)がexactなtruncation余りを返し**、NumPyの符号補正と基本演算が上のRN条件なら、

    d_T(wrap64(x),x) < 64u.

y∈(−D64,2D64)なので必要なperiod整数jは−1,0,1のみ。
加算誤差≤12u+η、負余りのD64加算≤8u+η、最後の減算≤8u+η、
period差≤|j|·|D64−2π|<4uを足せば64uに収まる。
負余りの符号補正自体はexactとは仮定していない。正負πの代表元差は円周距離で扱う。
これが対象runtimeに適用できれば、上のwrap前補題と合わせc_m≤128uが得られる。
**本追補はその値をproduction c_mとして供給しない。**

[NumPy npy_divmod SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/npymath/npy_math_internal.h.src)は
fmod後の符号補正を実装し、[glibc2.35 generic fmod SOURCE](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/dbl-64/e_fmod.c)には
exactなshift/subtractの方式がある。しかしinstalled libmの選択kernel・buildとの同定と、この限定域の前提確認は未完了。
generic SOURCEの説明だけを現在のbinary全域に転用しない。

## 全候補との対応・距離・argmin

θからfloat principalへの円周誤差≤a_mなら、principalの整数巻数に応じて理想rootのlabelを巡回置換する。
seamでlabelが変わっても、全q rootの集合は同じ対応を持つ。
全kの候補誤差≤c_mかつc_m<π/qなら、異なるfloat候補の円周間隔は
2π/q−2c_m>0。q個の生成loopと合わせ、重複・欠落なしの全単射対応が得られる。

B_m=b_m+a_mとし、帰納的な前推定誤差≤2B_(m−1)/q+c_(m−1)、現在の正解root誤差≤B_m/q+c_mなら、
正解候補と前推定の距離≤R=[2B_(m−1)+B_m]/q+c_(m−1)+c_m。
同じperturbation前の理想rootと理想前推定を使うと、任意の誤候補との距離差の下界は

    2π/q−2[2B_(m−1)+B_m]/q−2[c_(m−1)+c_m].

全候補の計算距離に絶対誤差≤d_mがあれば、計算された差はさらに2d_mを失うだけなので、旧Dの十分条件

    π−[2B_(m−1)+B_m+q(c_(m−1)+c_m+d_m)] > 0

で一意な正解argminが得られる。Dのadapterは別に全roundのB_m+q_mc_m<πも要求する。
q0のrootは一つ。m≥1は上のstrict条件からc_m<π/q_mも従う。
有限floatのstrict比較ならargmin自体に演算丸めはなく、tie/NaNがあるとこの含意を適用できない。

**d_mが最大の残存課題。** Δ64=RN(candidate−previous)は概ね[−2π,2π]で、基本減算の誤差は
≤8u+ηという条件付き上界を置ける。1j倍とabsは有限入力なら符号・成分操作で扱える。
しかしcomplex expと、その出力に対する配列atan2の全域誤差が不足する。
例えば|exp64(iΔ64)−exp(iΔ64)|≤e_exp<1と、実際のatan2に円周誤差≤e_angle、
有限principal rangeの契約が得られれば、概念的には

    d_m ≤ (8u+η)+asin(e_exp)+e_angle

という形に接続できる（absは円周距離への1-Lipschitz写像。π64<πであることも定数で確認可能）。
これらの**未取得契約に仮の数値を割り当てるwrapperは作らない**。

root間隔が大きくても、遠い候補の計算距離が誤って小さくなる可能性を排除する根拠にはならない。
人工テストでは全てfinite・nonnegativeの距離でも遠い候補が勝つ注入例を確認した。
これは実libmの反例ではなく、全候補のd契約が必要な理由の検証である。
近い2候補のみの比較や保存平均のgapから、距離計算全域の保証を主張しない。

libm契約の代わりに全到達引数を区間で被覆する道もあるが、単純列挙は
root生成Σ|S_m|q_m、距離は前roundの可能estimateとの組合せ（粗くO(Σ|S_(m−1)||S_m|q_m²)）、
全履歴は∏|S_m|となる。正解labelが既に帰納的に証明された部分なら前estimate集合を削減できるが、
未証明のbranchを仮定して探索域を削減しない。H4の|S_m|やcall数は今回算出していない。
距離引数の対称性・範囲blockによる一様誤差/順序証明が未構成のため、これを科学実行案として採択しない。

## 人工検証・次の最小作業・STOP

新規[`test_decoder_numeric_preparation.py`](test_decoder_numeric_preparation.py)16件と、
変更しない`test_three_gaps.py`28件、計44件PASS。新規テストは192 bit、外向き上界照合は384 bitも使用した。
全N≤19の小さい人工mean格子、scalar一roundの全成功pair、負実軸・±π、origin、empty/nonfinite、
q2..32の全root、q16までの実SOURCE候補捕捉、strict不足・tie・遠い候補の誤距離を確認した。
旧checkerの全36人工履歴・誤branch検出・signed zero/subnormalテストも再利用。
候補捕捉と誤距離注入はテスト中のAST-loader namespaceだけに適用し、既存decoder SOURCEは変更しない。

再現コマンド（repository rootから、同等依存環境で）：

```sh
cd docs/research/rpe_three_gaps_preparation_20261104
python -m unittest -v test_decoder_numeric_preparation test_three_gaps
```

次の最小作業は、ユーザーが継続を指示した後にCodexが行う**技術準備**：

1. 全18roundを対象とする測定平均→scalar decoderのcallerを確定する。既存経路のどれを保証するかを明示し、無断統合しない。
2. そのruntimeのremainder／complex exp／atan2について、限定引数域の誤差証明と実kernelへの接続を取得する。
   version一般論やdynamic trace単独では精度証明にならない。OS制限の回避や停止済みBLAS系列の再開は必要としない。
3. a/c/dが揃った場合に限り、旧36軸・A/C・ideal marginを再利用する限定H4 SOURCEを作る。
   入力commit/path/hash、precision、全成功集合、計算量・call数、strict比較とSTOPを固定・公開する。
   今回は③未達なので、その科学call範囲・precisionを架空に確定しない。
4. ユーザーの**別途明示した科学実行承認後**にのみ、全17 margin、alias/lift、最終energy余裕を数値判定する。

将来の最終円周誤差は(b_17+a_17)/q_17+c_17。alias priorと代表元境界のstrict余裕を確認した後にのみ、
U(t)=exp(−itH)の符号でE_hat=−φ_hat/δとし、energy上界へ1/δで移す。
binary64のδをexactとしてRN除算誤差も別に包含し、Cのreference誤差を一度だけ追加する。
U_cat・PF/RTE誤差・α・referenceを再配分/二重加算せず、保存目標は変更しない。
AのPr(G)が実装energy成功確率へ移るのは、G全域でこの決定論的な接続が全て成立してからである。

現時点：新H4科学call=0、H4成功格子評価=0、a/c/dのproduction値=null、実装energy上界=null。
`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持。**公開照合後mandatory STOP。**
