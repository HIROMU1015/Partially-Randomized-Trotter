# N5-B3後：precision・domain・word policyの候補

2026-11-08 JST。基準2886ac07c1ec706a1180d107e0111745e74aec1f。[REPORT](REPORT.md)。以下はSOURCEに基づく設計候補・数理条件の整理であり、実行可能policyや新しい研究保証の採択ではない。新しい数値評価・価格集計はしていない。

## 1. 保存capをそのまま必須precisionにしない

[CT-0 stage_c](../../../src/trottertracks/pr_reference_ct0.py)は、旧raw disk d_old、raw半径rho0、PF控除済みB、保存整数Nから、同じHoeffding条件を保つ追加raw disk余地cを求める。その後

`E_saved = c / (2 sqrt(2))`, `epsilon_uniform = E_saved / L_RZ_max`

を外向きに評価した。M17ではN_B=626／N_R=630を保持する。一様capの導出は、すべてのRZに同じ局所Frobenius上界を与え、それをspectral上界としてjointへ加算する**十分条件**である。

保守性は、全possible K2 application/run、context-max lowering、basis共有をすべて失う場合、roleごとのmaxを同時に足すこと、全位置同精度、Frobenius→spectral、X/Y max、局所telescopingにある。どの保守性が数値的に支配するかは今回評価していない。Finite cohort平均や期待RZ数を、そのまま全support誤差和へ代入して緩めない。

Role／key別の候補は

`E_ax(P) <= sup_omega sum_i e_i(P,omega,ax)`

を、保存の安全な物理多重度又は新しく資格したjoint support制約で囲う方法。高多重度frameと少数のscalar／中央回転へ同じ局所capを強制する必要はない。**ただし中央へ配分を移すだけではframeや他roleの誤差は減らず、総和を実際に閉じる必要がある。**

Pure common joint scalarを分離する候補Gは[N5-B1/B2](../pr_reference_n5b2_static_20261108/JOINT_SCALAR_FULL_PATH_CONDITIONS.md)のSOURCE条件付きでのみ使う。厳密なlogical wordとRZ targetの差、SOURCE vacuum／scalar／eventのrelative角、compiler／finite-bit等の旧diskは残る。旧certificateのemitter／shell値を0へ書き換えない。

X/Y別のjoint spectral上界epsilon_X/Yが資格できれば、既存raw座標の意味を保った設計式として `d_syn <= 2 sqrt(epsilon_X^2+epsilon_Y^2)` を検討できる。旧 `2sqrt(2) max(epsilon_X,epsilon_Y)` はその安全な特殊上界である。これは新規採用ではなく、normalization・測定定義・両軸の誤差帰属を確認してから使う候補。d_oldとd_synは同じraw単位で一度だけ合算し、PF位相予算を二重控除しない。

## 2. 現wordの保証を調べる、合成前の候補

### 保存delta/Gからの上下囲い

固定b36の選択basic productをW、grid角をa=n2^-36とする。保存guard和Gは `||W-RZ(a)||_F <= G` を保証する。delta=|theta-a|は基準RESULTSへ保存済み。RZの対角式から

`D_F(delta) = ||RZ(a)-RZ(theta)||_F = 2 sqrt(2) |sin(delta/4)|`。

従って、三角／逆三角不等式に基づく設計式は

`L_F=max(0,D_F-G) <= ||W-RZ(theta)||_F <= U_F=D_F+G`。

一般の2×2差行列について `||A||_F/sqrt(2) <= ||A||_2 <= ||A||_F` なので、保守的spectral囲いも `[L_F/sqrt(2), U_F]` とできる。**数理的な設計式としてのCodexによる導出で、今回の研究保証へは未採用・未数値評価。** Exact W、固定word順、unitary basic、complete actual adjointが前提である。

これにより、Q+Gがcap外という上界の不成立だけでなく、同じwordが局所capに入る余地があるかを、word生成・新guardなしで調べられる。Lower endpointがcapを超えた場合も、当該word／norm／局所割当を排除するだけで、より自由なjoint配分やm17 RPEを排除しない。Frobenius下界だけをspectral下界と呼ばない。

### 実wordのguard再評価

上下囲いで未決なら、4中央角の**同じb36／同じbasic／同じ順序**のproductについて、既存exact_word_matrix／strict_guardを用いて直接targetとの誤差を囲う候補がある。これは新しいword合成ではないが、文字列連結と新strict guardは新科学実行であり別承認が必要。

旧strict_guardは既にexact cyclotomic ringとoutward intervalを用い、長いgate積のinterval wrappingを避けている。単にdpsを増やせば大幅に良くなるとは仮定しない。改善し得るのは、productへの直接評価によるtriangle上界の余裕、又は数値包絡幅である。真の近似誤差はdpsだけでは変わらない。

SU(2)同士の差ではFrobeniusとspectralの関係をさらに鋭くできる場合があるが、保存wordのdeterminantをexact W込みで確認することが条件。`up_to_phase=False`だけでその追加条件を仮定しない。今回はdeterminant分類も新規実行していない。

## 3. Word／precision候補の比較

| 候補 | 既存方式との差・最小変更 | 再利用できる資産 | 誤差／Tへの作用・成立条件 |
|---|---|---|---|
| P0：Role/key別予算 | 一様epsilonを固定しない。SOURCEとwordは当初変更なし | 保存m17 role上界、CT-0 geometry、N4-Cの両軸規約 | 中央の局所cap超過をjointで許せる可能性。全role和を閉じ、N626/630を保持する必要。T増を伴わずに証明だけ改善する可能性があるが未評価 |
| P1：同じwordの鋭い保証 | 上下囲い→必要時だけdirect guard。Word選択は不変 | 保存delta/G、exact_word_matrix、strict_guard、interval端点 | Tは同じwordなら不変。包絡だけの改善と実誤差の変更を分ける。下界が対象capを排除したら同じ検査を続けない |
| P2：保存recordの再選択 | 新policy overlayで、literal一致する有限recordからguard／実Tの候補を選ぶ。旧C0を上書きしない | C0299、43 basic、85 certificate、固定文献／SOURCEに記録された旧strict inventory | 同じliteral・strict意味・guard方式を確認。Actual guard/Tはprecisionラベルに対して単調とは限らない。新規合成なしで対応できるかは未確定 |
| P3：より細かいdyadic | bと指数域・基本precisionをparameter化した別provider。現SOURCE定数は不変更 | exact integer ties-to-even、ascending product、43の共通指数、actual adjoint | Qを減らせるが、chunk数・欠損basic・guard和・Tが増え得る。bを増やすだけではGを解消しない |
| P4：Signed digit／小window | 同じgrid角を符号付きchunkで表す有限policy。順序とcarry域を固定 | 保存positive basicとそのactual adjoint | Chunk数を減らせればguard和・Tも減る余地。Carryで新しい上位指数が必要な場合がある。最小chunk数と最小Tは異なる。近似wordの可換性や自動相殺を仮定しない |
| P5：有限hot keyだけ追加合成 | 目標誤差を先に固定し、中央等の有限missing/不適格keyだけをstrict合成。残りは再利用 | 既存acquire／strict_guard／record再利用、4中央literal | Dyadic Qを避け直接targetへ近似できるが、取得成功と実Tは未取得。少数の高精度wordが全域policyを完成するわけではない |

同じgrid角のsigned digit化だけではQは変わらず、同じbasicのguard包絡を鋭くするだけでもgrid差deltaは変わらない。必要な分岐は、予算／保証を鋭くする、grid又は直接targetの近似構成を変える、のいずれか又は組合せである。Direct guardは相殺も含むword全体とtargetの差を囲うが、理想grid差を0と仮定する操作ではない。

P2の既存7603 inventory等を再利用する場合、別Hamiltonianのwordでも**同じbinary RZ targetそのもの**へのstrict recordなら候補になる。しかし旧Full Y4のwhole保証・価格差をPartial S2へ移すことはできない。Guard方式のprovenanceが欠ける旧recordは未資格のままとし、欄を推定で埋めない。

有限候補の配分には既存[strong-baseline optimizer](../../../src/trottertracks/track_c2_b_strong_baseline.py)の「float solverは候補だけを返し、Fractionで実行可能性を再確認する」部分を再利用できる。一方、offers_for／optimize_caseは登録2精度・nominal 10^eモデルに特化している。新catalogへそのまま呼ばず、必要ならmultiple-choice制約へ最小変更する。旧C2の最適性gap・下界・alpha最適化を今回の証明へ移さず、旧STOPも解除しない。

## 4. Domainを直す候補

いずれも**SOURCEのvacuum生成を変えず、その後のactual native thetaを受けるpolicy層**の候補。B′scalarの実域外literal、vacuumの包絡超過、eventの安全supersetは[N5-B2の分類](../pr_reference_n5b2_static_20261108/M17_ANGLE_AND_ROLE_GAPS.md)のまま区別する。

### D1：既存最大basicの反復で域を広げる

正のgrid indexを `|n| = c * 2^(b+6) + r`、`0<=r<2^(b+6)` と整数で分解する。Ideal角は `c*64 + r*2^-b`。既存j=6 wordをc回、その後に固定順序の低位basicを使う数学的構成で、追加basic取得なしに有界Lへ有限停止を拡張できる候補である。

Guardは `Q + c*g_6 + sum_low g_j`、Tは `c*t_6 + sum_low t_j`。cの上界はLから定め、負側はcompound全体のactual adjoint。浮動remainder angleを新しく作ってsnapする必要はない。単なるL定数変更ではなく、別providerにmultiplicityと証明を追加する。Tはcに応じて増え、繰返し二乗の古典処理を物理gateの無料化に数えない。**Domain rejectの構成上の解消候補であってprecisionや総Tの改善証拠ではない。**

### D2：高位・低位basicの有限拡張

既存N3の式 `N_b=floor(L*2^b+1/2)`、`J_b=bit_length(N_b)-1-b` を使い、指数-b…J_bへ拡張する。共通指数は保存wordを優先し、目標guardへ不足する部分又はmissing指数だけを取得候補にする。B3中央の細分化と大きいvacuum/scalarの域を同じ枠で扱えるが、b・J_b・内部guard・価格は未選択である。有限keyの取得成功とstrict guardを別途確認する。

### D3：Strict scalar補償付きの周期表現

数理恒等式は、任意整数kについて `RZ(theta)=(-1)^k RZ(theta-2*pi*k)`。したがってexact signをW^(4k)等として保持すれば、strictな小残余角表現を設計できる。ただしbinary tau0を真の2piと同一視しない。

例えばkをexact binary theta/tau0の整数丸めで定め、binary残余rを使う場合、追加relative角差は `eta = theta-2*pi*k-r`。安全なFrobenius項は `sqrt(2)/2 * (|theta-k*tau0-r| + |k|*|tau0-2*pi|)` で囲う候補となる。選んだkの最適性は恒等式に不要だが、残余域と誤差の証明は必要。高精度の真pi区間による還元も別候補である。

Wの符号補償、丸め・周期差の予算、正負whole adjoint、端点、同一literalの選択規則を新規に検証する。候補Gでも**相対角etaは消えない**。現policyの2pi key同一視禁止を解除したものではなく、新方式の採否には明示判断と実行承認が必要。SOURCE段階のP角を書き換えたりphase fitで補ったりしない。

## 5. 強baselineと全体費用

旧policyは履歴比較F0として保存する。主比較は、同じarchive・価格取得機会・domain構成・precision候補・保証改善を与えた**強いB′**とReference。Literal一致の共通word、actual adjoint、同じlaw、同じ情報条件を保つ。候補だけに真E0／gap／groundの古典記述を無料供給しない。

Finite record選択の設計は、各keyの候補を一つ選ぶ整数変数z、誤差制約 `sup_omega sum n_i(omega) e_i <= E_saved`、費用 `sum mu_i t_i` と書ける。誤差には全supportの上界、費用には同じcanonical lawの期待多重度を使い、取り違えない。実離散Tと正確な可否を採用し、LP下界は実行可能baseline価格ではない。

M17の費用は `T_B=626*(E[T_BX]+E[T_BY])`、`T_R=630*(E[T_RX]+E[T_RY])`。共通frameの1-shot価格はcancelしても、異なるshot重みの差では絶対価格が残る。局所precision向上でshotを勝手に減らさず、最初は保存N保持を可否条件とする。

全18roundは `sum_m sum_axis N_arm,m,axis * E[T_arm,m,axis]` で比較する。M17の成功だけで他roundを資格せず、保存各roundのPF／r／bits／誤差余地を使う。Shared cacheは古典的再利用であり物理Tの共有ではない。未取得vacuum価格、domain未資格、未確定wordを0にせず、CX／RZ／depth／追加qubitをTへ合算しない。

既存C2の共役block誤差は、帰りが同じ近似Bのactual adjointなら参照できるが、現在のSOURCE inverse／別native loweringへ無条件には移らない。今回旧C2・fusion・SK・残差normalization研究を再開していない。

**複数候補を残す。** P0/P1は保証の保守性、P2/P4は取得済みwordの使い方、P3/P5は真の近似精度、D1–D3はdomainを扱い、互いの不足を代用しない。未取得Tと成功保証が閉じる前に勝者・改善率を認定せず、[次計画](NEXT_VALIDATION_PLAN.md)で段階を分ける。
