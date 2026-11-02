# N3占有情報を保持する保証：数学検証と比較設計

2026-10-12 JST。状態：`N3_REDESIGN_PREPARATION_COMPLETE_MANDATORY_STOP`。
今回の認可はレビュー由来の理論・保存証拠の検証と次の比較設計まで。
新しい科学batch、生成器の移送実行、N1/N2再実行は未認可・未実行。
中心テーマ、実分子への拡張、総量子資源優位は未採択。

## 1. 入力・証拠と今回確認した範囲

利用者資料は[受領レビュー](../../artifacts/n3_occupation_redesign_preparation/2026-10-12/review_intake/n13_transfer_recovery_scientific_review_2026-10-12.md)。
[intake receipt](../../artifacts/n3_occupation_redesign_preparation/2026-10-12/intake_receipt.json)
は元Markdown全体と付録10ファイルのSHA-256を保存する。
付録のclosing fenceに必要な改行だけを元ファイル長に合わせて除き、全10個の記載hashと一致した。
受領コードは来歴資料として保存し、実行・importしていない。
新しい[stdlib checker](../../src/trottertracks/representation_exploration/n3_occupation_certificate_audit.py)で証拠を検査した。

基点はN3 publication `64cf5ed227b7e4a74b37453cbe733db5f5d51f56`。
元source `178d69a31dc7f2f1162d76294ba35a94b67d7fee`、登録結果、STOPは保持する。
入力pack SHAは`9e99bda0d98ede7a5eea97002f36ab00378655884f35b0b0aec9af7692078d12`、
登録ledger SHAは`488c56f2cd8ffdf392dc6fc2912d65f54ed5eca00472f7b5067958229cad7fa7`。
T3をpackの係数そのものと照合し、ledgerの同じcase hash・元Uに結び付けた。

今回のdevelopment証拠は実対称one-body＋非負unordered density、一外部mode、
active m=7、ν=3、full modes=8、外部d=7、scale=2、結合振幅1/40、high_coordinates。
geometry・化学basis・DF rank・L_Dは適用なし。時間発展/PF/PR窓も適用なし。
目標投影エネルギー幅δ=1/250。既存12入力はすべてdevelopmentで、独立移送成功数は追加しない。

## 2. CAR・固定sector・外部densityの独立導出

粒子数保存Hamiltonianを
\[
H=\sum_{ij}h_{ij}a_i^\dagger a_j+\sum_{i<j}J_{ij}n_i n_j,
\quad h=h^\dagger,\quad J_{ij}\ge0
\]
とする。外部dが空のP、占有されたQに固定ν-sectorを分ける。
activeはm個、1≤ν≤m。一対のdensity項を一度だけ数える。
\[
A=d\Gamma_\nu(h_A)+V_{A,\nu},\qquad V_{A,k}\succeq0,
\]
\[
C=h_{dd}I+d\Gamma_{\nu-1}(h_A+\operatorname{diag}J_{id})+V_{A,\nu-1}.
\]
v_i=h_{id}、b=Σ_i v_i a_iとしてP←Q結合はb†a_d。
\[
BB^\dagger=P b^\dagger a_d a_d^\dagger bP=b^\dagger b
=d\Gamma_\nu(vv^\dagger).
\]
従ってν≥1のこのsectorで||B||²=||v||²。ν=0を無条件にこの等号へ含めない。
途中indexのdを外したQ-sector同型にはfermionic parityの対角符号が必要。
新しいsynthetic occupation実装でd=0/2/3、ν=1/2/3の9組に対し、
BB†恒等式と符号を含むC分解を有理数行列の一致として確認した。
スペクトル・ground-state計算は行わない。

## 3. 固定粒子数の固有値和証拠

Hermitian Gについてσ_k(G)=Σ_{j=1}^k λ_j↑(G)、σ_0=0。
一体固有basisへのunitary変換とPauli排他原理より、
λ_min(dΓ_k(G))=σ_k(G)。これは固定k-sectorの恒等式であり、
interacting Aそのものを一体スペクトルに置き換えたものではない。

Z≽0、G−τI+Z≽0を有理数で確認すると
\[
\sigma_k(G)\ge k\tau-\operatorname{tr}Z.
\]
証明：0≼γ≼I、trγ=kのKy Fan表示で
tr(Gγ)≥kτ−tr(Zγ)≥kτ−trZ。
これは既知のtrace最小化原理を用いる有限証拠である。
[Liang et al., 式(1.1)](https://arxiv.org/html/2104.00257v1)
に最小固有値和の変分表示がある。SDP化自体を新規性としない。
数値固有分解は将来のτ/Z proposal専用で、今回は保存τ/Zを検査するだけ。
k=0の証拠は厳密なzero、k=mではtrace恒等式を利用できる。

## 4. 新条件とSchur証明

係数から有効な上界a=λ_min(A)≤U、下界C≽cIを取得する。
δ>0、c>U、D=c−U+δ。aは証明中だけに使い、算法への入力にしない。
Kκ=κh_A−vv†について
\[
0\le\kappa\le D,\qquad \underline\sigma_\nu(K_\kappa)
\ge\kappa U-\delta D
\tag{N3-OCC}
\]
ならκV_A≽0と第2節から
\[
\kappa A-BB^\dagger\succeq(\kappa U-\delta D)I.
\]
z=a−δとするとC−zI≽DI>0、逆行列は(C−zI)⁻¹≼I/D。
したがって
\[
\begin{aligned}
A-zI-B(C-zI)^{-1}B^\dagger
&\succeq A-aI+\delta I-BB^\dagger/D\\
&\succeq(1-\kappa/D)(A-aI)+\kappa(U-a)I/D\succeq0.
\end{aligned}
\]
block Schur補元によりH−zI≽0、変分原理によりE0(H)≤a。
結論は0≤a−E0(H)≤δ。c>Uの契約を維持し、その前提を事後緩和しない。
Schur補元の使用は既知の方法で、
[Dusson et al., Theorem 1.2](https://arxiv.org/html/2105.02058v1)
が近接する一次資料。本資料のoccupation射影については上の有限次元証明で検証する。
引用論文の固有空間射影・gap仮定をそのまま借用していない。

## 5. 旧条件との包含と有限生成器の区別

旧証拠κh_A−μI−vv†≽0、κU−νμ≤δDなら
σν(Kκ)≥νμ≥κU−δD。新証拠にτ=μ,Z=0を使用できる。
κ=0ならK0=−vv†、σν(K0)=−||v||²であるから、
δD≥||v||²のnorm-onlyを含む。τ=0,Z=vv†で厳密に移せる。
これらをexplicit fallbackとして実装すると、既存のvalid証拠を数値探索により失わない。
ただしfallback受理を新条件固有のgainへ数えない。

数学的可行集合の包含は、有限proposalが全証拠を発見する証明ではない。
σν(Kκ)−κU+δDはκに対して凹だが、丸め・縮退・PSD境界では未判定があり得る。
optimizer SUCCESSや浮動小数点の正marginのみで受理しない。
新しい有限生成器はこの準備では実装・実行していない。

## 6. 保存T3証拠の独立検査結果

[一次audit](../../artifacts/n3_occupation_redesign_preparation/2026-10-12/independent_witness_audit.json)：
`EXACT_REVIEW_WITNESSES_PASS`。元行列からのLDLと全要素再構成、zero pivotのzero row条件で確認。
review側のSchur消去コードと記載PSD判定フラグを証拠として信用せず、数値toleranceはzero。
756 assertions、7 distinct PSD matrices。監視された有理量の最大859 bits。
bit監視はdecoder、LDL/再構成、明示q/totalに適用し、helper構成の全一時値を計測したとは主張しない。

| 共通taskと下界 | exactに確認した結果（数値は丸め表示） |
|---|---|
| U=−4/5、δ=1/250、c_occ≈−0.6867984155405793 | 新条件margin≈+0.00011627668509375612 |
| 同じU,c_occ,δのnorm-only | margin≈−0.0001561936621623172、不受理 |
| 同じU,c_occ,δの旧LMI族 | 非zero PSD W、tr(WR)=0、tr(WM0)/trW≈−4.4559099334e−5。すべてのκで不可能 |
| U,δを保持しc_dh≈−0.5902597889520015へ変更 | norm-only margin≈+0.0002299608441919942、受理 |

W=t uuᵀ+(1−t)wwᵀ、0≤t≤1。W自身も有理PSDとして別に検査した。
Mκ=M0+κRのtraceが全κで負ならMκ≽0は不可能。
μを最小許容μ0=(κU−δD)/νへ移すと旧条件はMκ≽0と同値なので、
μを変えてもこの旧族を救えない。別c/U、別族の不可能性は示していない。

強化対照はp=ν−1=2に対して全21対角occupationの係数エネルギーを列挙し、
c_diag=−1/3を再計算、O=h_A−diag(h_A)のσ2下界証拠を検査する。
C≽[c_diag+underline σ2(O)]Iは最小固有値の和の下界による。
任意pで対角最小化が安価・多項式とは主張しない。
Uには明示determinantのRayleigh値≤−4/5を係数から再確認し、trial存在を証明した。

この検査では真のa/E0/gap、付録の診断widthを参照しない。
eigensolver、optimizer、many-body diagonalization、compile、samplingはすべて0回。
数学的分離は確認できたが、強いnorm-onlyとの算法全体の利益はこの例で支持されない。

## 7. 費用と性能準備

今回の支配処理はm=7の7個の有理LDL検査、O(m³)有理演算＋係数occupation列挙。
matrix内容をkeyとする不変PSD cacheを使用する。小さい単一保存証拠なので1 process/1 thread。
CPU32/affinity32、MemTotal約62.5 GiB、MemAvailable約51.7 GiBを確認。
nvidia-smiはdriverへ接続不能、GPU backendは利用しない。
観測wall≈0.02464秒、CPU≈0.02461秒、Linux max RSS 18,350,080 bytes。
この一回の保存証拠検査の値で、生成器の速度や大規模での効率を意味しない。
逐次/並列の比較がないのでspeedup倍率を主張しない。

一般費用は以下を別fieldに記録する設計へ固定する。
trial U：determinant/二determinantの数と係数評価、
c0：一体下界proposal/PSD、c_occ：一体固有値和proposal/PSD、
c_dh：choose(m,p)の対角列挙とOの固有値和証拠、
判定：κ proposal・数値eig呼出し・丸め候補・exact PSD・証拠bytes・bit長。
算術演算数が多項式でも、有理bit長や対角列挙が一般fillingで大きくなる点を残す。
大系でのtrial U・density下界取得の効率性は未解決で、今回評価しない。

## 8. 次の比較設計と判断点

[結果前契約案](n3_occupation_transfer_design_v1_20261012.md)と
[新8係数入力](../../artifacts/n3_occupation_redesign_preparation/2026-10-12/proposed_transfer_inputs.json)
を結果計算前に保存した。新入力のU/c/κ・受理・真値は計算していない。
その入力は無作為分子標本・統計的blindではなく、設計に未使用の合成係数scope。
比較は下界c0/c_occ/c*=max(c0,c_occ,c_dh)×norm/five/旧continuous/new OCC。
各行でU,c,δを共通化し、旧3法の安全なORを主対照とする。
強い外部下界を使ったORを超えない場合、新条件の独立主題化を支持しない。

N1：復旧済み新結果を受領し、現有限候補のscan停止。N2：補助修正を保存、探索保留。
旧Track A/B、N1/N3の元batch/STOPは変更しない。
次のGPT判断へ渡すものは本数学audit、強化対照のexact証拠、費用分解、限定契約案。
科学実行には新source/検証と別の明示認可を必要とする。
