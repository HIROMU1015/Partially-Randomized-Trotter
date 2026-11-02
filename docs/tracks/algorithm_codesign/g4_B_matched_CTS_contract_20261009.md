# G4-B: matched finite CTS — synthesis前の固定仕様

G3再レビューv2の§13.3に基づく新しいG4-B。G4-Aの条件付き分離認証が成立したため、
既知toyについてこの対照だけ取得する。旧authorization・markerは使用しない。
この仕様、実装、key inventory、focused testsをcommitしたclean HEADから一回実行し、retry 0。
結果の勝敗によらずmandatory STOP。G4-Cは設計文書だけとする。

## 一次文献と有限specialization

Peetz, Smart, Narang, *Quantum simulation via stochastic combination of unitaries*,
npj Quantum Information **12**, 52 (2026), DOI 10.1038/s41534-025-01168-w。
[出版本文](https://www.nature.com/articles/s41534-025-01168-w)の
Methods「Convex Taylor sampling procedure」、Theorem 1/Eq. (5)–(6)、
[Supplementary Note 5, Eq. (13)–(15)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41534-025-01168-w/MediaObjects/41534_2025_1168_MOESM1_ESM.pdf)
のoperator decompositionをM=3で有限化する。

文献のSCU channel estimatorをそのまま移すものではない。そこで構成されるunitary ensembleを、
G3と同じcoherent weighted Hadamard first-moment taskへ接続するadapterである。
channelだけの一致や文献のchannel varianceを代入せず、full operator meanを別途検証する。
独自のCTS最適化版を採用したとは主張しない。

対象は2 system qubits、追加workspace 1、入力|00〉、σ=+1、
R=3/4 ZI+1/4 V†IZV、V=exp(-iπXX/16)、x={1/8,1/4}、P3(-ixR)。
分子、geometry、DF rank、L_D、PF windowはこのtoyに該当しない。
I1でPauli coefficientを取得する。I0 acquisition advantageは検証対象にしない。

## literal CTS ensemble

c=cos(π/8), s=sin(π/8)。Q1=c IZ+s XY。
厳密なPauli積から

- R²=5/8 I+3c/8 ZZ
- R³=(15+3c²)/32 ZI+7c/16 IZ+5s/32 XY+3cs/32 YX

を導く。P3の実補正は−5x²/16 I−3cx²/16 ZZ。
虚係数bは、ZI: x[x²(c²+5)−48]/64、IZ: cx(7x²−24)/96、
XY: sx(5x²−48)/192、YX: csx³/64。

Ls=Σ|b_k|、N=sqrt(1+Ls²)、p_k=|b_k|/Lsとする。
real correctionを負のPauli event二つ、rotation部を
U_k=(I+i sign(b_k)Ls P_k)/N、coefficient N p_kの四eventとする。
Σ coefficient×unitaryはfull P3そのもの。実補正の−Iをrotation側Iと融合しない。
状態のeven-parity部分空間だけへtargetを縮小しない。
−I・−ZZはcontrolled ancilla Zを含む。厳密なrelative phaseを保持する。

c⁴=c²−1/8のquartic fieldによるexact algebraを用い、指定正根を384-bit
dyadic sqrt enclosureで包む。Ls intervalのrational midpointをrotation ratioとして
**合成前**に固定する。理想角との差は2|Ls−midpoint|をjoint operator errorへ計上する。
coefficient midpointのL1誤差も別に精度予算へ戻す。

## 固定取得・law集合

- 新規RZ primitive keys: 2 x × 3 ε × scale±1 = **12**。
- ε={1e-3,1e-4,1e-6}。旧R1 pygridsynth identity/optionsを変更しない。
  request ε/4、up_to_phase=false、strict Frobenius operator guard≤ε。
- 新規native event条件: 各xにexact real二つ＋4 rotation×3 ε = **28**。
- 四つのrotation labelごとに三精度を使用できるpure profiles: 3⁴/x、計**162**。
  これは結果前固定したCTS対照の自由度であり、同じ三精度を持つG3側より制限しない。
- 同じ有限proposal規則: coefficient-proportionalとcoefficient/sqrt(cost)を
  η={0,1/2,1}で混合。zero-cost massはcanonical及び{1/64,1/16,1/4,1/2,3/4}。
  zero-cost eventも除かず、2^60 largest-remainder dyadic samplerをfull supportで構成する。
  proposal重複はexact equalityで除く。探索後のgrid、precision、angle追加なし。
- T/CX/1Q各axisについて同一confidence policyでlawを選択し、全座標を保存。
  Tがprimary、1Qはauxiliary。新たな5% materialityや研究GOを作らない。

ε_axis=1/200、α_axis=1/5280、m2=Σq w²、L=max|w|、
δ=coefficient L1+2Σ coefficient×(strict native error+angle error)。
残差ε′=ε_axis−δ>0に対し
n=ceil[ln(10560)(2m2/ε′²+4L/(3ε′))]、cap 10^9/axis。
lnはG4の独立有理区間の上端を使用し、point signalでshotsを減らさない。
G_T=2n E[T]、G_CX=2n E[CX]、G_1Q=n(2E[1Q]+5)、workspace=1。
これは固定されたsufficient-shot policyに対する会計で、物理的最小shot数の主張ではない。
native IRの共通exact cancellation後のadditive synthesized-primitive費用。
full-wrapper post-synthesis optimizer、実量子測定は実施しない。

## 資源上限・launch・保存

single process、wall 600s、CPU 480s、RSS 512MiB、virtual address 1536MiB、
periodic watchdog 0.2s。per-key wall 30s/CPU 20s、sequence 20,000 characters。
law proposals最大10,000、全G4 packet最大32MiB。LP・trajectory・GPU・DF・NPZなし。
固定scopeのhash、G4-A gate、12 keys、runtimeのpackage lock/.py tree、clean HEADを確認し、
exclusive phase_B_consumed.jsonを作成した後に合成する。
partial failure、timeout、guard failureはG4_B_TECHNICAL_INCONCLUSIVE、retryしない。

12 sequence identity/count/error、28 native IR cost bindings、162 profiles、
486 profile/axis winners、6 complete selected laws、m2/L/bias/shot/resource/certificate、
tool/source/scope identityとusageを保存する。negative prefixは全体結果としない。
保存後の再照合はread-onlyのexact arithmeticとprovenanceだけ。
old result/source/authorization/markerとTrack Aを変更しない。

17 focused testsはoff-domain x=1/3のsynthetic semantics/certificate mutants。
registered xのstatic algebra/key一覧を確認しても、合成・費用・science signalは取得しない。
G4-Aの独立証明は旧G2/G3 helperを使用しない。G4-Bのnative lowering/primitive accountingは
旧R1の凍結moduleを共通利用するため、独立外部再現とは呼ばない。
