# C1 fixed macrocycle：技術設計・最小判別の準備 v1

2026-10-13。元の量子化学Hamiltonianのground-energy計算を低コスト化する目的を維持する。
ユーザーが明示した[完了GPT選定レビュー](qc_four_algorithm_candidates_scientific_selection_review_2026-10-13.md)を採用したCodex技術準備である。新しいGPT科学レビューではない。
レビュー原文40,750 bytes、SHA `4558a0adefab4cc0d80c86328ff73cf6e4af2207376310733e142b85919ea5a1`を保存した。
C1は最初の成立可能性判別の準備対象で、新規算法・中心テーマの採択ではない。C4は次点、C2は強対照、C3は保留。

今回完了したものは、exact BCH辞書、選択規則、回路費用template、限定pilot source、認可gate、独立保存checker、登録外技術検査と実行前契約。
登録入力へのD代入・selector実行、固有値/PF性能計算、分子生成、compile、samplingは行っていない。
[準備証拠](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/off_domain_checks.json)、[実行契約](qc_c1_macrocycle_preregistration_v1_20261013.md)を対応させる。

## 1. 辞書を三つの非自明な順序classへ固定

fragment A,B,CはHermitian、相互に非可換。各fragment内部の演算子は可換という最初の狭いclassを使う。
対称S2は、行列積を左から記載して

\[
 S_{ABC}(h)=e^{-ihA/2}e^{-ihB/2}e^{-ihC}e^{-ihB/2}e^{-ihA/2}.
\]

X,Y,Zをそれぞれのhalf propagatorとすると、SABC=(XYZ)(ZYX)、SCBA=(ZYX)(XYZ)。
二つは可逆因子による相似変換なので全hでisospectralである。この違いを固有位相改善と数えない。
ABC/CBA、ACB/BCA、BAC/CABの三組をquotientし、代表ABC,ACB,BACのみを残す。
特定Hでさらに偶然の同値性が生じる可能性は残る。将来reference側で診断するが、真値を使って辞書を作り直さない。

各代表jと別代表kで、j,k,jのpalindromeと時間a,1−2a,aを用いる。
a∈{1/8,1/4,1/3,3/8}で24候補、全時間正、各fragmentの合計時間はh。
係数や期待値の結果からaを増やさない。a=1/3の6候補は既知のuniform ordering混合という対照でもあり、二重に新しい候補と数えない。

## 2. 物理符号を含むBCH主要項

\[
S_j(h)=\exp[-ih(H+h^2D_j+O(h^4))],\quad H=A+B+C.
\]

ABC順序について、独立したStrang再帰から

\[
 D_{ABC}=\frac{[A,[A,B+C]]}{24}
 +\frac{[B+C,[A,B+C]]}{12}
 +\frac{[B,[B,C]]}{24}+\frac{[C,[B,C]]}{12}.
\]

他の順序は文字を置換する。自由非可換代数でprod exp(tA)のlogを有理数展開したcubic項の**負**が物理Dである。
(-i)^3=iという符号を保存し、再帰式とは別のexp/log実装で全6順序を照合した。
二次・四次log項が消え、全24macroの

\[
 D_\theta=2a^3D_j+(1-2a)^3D_k
\]

が一致した。これらは係数を代入しないexact identityの検査で、数値性能ではない。
周期macrocycleは同じunitaryを反復し、stepごとに別kernelを無作為にdrawしない。

## 3. 同符号・単純細分化・共有の制約

真のground期待値d_j=⟨ψ0|D_j|ψ0⟩が全て同符号かつ非zero、正時間、独立の加法費用g_jでは

\[
(\sum_\ell g_{j_\ell})\sqrt{\sum_\ell |d_{j_\ell}|a_\ell^3}
\ge\frac{\sum_\ell g_{j_\ell}}{\sum_\ell |d_{j_\ell}|^{-1/2}}
\ge\min_jg_j\sqrt{|d_j|}.
\]

これはレビューのHölder制約を使用したもので、新しい定理と認定しない。
uniform subdivision mではd→d/m²、周期費用g→mgで、主要量J=g√|d|は変わらない。

今回のsharingは隣接する同一fragment exponentialを融合するだけ。noncommutingな再順序化はしない。
単一kernelの反復も外側fragmentの境界を融合できるため、単一の費用は五stageではなく周期四stageの費用を使う。
fragment一回の費用をw_A,w_B,w_C、単一のsteady費用をg_jとすると、cycleの境界融合で
gθ≥Σg_j。理由は各kernelに追加される外側fragment費用の総和を、境界での融合が超えて除去できないためである。
よって、この特定のsharingだけでは同符号制約を回避しない。独立の回路共有改善を現設計の成果と主張しない。

残る判別対象は、(i) 異符号の真の主要energy誤差、(ii) 同一近似stateのjoint幅による不確かさ低減、
(iii) 主要項だけでは説明できない有限hのbias/overlap、(iv) native costと既知の強い対照を含めた条件付き利益。
近似stateで異符号に見えるだけでは(i)を確認したことにならない。

## 4. 真値を使わないstate情報とselector

近似state ρからdtilde_j=Tr(ρDj)を取得し、trace norm幅dρ≥||ρ−|ψ0⟩⟨ψ0|||1が有効なら

\[
|d_\theta|\le |\mathrm{Tr}(\rho D_\theta)|+d_\rho\|D_\theta\|
\le |\mathrm{Tr}(\rho D_\theta)|+d_\rho\|D_\theta\|_{\rm Pauli,1}.
\]

同一ρのため、Dを先に合成してからnormを取る。独立boxの
|dtildeθ|+dρ(2a³||Dj||1+b³||Dk||1)も同じ情報で計算して比較する。
有理Pauli積の位相と係数をexactに保持し、Hermitian Dに虚係数が残れば拒否する。
degree3の展開はfragmentのPauli項数の合計Pに対して素朴にO(P³n)の文字積処理で、候補間のD合成は生成済み項数に線形。
Hilbert-space全state列挙を運用側に要求しないが、P、係数bit長、測定する演算子のbody rankが大きければ安価とは限らない。
pilotではDのPauli項数・最大係数bit長も記録する。多軌道分子RDMの取得やDF回転fragmentへこのtemplateを自動適用しない。

最初のpilotではρ=|00⟩または|11⟩で、Pauli期待値をbitstringから計算できる。
状態情報measurementはこのtoyに限ってzero、準備はX0個/X2個、古典的係数演算と選択の時間を記録する。
これは分子RDM・近似ground-state oracleが無料という仮定ではない。
一般にPauli/RDM測定で各幅η_jを取得するなら、shot数は少なくとも係数scale²/η_j²とconfidenceに依存し、
state準備・measurement grouping・情報取得をどのenergy推定回数へ償却するかが別に必要。
分子state準備/測定/有効gapの実費用はUNKNOWNとして次段階の投資障害を残す。

toy A=sI−aZI−bIZ−jZZ、a,b,j>0のground |00⟩とgap γA=2min(a+j,b+j,a+b)はliteral係数で閉じる。
v≥||B+C||を係数L1で取得し、v<γA/2ならγ=γA−2v>0、
dρ≤min(2,2v/(γA−v))を使用する。
Weylのgap boundと、Q(A−E0)Qの下界γA−vから得る固有vector成分boundであり、full-H eigensolverの出力を使わない。
|11⟩と条件不成立の場合は自明幅2だけを使い、gap未認証と記録する。

各methodをJ²=g²×{empirical |dtilde|, joint幅, box幅}で比較し、同点は固定ID順。
empiricalはcertificateではない。辞書/価格/近似state/幅/sharingは全selectorと既知対照で共通。
全選択を保存しSHAをsealしてから、別workerでreferenceを解く。参考energy、gap、state、真dをselectorへ戻さない。

## 5. 係数で閉じる有限hの保守的bound

Λ=Σfragment係数L1とする。正時間palindromeの複素zで||S(z)−I||≤exp(Λ|z|)−1。
R=1/(4Λ)の円上で右辺<1/3、principal logのnorm<1/2で、log Sはodd。
Cauchy係数boundと幾何級数から、|h|≤R/2で

\[
\|H_{\rm eff}(h)-H-h^2D\|\le K h^4,
\qquad K=\tfrac23(4\Lambda)^5.
\]

有効γ>0、h²||D||1≤γ/4なら、reviewのisolated-ground perturbation boundより

\[
|E_{0,\theta}(h)-E_0|
\le h^2\bigl(|dtilde_\theta|+d_\rho\|D_\theta\|_1\bigr)
+h^4\bigl(K+2\|D_\theta\|_1^2/\gamma\bigr).
\]

有理h/Λ/γ/係数で右辺をexact Fractionとして保存する。float評価値を有理certificateと呼ばない。
非常に保守的で、sharpなfinite-step certificateを獲得済みとは主張しない。
reference前に固定h gridからこの幅がPF予算以下になる最大hも選び、未認証ならnullにする。
このPF energy boundは、ground branchを量子実験で識別する手順、state overlap、統計confidenceの保証とは別。
S4は負時間なので同じΛを流用しない。現在のpilotはS4/C2を数値診断の強い対照に使い、S4の有理剰余保証は出さない。
S4係数のfloat roundingと全数値biasには1e−10の判定marginを入れるが、区間演算のreference証明ではない。

## 6. Controlled費用と公平な比較

nonidentity Pauli Pのsupport kに対して
CU=e^(−iθP/2)e^(+iθ Zcontrol P/2)を使う。
shared basis変換下の別parity ladder templateではRZ2、CX4k−2、X supportごとbasis Clifford2、Y supportごと4。
scalar exp(−iθI)もcontrolの相対phaseとなるためRZ1とphase bookkeepingを保持する。
gate countをcompiled/T/depthの実測と呼ばない。ancillaはこのtemplateでzero、energy計測のcontrol qubitは別に必要。
全methodに同じ隣接融合と周期境界融合を使い、basis共有・routingの追加最適化はしない。

RZ/CX/basis Cliffordを個別に保存する。価格κ=3,6はCXの仮説価格で、one-qubit primitiveを1としたg=RZ+Clifford+κCX。
T費用との対応はUNKNOWN。single/subdivision m2,3,4は同費用macroとの比較を覆う範囲として固定する。
各macroのgate予算内に入る細分化を残し、より高価なbaselineへ勝ったことを主張の根拠にしない。
S4は各代表の五stage Suzuki p,p,1−4p,p,p、p=1/(4−4^(1/3))。
C2は同kernelのm1/m2から(4E(h/2)−E(h))/3を作る。
外挿には二つの推定費用とstatistical weight |−1/3|,|4/3|を含め、無料後処理だけと数えない。

cost proxyは、energy precision εstatでのHeisenberg evolution時間を仮定した
(g/h/εstat+state-prep)/ground-branch overlap。
C2はAi=(gi/hi)/piについて(Σsqrt(Ai|wi|))²/εstatと二つの準備を足す。
confidence overhead、ground branch判別、routing、T synthesisの全費用は未評価。
比較gridから真biasを使って選ぶ最良hのfrontierは**事後oracle診断**で、運用selectorの成功と数えない。
真値なしで選んだcertificate hを別出力にし、情報/剰余によりnullになる場合も保存する。
最終QPE/RPE総費用の改善を実証したというclaimは許さない。

## 7. 方法差の現在の結論と次の停止地点

非isospectralな辞書と、joint状態幅・controlled価格を持つ具体的な生成規則まで定義した。
ただし固定正時間macrocycleそのものは既知compositionであり、このcatalogと情報を既知法にも与えれば生成器は一致する。
sourceのstrong-known-composition欄はその同値性を明示するcontrol。これに対する独立の生成優位を主張しない。
未確定なのは、狭いtoyでの有限h/cost/information tradeoff、単一/細分化/高次/外挿で説明される範囲、分子への有用性。
新しいprimitiveを認めた・独立新規性を確認した、という判断は行っていない。

この準備の次は、[固定scope](qc_c1_macrocycle_preregistration_v1_20261013.md)一回についての**別のユーザー実行承認**。
既存認可を流用せず、source/input/plan/runtimeの固定identityを照合して初めて起動する。
結果/失敗/独立監査を保存・公開後にSTOPし、重要な新結果またはmajor semantics不整合があればGPT科学判断の開始承認へ戻す。
同じ選定レビューや軽微なsource修正ごとの再レビューは求めない。
N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template終了/他Trackの既存認可とSTOP/global manifest既存description FAILを保持する。

### 関連する一次文献の限定確認

形式BCHとapproximate-state energy設計について[Maxwell et al., arXiv:2606.30738v1](https://arxiv.org/html/2606.30738v1)、
periodic ordering/compositionの既知性について[Kronenberger et al., arXiv:2602.18913v1](https://arxiv.org/html/2602.18913v1)の関連節を確認した。
一般先行研究調査・新規性認定を繰り返していない。外挿とC4の版情報は受領レビューの参照を保持する。
本稿のexplicit辞書、sharing制約、保守的Λ boundはCodexの技術的導出として検査範囲を明示し、文献の新規定理と混同しない。
