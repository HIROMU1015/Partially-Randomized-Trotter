# N3 保存証拠の保証幅：数学契約と独立技術確認

2026-10-12 JST。完了済み[GPT科学レビュー原文](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/review_intake.md)の第7・8節を技術的に確認する。
今回の認可は証拠抽出、式の確認、synthetic検査、比較・実行仕様の準備まで。
固定8入力の新しい保証幅、E_forced、大小・改善率は未評価。研究の採択判断や新規性の確定を代行しない。

## 1. 前提と保証の意味

実対称one-body＋非負unordered density、一外部mode、固定ν、1≤ν≤m。
元の係数trial上界a≤U、外部sector下界C≥cI、g=c−U>0を維持する。
保証対象は基底エネルギーの射影差0≤a−E0≤δ。時間発展normや量子総資源とは別である。

固定κ≥0のK=κh_A−vvᵀについて、保存された証拠がZ≥0、K−τI+Z≥0を与えるなら、
任意のrank-ν射影Πに対し

\[
\operatorname{tr}(\Pi K)\ge\nu\tau-\operatorname{tr}(\Pi Z)
\ge\nu\tau-\operatorname{tr}Z=:s.
\]

従ってσ_ν(K)≥s。元のSchur保証の十分条件は

\[
\delta>0,\quad\kappa\le g+\delta,\quad\delta(g+\delta)\ge q:=\kappa U-s.
\]

PSDの条件はδや比較先cに依存しない。保存κ・μ・τ・Zを変更せず、共通U,cへこの不等式を適用できる。
元のxを保存してκを新しいδで作り直す処理は行わない。

## 2. 固定証拠の下端

δ≥0でf(δ)=δ(g+δ)は厳密単調増加する。
q≤0なら二次条件は自動成立。q>0ならその非負rootが下端となり、box条件δ≥κ−gと合わせて

\[
\boxed{\delta_w=\max\{0,\kappa-g,(\sqrt{g^2+4\max(q,0)}-g)/2\}.}
\]

これは固定証拠に対する必要十分な許容域の下端であり、物理的射影誤差そのものではない。
δ_w=0はδ>0契約におけるinfimumとして記録する。数値証拠は正の有理数δ_proofを別に保存する。
κ、δ、gはenergy、K・s・qはenergy²。単位と正のscale変換が整合する。

NORMはκ=0、s=−β²（β²=||v||²）で同じ式になる。
FIVE/OLD-DIRECTの保存PSD証拠はK−μI≥0からs=νμ。
旧証拠のOCC埋込みτ=μ,Z=0は同じ幅となり、fallbackを新しいOCC利益に数えない。
この確認は既存式の利用条件の技術確認であり、新条件族の新規性判断ではない。

## 3. 厳密な実装・検査

実装は`n3_saved_width_v1.py`。入力はFractionのみ、float/boolを拒否する。
box下端が既に二次条件を満たす場合は正確な有理数を返す。
判別式の分子・分母がともに平方数なら正確な有理root、それ以外はinteger isqrtで平方根を128 fractional bitsの外向き区間へ囲む。
δ区間幅は最大2^-129（保存係数のenergy単位）。結果を見たbit増加・追加精度段階はない。

独立したendpoint checkerは平方根を再計算せず、多項式だけで確認する。
上端hはh≥max(0,κ−g)、h(g+h)≥q。
下端l≤真の下端は、l≤max(0,κ−g)またはl(g+l)≤qで確認する。
正のδ_proofについてもboxと二次不等式を厳密確認する。
集合の最小幅は[min_i l_i,min_i h_i]、安全側上端の証拠IDを残す。

OCC区間上端<BASE区間下端だけをstrict改善とする。逆向きも同様。
両方が同じ一点有理区間ならEXACT_EQUAL_WIDTH。それ以外の重なりは未判定とする。
表示丸め、epsilon tolerance、float scoreによって勝敗を決めない。

rational operand/result cap4096bits、root scaling用explicit integer cap8192bits。
Fraction内部temporary integerは完全監視しない。独立endpoint検査はexact Fraction演算であり、途中全演算のbit長統計とは区別する。
source内のcap超過はTECHNICAL_STOPとして保存し、精度・候補・入力を増やして再実行しない。

## 4. 改善機構の式の確認（実8入力への数値適用なし）

β>0、u=v/βについて、uを占有するrank-ν射影の最小一体energyは

\[
E_{forced}=u^Th_Au+\sigma_{\nu-1}(h_A|_{u^\perp}).
\]

射影をuuᵀとu⊥上のrank-(ν−1)射影へ分解すればこの表示になる。
これはactiveの相互作用込み基底energyではない。真のmany-body状態・gapを入力にしない。

φ(κ)=σ_ν(κh_A−vvᵀ)−κUは、コンパクトなrank-ν射影集合上のaffine関数の最小値なので凹。
κ=0の最小化射影はΠu=uを満たす。正κの最小化射影の部分列極限と、κ=0で最小傾きを与える固定trialを使う上下挟みから
φ'(0+)=E_forced−Uとなる。
E_forced≤Uならその固定trialからφ(κ)≤−β²。NORMより小さい幅は理想OCC族でも認証できない。
E_forced>Uなら十分小さい正κでφ(κ)>−β²。NORM rootでは正marginを得てboxもstrictに満たせるため、連続性で少し小さいδを認証できる。

境界条件はg>0、β>0、1≤ν≤m。β=0ではNORM下端0なのでstrict改善の主張から除外する。
ν=1でもNORMとの差はあり得るがOCC族と旧LMI族が一致するため、強い既存対照との差分にはならない。
ν=mや固有値縮退も、射影集合の変分表示・片側微分の議論で扱う。微分可能な最小固有vectorを仮定しない。

この機構はNORMとの差の解析であり、FIVE/OLD-DIRECTを含むBASEへの優位を導かない。
固定8入力のE_forcedや追加固有値は計算しない。有限保存証拠がこの理想条件を満たすか、発見できるかも別問題。

## 5. 既存原理との関係・次の判断の材料

Ky Fan型trace最小化は既存原理。[Liang et al.の一次資料](https://arxiv.org/abs/2104.00257)の公開abstractは、その原理の一般化を扱うと明記している。
Schurによる厳密固有値評価も既存研究がある。[Dusson et al.](https://arxiv.org/abs/2105.02058)の公開abstractはFeshbach–Schur法による固有値・固有関数評価を扱う。
今回確認したのは公開書誌・abstractの位置づけであり、全文の網羅的な先行性監査ではない。現在のoccupation射影へ各論文の定理を無条件で移植しない。

比較仕様で未解決のまま残す論点は、(a)有限保存集合でBASEより狭い幅が残るか、(b)旧証拠coverageの限界に依存するか、(c)その幅差を使う研究taskが具体化できるか、(d)原理の組合せを超えた非自明な差分が既存研究から独立か、である。
これらの科学的価値・継続判断は新結果後のGPT判断事項。Codexは今ここで中心テーマを採択しない。
