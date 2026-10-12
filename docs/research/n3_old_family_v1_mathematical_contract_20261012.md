# N3旧条件族診断 v1：数学・比較・実行契約

2026-10-12 JST。[完了済みGPTレビュー](../../artifacts/n3_old_family_v1/2026-10-12/preparation/review_intake.md)第7・11節を具体化する。
準備・syntheticは認可済み。**S1二例への新しい診断は別の明示実行認可が必要。**
中心テーマ未採択、元科学結果・gate・STOP、N1停止・N2保留は保持する。

## 固定対象と保証の意味

既存のS1_edge_pair/S1_distributed_threeだけ。実対称one-body＋非負unordered density、star、full10→active9、ν=4、一外部mode8をemptyへ射影、scale1.25、元δgoal=.005。
化学geometry/basis、DF rank/policy、L_D、PF/time-evolution窓は適用なし。8入力からのposthoc developmentであり、新規独立移送ではない。
同じ係数・v・保存trial上界U・保存c_star・OCC勝者を保持する。
δは各保存OCCの**正の有理上端positive_proof_delta=highそのもの**。丸め小数・レビューの説明用δ_reqは使わない。
[固定入力索引](../../artifacts/n3_old_family_v1/2026-10-12/inputs/index.json)は元pack/ledger/幅JSON/証拠IDへのpointerとSHAを保持する。
保証は基底エネルギー射影差0≤a−E0≤δ。真の誤差、量子回路費用、追加mode削減を診断しない。

## 旧族の一変数表示と独立確認

g=c−U>0、D=g+δ>0。旧条件は

\[
0\le\kappa\le D,\quad \kappa h_A-vv^T-\mu I\succeq0,
\quad \kappa U-\nu\mu\le\delta D.
\]

最後の条件はμ≥μ0=(κU−δD)/νに等価。任意の可行μをμ0へ下げると(μ−μ0)Iを加えるためPSDを保持する。
逆にμ=μ0を採れば旧条件を満たす。従って保存されなかったμを含めた旧族の可行性は

\[
M_\delta(\kappa)=\kappa(h_A-U I/\nu)-vv^T+(\delta D/\nu)I\succeq0,
\quad\kappa\in[0,D]
\]

に等価。M0=Mδ(0)、M1=Mδ(D)。κ,U,c,δはenergy、μ,Mはenergy²。
実装のproducerはこのaffine形、独立checkerは元h_A/v/μ0から再構成する。

**回収**：有理κ∈[0,D]とμ0に対し、元行列の有理PSDが成立。浮動小数の最小固有値≥0だけでは回収にしない。

**分離**：有理W≽0、trW>0、tr(WM0)<0かつtr(WM1)<0。
traceはκのaffine関数なので区間全体で負。PSDなM,Wのtrace積は非負であるため、旧族のどのκ/μも可行でない。
Wの非零性はPSD＋正traceで確認する。零trace・片端の零margin・非PSDのWは拒否する。

**NORM**：同じv/U/c/δでδD<β²=||v||²を別に確認する。
NORMが通る場合、旧族separatorがあってもNORMを含む対照全体との分離には分類しない。
**FIVE⊂旧族**：凍結元sourceのgenerate_method(FIVE)はκ=vᵀ(h_A−λI)⁻¹v、μ=κλをold_candidateへ渡す。
old_candidate/marginsは上の同じPSD・box・affine条件を使う。FIVEの有効な証拠は旧族の可行対なので、全旧族分離はFIVEも包含する。
この対応はsourceの静的確認であり、S1で新たなFIVE候補を評価しない。

## 候補生成と有限予算

順序を結果前固定する。

1. vとcoordinate方向のrank-one W=yyᵀ/||y||²を検査する。両端traceをcacheし、二方向の凸混合を有理の開区間不等式で検査する。
2. κ=D*j/16（j=0,...,16）のexact primal PSDを検査する。
3. 未解決の場合のみNumPy eighの最小固有値をscoreとする一変数golden探索（60更新＋初期2点）。
   abscissa・最小固有vectorを40-bit dyadicへ量子化し、exact primalまたはrank-one/混合dualを検査する。

数値scoreが正・負・solverが終了したことは数学的分類に使わない。scoreはconcaveなλminの候補生成だけ。
producerのPSDはexact Schur消去、独立checkerは凍結旧Auditの元行列entryからのLDLと全行列再構成を使用する。
checkerはproducer/NumPyをimportせず、独立にμ0・両端trace・NORM・入力identity・費用event数・STOPを確認する。
過去のsecondary_separatorの方向・混合原理と、既存Audit/MemoryWatchを再利用し、元source/run()は変更・再実行しない。

per-case上限：eigh64、primal PSD96、ray80、混合scalar1024、explicit有理演算2,000,000、rational4096bits。
分母0/bool/float/非対称/次元不整合/oracle fieldを拒否する。Fraction内部temporary整数すべての監視とは区別する。
同一abscissa・rayはcache。混合上限でそのsubstageを打ち切っても、残りの固定primal探索は維持する。
保存OCC/旧U,cの監査は既存proof replayであり、上の「新primal提案96」には含めずchecker費用を別記する。

## 実行・資源・保存

CPU32、Python3.12.3/NumPy1.26.4。2 spawn workers、BLAS/OMP/MKL/Numba/Rayon各1。GPUなし。
支配費用は小行列の有理PSDと保存proof再構成。両端trace・abscissa cache、analytic優先、独立2入力の並列を使用する。
worker AS2GiB、runner＋descendantsのsampled RSS4GiB、開始前available6GiB以上。
JSON shard32MiB、全output64MiB（通常62MiB＋停止記録予約2MiB）、failure64KiB、stdout/stderr log8MiB。
wall/CPU time capなし。MemoryWatchは50ms目標samplingで、間の瞬間peakを測定したとはしない。
source/input/planに結び付く新authorization、専用execution branchのclean worktree、fresh diagnostic1を必要とする。attempt1/retry0。
元saved-widthの消費済み認可はschemaで拒否する。ここではauthorization templateの認可flagをfalseのまま保持する。
source/hash/input identityとschemaだけのpreflightは認可不要。S1でM/κ/μ/W、PSD、numeric scoreや採否を新評価する経路は認可後だけ。

新入力、U/c再計算、OCC再探索、E_forced固有値、真ground/gap、compile/shots、他base、元batch再実行、N1/N2再開は0。
有限探索の不成功はUNDETERMINED。bit/operation/memory/出力制限や異常計算はTECHNICAL STOP、証拠不整合はSOUNDNESS STOP。
失敗・部分結果を保存し、精度・予算・対象を増やす自動再試行をしない。全結果後Mandatory STOP。

## 結果の判定と次の科学判断

回収、厳密なNORM＋旧族分離、旧族だけ分離してNORM回収、有限未判定、技術停止を区別する。
回収が得られてもOCC全体を無価値としない。分離が得られても20.424%/0.343%を最適旧族との幅差へ読み替えない。
この固定幅診断だけでは旧族の最小幅、全既知法への優位、汎化、新規性、量子資源削減を採択しない。
新しい回収/分離結果または未判定後の情報価値はGPT科学判断事項。通常のsource/テスト/公開完了だけで同内容の詳細レビューを追加要求しない。
前提・意味論・比較公平性に実質的な不整合が見つかれば、実行前にもGPTへ戻す。
