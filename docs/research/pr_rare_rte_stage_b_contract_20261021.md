# PR希少RTE期待費用：段階B固定科学契約

2026-10-21 JST。先行調査42ed0876を正式入力に、現在ユーザー指示で保存値解析one-shot1回を承認された。SOURCE commitと入力・runtime・承認receipt・新grantのhashを実行前に結合する。旧grantの再利用、retry/resume、段階C/Dの実行承認を含まない。

## 対象と推定量

linear H2/H4/H6/H8、1 Å/STO-3G、tol-only DF3/10/19/28、T=.8/P=0、通常signed/factor1。H2 p1 q1/q4 r2 K4、H4 DF10 p3q1r4K2/p2q8r2K4、H6 p4 q2/q8 r4 K2、H8 p8q2/p7q16 r4 K4。各独立P、保存PF、analytic shots、state/Hamiltonian/compiler identityを維持する。探索・旧union・条件付きpilot・legacy DF12をpoolしない。

各trajectory G=N_cos*RZ_cos+N_sin*RZ_sin。同trajectory両軸の共分散を保って集計する。n>=2で不偏標本分散/nの平方根をSEとし、n=1はSE=null。sample min/maxは観測範囲のみ。global mean exactとpaired SEを保存Pへ再現する（meanはexact Fraction一致、SEは相対1e-12以内）。

## 法則・数値精度・normalization

元rte.pyの a_k=|tau|^k/k!*sqrt(1+tau²/(k+1)²)、B_K=sum a_k、p_k=a_k/B_K、R=qr を用いる。主数値評価は**保存binary64 tauを厳密に取り込んだcanonical有限law**をDecimal100桁で評価する。保存weight/probability/B/logBを残し、元SOURCE定義と比較する。保存floatを正確実数やRNG bit-levelの保証へ格上げしない。

保存categorical pの和と1との差を明示し、そのexact binary64 pを和で割ったsimplexを丸め診断として別に評価する。これは旧保存law・sampler・結果を書き換える操作ではない。主canonical評価との各層mass差<=1e-12をgateとする。保存p/weight/Bとの数値差・signal logBとR logBの差も相対または絶対1e-12をgateとする。これらは数値整合のengineering gateでありinterval certificateではない。

S0_ONLY=N2=N4=0、S2_SINGLE=N2=1,N4=0、S2_MULTI=N2>=2,N4=0、S4_PRESENT=N4>=1。正のbinomial和でmulti/S4質量を評価し小さい差の桁落ちを避ける。K2のS4は支持外。K4のpositive massは未観測でも0にしない。総質量誤差<1e-90。

lambda_Rの独立した新DF/tail生成はしない。保存0次eventの c_m=a0*p_l と保存 |h_l| から実効lambda=|h_l|*a0/c_mを復元し、tau=lambda*(.8/q)/r と全保存0次eventで相対1e-12照合する。この復元は元SOURCEの既存式によるscalar照合であり、独立した全tail係数L1 norm監査とは呼ばない。保存tau/元preparation hash/input bindingを根拠として区別する。新規preparation・Gaussian・回路・DF生成は0。

## 条件付き期待費用・順位感度

E_plugin[G]=sum_observed pi_s*mean_hat_s + sum_unobserved pi_s*mu_s。観測層のmean_hatは経験推定、mu_sは未知母平均でありnull/変数を維持する。K2のsupport外mass0を未知費用0という仮定へ変換しない。

K_obs=sum_observed pi_s*mean_hat_s、U=sum_unobserved pi_s、D=G_PF-K_obs とする。未知層のweighted mixture mean mu_U=(sum_unobserved pi_s*mu_s)/Uは定義であって各層同費用の仮定ではない。U>0なら境界mu_U*=D/Uと元の多変数不等式を出す。D<0は観測mean_hat固定の下で未知費用非負の全域がPF低費用、D>0はmu_U<境界でPR低費用、上側で逆転する。いずれも観測層推定誤差・未被覆component列を含む母平均保証ではない。U=0はplugin比較だけとし、精度保証を追加しない。

保存有限標本mean、層別経験mean、未知muに条件付き期待値、数学的上界、判定不能を別fieldにする。数学的全support compiled上界は新規採択せずnullを保持する。未知muの仮定値や保存最大のupper bound化、候補再選択、shot再設計はしない。

## 独立数値検証

同一one-shot内で別実装を用いる。tauのexact Fractionからinteger isqrtによる130桁有理平方根でlawを評価し、4状態DPで全R eventの質量をexact Fraction合成する。主経路の閉形式/正のbinomial和を呼ばない。質量差<1e-85。費用はinteger raw momentsからexact Fractionのvarianceを求め、主経路の中心化二段階と照合する。mean/varianceはexact一致、SE・境界は相対1e-85以内。数値一致は新しい科学的certificateではない。

## 科学call予算・安全な終了

保存解析pass1、primary条件8、独立数値照合8、保存結果の確率図描画1。新sampling/simulation/signal/SCF/DF/state/circuit/compile/候補選択/旧cohort混合は各0。新markerをO_EXCLで一度消費、各callは実行前に非返還ledgerへ記録。失敗・不一致・実OOM/入力SOURCE/runtime drift・不足定義・予算超過で部分結果を保存しSTOP、同grantのretry/resumeを拒否する。

現在の入力は1,526 trajectoryのJSON scalar/event metadataで、source/providerを起動しない。小さい数値処理と入力decodeの重複を避け、通常OS scheduler上の単一workerで主/独立検証を同じone-shot内に置く。worker数の恒常上限・CPU affinity・RAM/AS/RSS/wall capは設定しない。fresh affinity、MemAvailable、既存cgroup/quota/PSIを読取記録し、一時PSI・他worktree更新では停止しない。共有設定と他ユーザー/別4次PF研究を変更しない。

## 来歴・成果物・公開

GitHubから42ed0876と元rte.py/pr_cost_native.pyを実取得し、先行固定証拠・原P result/trace/axisをhash照合した。raw/seed/内部locatorは非公開作業域に保存する。公開候補artifactは新SOURCE/contract/承認receipt・消費済みgrant digest、結果JSON/CSV、図、独立数値監査、native exit/wait4/reap、旧証拠hash保全、STOP。新GitHub pushは別承認まで0。実行後は段階Bでmandatory STOPし、詳細GPT/Claude review・evidence freeze・C/Dへ進まない。
