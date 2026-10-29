# 通常CDF7：限定1-fitでH4のDF物理予算PASS

2026-10-30 JST。**CDF7_DF_ELIGIBLE、25/25 logical units COMPLETE、correctness FAIL0、初期化1件・optimizer1回、retry/resume0**。保存直鎖H4・1.00 Å・STO-3G、Nα=Nβ=2／sector36に対し、通常number CDF7の終了sector誤差＋engineering guardは **3.022641747295844e-7 Ha**、予算 **.00019875 Ha**を満たした。初期値は9.064703524309164e-4 Haで予算外だったが、表現correctness通過後の1-fitで改善した。

これは一つの初期化・目的・solver条件におけるDF数値pilotの適格性であり、PR性能、最終CA、回路資源、rank7の最小達成誤差を示す結果ではない。

## 1. 許可範囲・SOURCE・入力

ユーザーは技術準備・人工検証・SOURCE/契約固定から、rank7初期化1件、1-fit、初期／最終DF評価、結果公開までを一括許可した。追加承認・承認証明・新grantは要求していない。[Shared-basisの非実行契約](research_c_shared_basis_df_contract_unapproved_20261029.json)はexecutable=false／call0のまま保全し、別のCDF7契約を新設した。

| 項目 | 固定値 |
|---|---|
| Branch | `codex/research-c-cdf7-onefit-20261030` |
| 起点 | `0876449d7ce75e25277865834efbc3bafaed7b3e` |
| 公開科学SOURCE | `c13d6f82a22d497b3334eaaa40d8430a5010f70f` |
| [科学契約](research_c_cdf7_onefit_contract_20261030.json) SHA256 | `e1b4f411043154663b29274e1ed4657b248203bad172cd074a21edf9258d5f41` |
| [SOURCE35 manifest](../../artifacts/research_c_cdf7_onefit/2026-10-30/technical/SOURCE.json) SHA256 | `a288dc933e4bfac3e8fdd562d644073358e180ecc41b9c0694a00d5bdba699c6` |
| 元P-H4入力 SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 入力来歴 | commit `86b8b002704aced8ca239414080f53793de81658`の元snapshot bytesを[new input copy](../../artifacts/research_c_cdf7_onefit/2026-10-30/input/P-H4.npz)へ保全 |
| Fresh output | `artifacts/research_c_cdf7_onefit/2026-10-30/run01` |

原inputはbyte同一コピーで、PySCF/SCF・integrals再生成なし。既存whitelistでchemist h/g・nuclear scalar・MO係数だけを読み、packet内の旧演算子・energy等を新評価へ転用していない。物理Hを一回構築してbefore／initial／finalのsector評価で共有した。

科学SOURCEをcommit/pushしremote SHA/treeを照合してから一度実行した。[Preflight](../../artifacts/research_c_cdf7_onefit/2026-10-30/execution_preflight.json)でSOURCE35・input8・旧保護2153件、依存版、fresh namespaceと空き資源を確認。科学実行後もSOURCE・入力・旧結果は不変で、run01の再起動・上書き・部分resumeはない。

## 2. 初期化・最適化・再利用：Q1／Q2

[技術準備](research_c_cdf7_onefit_preparation_20261030.md)の専用module/CLIは既存`c1_molecule_extension.factor`、`c_pr_aware_df`、corrected codec、SectorCAR／rank_assessment／representation_gateを使用する。新fitter、loss、gradient、sector Hamiltonian構築器は作っていない。旧runner・旧科学SOURCEも編集していない。

初期化はffsim0.0.83の明示的Cholesky、`max_vecs=7/tol=1e-8/cholesky=True/optimize=False`の一件。**実際に7 frame、J/Uともfloat64 `(7,4,4)`**を得た。一般実対称J／実直交Uを保持し、修正`real_schur_column_gauge_v1`で112変数（回転42＋J上三角70）へ変換した。Optimizerにはそのqualified x0をそのまま渡し、二度目のCholesky生成・codec再packはしない。Rank8からの切断や、8 frameを残した実効7の偽装ではない。

元gから補正したκ・核scalar一回・interleaved spinを維持。保存frameは0..6、元順prefix LD2／tail2..6である。初期budgetは**診断値のみ**とし、初期予算NOT_METでも表現／codec correctnessが通ればfitへ進む。旧Stage Aの初期予算gateを変更・回避して再使用したわけではない。

目的は通常CDFの既存`f=.5||g−g_DF||F²`を、正の固定初期scaleで割ったγ0の目的`f/f*`。今回の **f*=1.3366179678850646e-7 Ha²**（floor1e-14より大）を一度固定した。初期smooth massは既存scale helperのmetadataにあるが、γ0なのでpenaltyを加えていない。Tail-mass／basis-switch正則化、追加gradient閾値、rank/γ/solver探索は0。

L-BFGS-B、options=None、callback=None、jac=True、追加SciPy tolなし。ffsim tol=1e-8は初期化の設定である。固定SciPy1.14.1の既定ftol=2.220446049250313e-9／gtol=1e-5等を変更していない。正規化は絶対gradient・目的停止scaleに影響するため、同じrank8のraw目的やgradientとそのまま比較しない。

## 3. 初期／終了のtensor・物理精度：Q3／Q4／Q5

| 指標 | Cholesky before | Codec後 initial | 1-fit final |
|---|---:|---:|---:|
| Actual rank | 7 | 7 | 7 |
| Tensor Frobenius | 5.170334549880623e-4 | 5.170334549883050e-4 | 4.6164582354636235e-7 |
| Tensor max-entry | 1.7717216565062144e-4 | 1.7717216565073246e-4 | 6.00951019526933e-8 |
| `.5*Frobenius²`（NumPy） | 1.3366179678844632e-7 | 1.3366179678857182e-7 | 1.0655843319889956e-13 |
| Sector operator誤差 [Ha] | 9.064673381480225e-4 | 9.064673381493577e-4 | 2.992498412460217e-7 |
| Numerical guard [Ha] | 3.014281558702417e-9 | 3.0142815587024166e-9 | 3.0143334835627005e-9 |
| **誤差＋guard [Ha]** | **9.064703524295812e-4** | **9.064703524309164e-4** | **3.022641747295844e-7** |
| 予算 .00019875 Ha | NOT_MET（診断） | NOT_MET（診断） | **PASS** |
| General-K/raw CAR max-entry | 3.219646771412954e-15 | 4.107825191113079e-15 | 4.440892098500626e-15 |

3件ともsector36の一般K/raw CAR表現gateを通過した。Codec前後のsector operator差max-entry=2.6645352591003757e-15、Frobenius=1.4597259664520713e-14、operator norm診断=6.267474045098188e-15。登録component allowance1e-10／rtol0内で、初期tensorも保持した。Codecのframe別det・列符号・projector・Schur／exponential残差は[codec.json](../../artifacts/research_c_cdf7_onefit/2026-10-30/run01/prepare/codec.json)へ保存した。

DF判定は**物理sector operator norm＋guard**による。Tensor残差からPASSを推測したわけではない。Guardは既存`sector_operation_scale_guard8_proposal_v1`で、n=4/R7/d36とphysical・各generatorのoperation scaleを使うengineering値。厳密rounding enclosureや最終energy CA保証ではない。今回はfull256回路同値性やcontrolled operatorを新評価していない。

## 4. Optimizer停止・gradient：Q6

| 指標 | 実測 |
|---|---:|
| success／status | true／0 |
| nit | 215 |
| nfev／njev | 227／227 |
| 初期正規化objective | 1.0 |
| 終了正規化objective | 7.972243061464537e-7 |
| 終了raw objective（ffsim） | 1.0655843320300536e-13 Ha² |
| 初期正規化gradient L2／∞ | 5591.464490256435／2897.6006688882962 |
| 終了正規化gradient L2／∞ | .17106638509259028／.055562141116287024 |
| 終了raw gradient L2／∞ | 2.2865040401590195e-8／7.426535615019476e-9 |

生messageは **`CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`**。目的値の相対減少停止であり、正規化gradientはSciPyのgtolまで小さくなっていない。Successを停留点到達・大域最適性・最小達成可能誤差と呼ばない。Rawとnormalizedのgradientを区別し、追加iteration・再最適化は行わない。

開始／終了xと再評価gradient、optimizerの**生x/jac**、fun/status/message/nit/nfev/njev、最終J/UをNPZ／JSONへ保存した。初期／終了objectiveの独立tensorとの整合、returned jacとの整合も登録条件を通過した。取得済みNumPyとffsimの微小な残差差を隠すための値置換や閾値変更はない。

## 5. 保存CDF8との比較：Q7

下表のrank8は[保存Stage A](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)のJSONを読み取り専用で引用した。旧J/U再decode、DF norm・tensor残差の再計算、再fitは0。詳細は[5候補状態のCSV](../../artifacts/research_c_cdf7_onefit/2026-10-30/presentation/DF_comparison5.csv)。

| 終了候補 | Rank | Tensor Frobenius | Sector誤差 [Ha] | Guard [Ha] | DF予算 | Optimizer |
|---|---:|---:|---:|---:|---|---|
| 新通常CDF7 γ0 | 7 | 4.6164582354636235e-7 | 2.992498412460217e-7 | 3.0143334835627005e-9 | PASS | success／215 iter |
| 保存通常CDF8 γ0 | 8 | 5.607958698344648e-9 | 7.018855038843621e-9 | 3.278781198621627e-9 | PASS | success／97 iter |
| 保存PR-aware CDF8 γ.1（参考） | 8 | 6.285301032493658e-9 | 9.039161828383144e-9 | 3.2564531608037766e-9 | PASS | success／1129 iter |

保存CDF8のSOURCEは`e1085c95f5c01b68566e17d543ace5832cb94061`。初期値は既に最適化された保存CDF8を修正codecへ通したもので、新CDF7はrank7専用Choleskyの未最適化初期値である。CDF8のf*=2.1839632053132894e-9 Ha²も今回と異なる。Raw tensor loss、相対目的停止を使う点は共通だが、同一初期条件・同一正規化ではない。

今回分かったのは、**rank8から7へ変えた一つの構成でも、固定DF予算を満たす通常CDFを既存実装の1-fitで取得できた**こと。終了誤差が保存CDF8より大きいことを、rankだけの純粋な因果効果やrank7全体の限界と解釈しない。予算内だからといってPRやcompiled価格が改善するとは限らない。

最終適格時の登録補助診断は、LD2のexact tail mass **λ_R=2.200160822452493 Ha**、tail frame2..6、exact tail basis keyは5個。キーはraw resultへ保存した。これはlaw／PF／減衰／価格を新評価した値ではなく、係数と厳密keyの補助値である。Massやframe数だけからshots・native／総費用を補完しない。

## 6. 後続PRの最小技術設計・未承認案：Q8

後続では保存された**本CDF7 final.npz**を使用し、新fitは0とする。既存B1の一般K／spin対応basis／tail抽出、SectorCAR、修正済みtail component-record、部分PF／paired-RTE、RPE、affine損益分岐を再利用する。旧rank8 validator・factor9ループ・candidate/provenance固定のrunnerをそのまま再起動してはいけない。Rank7専用のinput/evidence/ID・8因子ループへの薄い接続と人工回帰が必要である。

[未承認PR契約案](research_c_cdf7_PR_contract_unapproved_20261030.json)へ、新exportのhash、元input、DF予算PASS、候補1件（LD2、partial S2 s1/r128/K0、δ=h=.1、m0..12）、共通CA／phase/failure配分、own-DF oracle／状態準備除外、比較対象、STOPを具体化した。

推奨する最初の科学範囲は**最大119 logical units、nativeなし**。

| 後続の提案段階 | 内訳 | Units |
|---|---|---:|
| Reference | 保存evidence/input/fit/構造、candidate、sector、8因子表現、和、own-ground/alias、law/tail、target/kernel、13 exact＋13 round＋13判定、summary | 62 |
| RPE | 3 laneの資格3＋13×3=39＋energy3＋shots summary3 | 48 |
| 損益分岐／判定 | affine3＋2仮定×2比較4＋将来nativeの必要性判定1 | 8 |
| 最終summary／STOP | 1 | 1 |
| **合計** | **candidate固有PF/RTEを再生成し、旧値を移送しない** | **119** |

比較は保存通常CDF8 γ0と保存PR-aware CDF8 γ.1 M4の同s1/r128/K0とする。既存CDF8の保存shell／summaryを再利用するが、CDF7のPF/RTE・ground・native平均価格を旧値で埋めない。Eigenpairはnew own-DF oracle用に必要であり、状態準備費用を含まない。

登録13 roundが不十分なら候補の限定的不適格を保存し、無断でs/r/Kを変更しない。条件付きenergy/shots/Wと有限正のRZ損益分岐まで確認してから、fixed32 native校正が必要かを別途判断する。Evidence laneの未知合成／law誤差はnullのまま、ideal／assumed laneは仮定を明示し、CA実達成を主張しない。

Native32は上記119-callに**含めない**。後の別契約候補は、base＋7 frameのraw8／Gaussian8へ対応した既存native経路で**最大323 units**、32×128 events、独立occurrence、full512 raw/compiled・actual adjoint・complex X/Y・P0、QPY128。同じprecision・head相対融合・tail exact-key共有、cross-head/tail融合なしを維持する。新cohort／seedとSOURCEを先に固定し、旧cohortをpoolしない。今はnative価格・shots・損益分岐・総資源を計算していない。

**本119-call JSONは技術設計案であり、`executable=false/implementation_ready=false`。** Rank7接続adapterと人工回帰、比較input/shellの全binding、科学SOURCEの最終固定は次の技術作業として残る。実行可能SOURCEが完成したという意味にはせず、別途明示許可なしにPRを開始しない。今回完了した1-fit SOURCEやShared-basisのcall0契約を、そのPR実行契約へ書き換えない。

## 7. 未達ならどうするか：Q9と今回の判断

今回の最終候補はDF適格なので、NOT_METに対する再fitは不要で、追加rank/γ探索へ進まない。一般に未達だった場合も、同じ初期化・scale・solverにおける一回の限定結果として、停止状態とraw gradient／物理誤差の関係を調べることが先決で、rank7不可能性や追加fitの高い優先度を自動認定しない。

現状はPR評価用の具体的な保存候補を得たため、**119-callのown PF/RTE・RPE／損益分岐検証には情報価値がある**という技術提案を返す。正式な採択・科学許可はユーザー／研究Cが判断する。Nativeや同CA総資源優位を予告しない。

## 8. Tests・call・資源・保全

人工・回帰は**新8＋既存37＝45 PASS、29 subtests PASS**。実ffsimのrank7初期化／qualified x0／実loss/gradientを用いてoptimizerをmockで遮断し、nit0で接続を確認した。人工end-to-endでは既存SectorCAR・物理H・representationをmockせず、初期予算未達からの継続、最終適格／予算未達／optimizer不成功、codec/表現FAIL時のSTOP、fresh再使用拒否、call25/24の計上を確認。一般Kのpassive-import回帰は別processでPASS。これはCI／外部独立再現の証拠ではない。

科学実行はPrepare6＋Fit5＋Evaluate13＋適格時aux1＝**25 COMPLETE／25 attempts／FAIL0**。[Kind別表](../../artifacts/research_c_cdf7_onefit/2026-10-30/presentation/calls.csv)、[親・prepare・fitのjournal/STOP](../../artifacts/research_c_cdf7_onefit/2026-10-30/run01/)。内部215 iteration／227 evaluationを追加fitとして数えず、実optimizer invocationは1。新rank・初期値探索・再fit・fallback・PF/RTE・ground・sampling・実回路／shots／総費用は0。

開始時affinity32 CPU、空きRAM約52.58 GiB、CPU/memory pressure avg10=0、load約1.09。初期2 snapshotを既存resource plannerで2並列、BLAS1で自job内のoversubscriptionを調整。Optimizer1回は逐次依存のため一process。Task-local `.venv`／`.venv-fit`とcacheを使い、既存依存をread-onlyで再利用し、共有設定・他jobは変更していない。CPU/RAM/worker/wallに新たな人工上限は追加しない。

[GNU time](../../artifacts/research_c_cdf7_onefit/2026-10-30/process_time.txt)：wall **3.72秒**、user19.14秒、system7.16秒、max RSS371032 KiB、swap0、exit0。Runner wall3.44149秒／process lifetime CPU9.70489秒／parent peak RSS288296 KiB。GNU time max RSSはこの計測の最大値で同時processの合算peakとは呼ばず、実測比較なしの高速化倍率も主張しない。

[実行summary](../../artifacts/research_c_cdf7_onefit/2026-10-30/execution_summary.json)、[postflight](../../artifacts/research_c_cdf7_onefit/2026-10-30/execution_postflight.json)へSOURCE35・input8・旧保護2153件のhash不変、raw34件／64280 bytes、科学call、未資格nullを保存。表示CSVは登録済み値の転記で、科学runnerの再起動・追加operator評価は行っていない。旧H2/H4/H6・grant／ledger／marker／STOP、Shared-basis非実行案も変更していない。

結果・raw・SOURCE・契約・STOP・後続技術案をcommit/pushし、remote SHA/tree・SOURCE/input・成果物を照合後**mandatory STOP**。次の担当は、**ユーザー／研究C**がPR119-call案の準備・実行へ進むか判断し、**Codex**は具体的指示と必要な許可後にrank7の薄い接続とSOURCE固定を行うこと。今回の許可で後続PR、native32、追加fit、別rank／γ／η／分子、詳細GPT科学レビューを開始しない。
