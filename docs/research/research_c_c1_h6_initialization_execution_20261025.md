# 研究C・C1：H6 CDF初期値変換の検証・保存値限定再解析

2026-10-25 JST。起点 `c7131bbbf6be10be3c1842344e4f2ad3a225a774`。
固定した新科学SOURCE `14cf35589a8501ef5db9a33700ed0c98bc1aa853`。
Branch `codex/research-c-c1-h6-initialization-20261025`。

**保存H6の旧初期化経路を再現すると、rank14／18とも、予算内の初期XDFが実パラメータ変換直後に予算外へ変わった。初期J/Uは保存XDFの対応prefixと成分まで完全一致した。従って初期変換の不忠実性の関与は実データで確認できた。ただし、最終予算未達の唯一の原因、修正後fitの成功は未確定。35/35 logical units、新fit・optimizer実行・retry/resume0、公開照合後mandatory STOP。**

[6状態比較](../../artifacts/research_c_c1_h6_initialization/2026-10-25/state_comparison.csv)／[frame別det・branch・残差](../../artifacts/research_c_c1_h6_initialization/2026-10-25/frame_diagnostics.csv)／[初期値identity](../../artifacts/research_c_c1_h6_initialization/2026-10-25/initialization_identity.csv)／[gradientの資格](../../artifacts/research_c_c1_h6_initialization/2026-10-25/gradient_diagnostics.csv)／[call](../../artifacts/research_c_c1_h6_initialization/2026-10-25/calls_summary.csv)／[要約](../../artifacts/research_c_c1_h6_initialization/2026-10-25/execution_summary.json)／[全原本・パラメータ・STOP](../../artifacts/research_c_c1_h6_initialization/2026-10-25/run01/)／[契約](research_c_c1_h6_initialization_contract_20261025.json)／[SOURCE・人工40 tests](../../artifacts/research_c_c1_h6_initialization/2026-10-25/technical/)／[旧H6報告](research_c_c1_h6_execution_20261025.md)。

## 範囲と来歴

ユーザーが技術検証から保存H6の限定数値再解析までを明示許可したため、途中の再承認・技術完了STOPを設けず、人工検証→SOURCE/入力/35-call契約固定・公開→fresh run01の順で実施した。承認原文保存・承認証明書・新grantは作成していない。

直鎖H6・1.00 Å・STO-3G、中性singlet、Nα=Nβ=3、sector400。既存h/g/constant、通常number CDF/XDF、元g補正κ・scalar・元frame順を保持する。CA0.00159 Ha、DF予算CA/8=0.00019875 Ha、`sector_operation_scale_guard8_proposal_v1`を無変更で使った。PFのL_D split・時間窓は今回非適用で、PF/RTE、ground/alias、回路・shots・総費用を評価していない。厳密CA資格も今回の対象ではない。

旧科学SOURCEは `33c380d8aa792e5119bb3adec5dd07d1e1377214`、保存結果commitは起点c7131bbbf…、保存input SHA256は `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447`。旧H6/H2/H4のSOURCE・結果・grant・ledger・marker・STOPは変更していない。

## 初期分解の同一条件を先に確認

固定ffsim0.0.83の`_double_factorized_compressed`は、`_double_factorized_explicit_cholesky(g,tol,max_vecs)`で初期J/Uを作り、`df_tensors_to_params(...,real=True)`でx0へ変換する。旧H6の`factor`はtol1e-8、L-BFGS-B、options=None、callback=None、jac=Trueであり、H4の別callerにあるoptions={}を混入させていない。

保存`XDF_full.npz`はmax_vecs21・実rank18である。一方、比較する旧CDFはmax_vecs14／18であるため、上限が同一だと仮定してはいけない。今回は同じ固定関数・元g・tolで**非最適化initializerを14／18の2回だけ再生成**し、保存fullの対応prefixとのJ/U、列符号、rankを確認してから変換を評価した。rank14では保存`XDF_selected.npz`とも照合した。

両rankともJ最大成分差=0、U最大成分差=0、actual rankは14／18だった。rank18ではmax_vecs21と18の上限差がこの保存初期分解を変えなかったことを実際に確認した。異なるrankの初期値と終値を混ぜていない。

## 同じrankの変換前・直後・保存終了後

tensor残差は元gとの差のFrobenius norm、目的関数は固定ffsimと同じ`0.5*||g-g_DF||F²`。sector誤差は既存`SectorCAR`／`rank_assessment`を用いた元物理Hamiltonianとの差のoperator normで、κは元gから補正したままである。

| rank | 状態 | tensor残差F | tensor目的関数 | sector誤差 Ha | guard Ha | CA/8判定 |
|---|---|---:|---:|---:|---:|---|
| 14 | 初期XDF・変換前 | 7.729927307e-5 | 2.987588809e-9 | 1.024738549e-4 | 1.138452778e-6 | PASS |
| 14 | 実パラメータ往復直後 | 9.428193023e-1 | 4.444541184e-1 | 2.354946984 | 1.138452778e-6 | NOT_MET |
| 14 | 保存CDF終了後 | 8.748970878e-4 | 3.827224571e-7 | 1.756118664e-3 | 1.195237099e-6 | NOT_MET |
| 18 | 初期XDF・変換前 | 1.421020959e-9 | 1.009650283e-18 | 1.731868468e-9 | 1.366146020e-6 | PASS |
| 18 | 実パラメータ往復直後 | 9.428200512e-1 | 4.444548244e-1 | 2.354979839 | 1.366146020e-6 | NOT_MET |
| 18 | 保存CDF終了後 | 2.927656914e-4 | 4.285587503e-8 | 4.679546622e-4 | 1.448732723e-6 | NOT_MET |

判定はsector norm+guard≤CA/8である。tensor normとHaのsector normは別の量として記録し、互いの代用にしていない。保存XDF14／CDF14／CDF18の3つのsector normは旧scalar結果とも固定allowance内で一致した。保存最終objectiveとの差はrank14で約9.09e-20、rank18で約3.54e-20だった。

両rankとも初期XDFは予算内だが、旧codec後には大きく悪化する。保存optimizerはその悪化したx0から目的関数を大きく減らしている一方、元の初期XDFより悪い終了点を返していた。この事実は「optimizerが全く改善しなかった」という説明とは異なる。

## det・logm branch・実反対称生成子

旧bridgeは[SciPy1.14.1の2次元logm](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.logm.html)をframeごとに呼ぶだけで、分岐選択・実部射影は追加しない。その後の固定ffsimの`antihermitians_to_parameters(real=True)`がlogの上三角実部だけを採り、unpack時には対角0の実反対称行列を作って指数化する。

実反対称Aのexp(A)はdet=+1だが、eigh由来の元UはO(n)でありdet=-1も含む。rank14の7 frame、rank18の9 frameがdet<0だった。J・rank・float64 dtypeは変換で保たれたが、回転とdensity projectorが保たれず、tensorが変わった。

さらにdetだけでは切り分けが足りない。両rankに共通の**frame9（0-based）はdet≈+1**だが、次の値を観測した。

| frame9の指標 | 値 |
|---|---:|
| 元Uのorthogonality max-entry | 4.44e-16 |
| eigenvalueと−1の最小距離 | 1.94e-14 |
| logmの虚部Frobenius norm | 0.01224596469 |
| 実部の反対称性残差F | 0.2456464545 |
| exp(logm(U))対Uのmax-entry | 3.22e-15 |
| 旧real codec復元U対元Uのmax-entry | 0.03006837075 |
| density projectorのmax-entry差 | 0.02128854656 |

全frameの複素logをそのまま指数化した残差は最大約3.55e-15だった。従って今回観測した大きい差を、単純に「logmの逆変換自体が大誤差だった」とは説明できない。−1近傍のbranchを含むlogを、実反対称パラメータへ制限した経路の不忠実性として扱う。det符号だけを直して全branchが解決すると断定しない。

## 初期／最終gradientと停止条件

既存ffsimの`_make_compressed_value_and_grad`を使い、実x0でのobjective/gradientを評価した。optimizer・minimizeは一度も実行していない。

| rank | 初期x0 objective | 初期gradient∞ | 保存最終fun | 保存nit/nfev | 最終再encode chartのgradient∞ |
|---|---:|---:|---:|---:|---:|
| 14 | 0.4444541184 | 0.3865359833 | 3.827224571e-7 | 254/263 | 1.044886270e-4 |
| 18 | 0.4444548244 | 0.3865339773 | 4.285587503e-8 | 159/168 | 3.475165598e-5 |

旧保存optimizerは両rankともsuccess=true／status0、messageは`CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`である。**元のoptimizer最終x/jacは旧workerが保存していない。** 最終J/Uを新しくencodeしたchartについて、tensor往復を確認してからgradientを計算したが、これを当時のjacや当時のprojected-gradient停止判定とは扱わない。初期x0のgradientと、最終再encode chartの診断量は資格を分けて保存した。

目的関数とsector基準の違い・停止条件の影響も残る。今回確認したのは初期変換での悪化と保存終了点の位置であり、最終未達の唯一の原因を確定したものではない。

## 修正案の人工検証と次の最小検証

人工例ではdet=-1の2次元反射行列で旧codecのtensor非保持を再現した。列符号U→UD（Dは対角±1）は各density U_pi U_qiを変えない。負detなら最後の列を反転する最小案を試し、人工reflection・一般対称J・2/6次元batchでtensor保持を確認した。

ただし、det=+1の人工π回転にも旧codec非保持の例があった。提案関数はtensor round-trip guardでこの未解決例を拒否する。汎用実対数・Gaussian合成器・新fitterは実装していない。修正案は人工データだけに適用し、**修正版のH6 codec評価やfitは0**である。

次の最小検証案は、保存H6の同じ初期J/Uに対して、列符号とπ branchを扱う修正codecのtensor／sector round-tripだけをoptimizer0で確かめること。全frameを保持しDF予算を維持できるかを確認してから、必要なら新しい1fit条件を別途判断する。いきなり21rankを再探索しない。ユーザー／研究Cがこの次範囲を採択・許可し、Codexがその後に実装・人工検証・限定評価を行う。

今回の結果だけから、修正後fitが予算に達すること、総資源が有利になること、H2/H4の既存数値結果が変わることは結論しない。重要な科学レビューの実施も今回の技術・限定再解析と区別する。

## call・資源・保全

codec側21＋sector側14、合計35 attempt／35 COMPLETE、unexpected correctness FAIL0。initializer再生成2・old codec2・final再encode2・tensor6・value/gradient4・sector6を固定契約のcomposite unitsとして計上した。元の21fit枠や旧STOPを再利用しておらず、新fit／optimizer iteration／rank探索／PF／回路／shots／総費用／retry／resumeはすべて0。旧codecの不忠実性はあらかじめ登録した診断対象であり、来歴やreference再現の重大な不整合とは区別した。

SOURCE37・保存入力17・ffsim Python2ファイルと固定版環境のpreflightはPASS。人工新規12＋既存回帰28=40 tests PASS。ffsim/SciPyのインストールファイル・共有venvは変更せず、旧per-matrix bridgeとprivateな固定ffsim関数を呼び出した。依存lockとPython symlinkを保持し、task-local cache/tempだけを使用した。

開始時affinity32 CPU・空きRAM約52GiB。旧H6の人工SVDによる選択を再利用して初期BLAS/OMP16、独立した6 sector状態はresource-aware 6 workers／process-local BLAS1で評価した。初期化同一性の確認とcodec診断は順序依存のため逐次。wall約5.037秒、親peakRSS651,520 KiB、child peakRSS424,712 KiB。記録されたCPU値はprocess lifetimeで、親子を足した同時測定値ではない。dense4096／controlled8192やfull-CAR tensorは作成しておらず、恣意的CPU/RAM/wall cap・共有設定変更・他ユーザーjob操作はない。speedupは主張しない。

契約SHA256 `b1367e4434892f74dc52c6902b22efa16745f80f12a298f5596846d6742d2b21`、SOURCE manifest SHA256 `e5f1543fb340febef575e25fba68ba6236818dafbc8863d28a933f725f870294`。ffsim固定SOURCEはutil.py `5ecfe167…`、double_factorized_decomposition.py `655453e7…`で旧lockと一致。実行後もSOURCE37、保存入力17、旧H6原本108／H2原本40、新raw43ファイルのbyte identityとcall整合を照合した。報告の作成は保存JSONの整形だけで、追加の配列decode・norm・gradient再計算はしていない。

これは固定SOURCEを用いたlocal科学実行の公開証拠であり、CIや外部独立再現ではない。SOURCE・結果をcommit/pushし、remote SHA・SOURCE・成果物を照合後、**mandatory STOP**。新規fitや追加科学計算へ自動進行しない。
