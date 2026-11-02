# 研究C：PR-aware DF Stage A・qualified x0接続修正

2026-10-27 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。起点：`4d7b88c7978604088de0a33f5aa0d2339c3301ba`。

**初回に資格付けしたx0を再parameterizeせず既存ffsimへ渡すよう修正した。人工・回帰61 tests＋29 subtests PASS、両環境のpassive preflight PASS。実分子のdecode・再解析、新fit、実optimizer iterationは0。** 停止済みrun01・旧契約・旧manifest・call・STOPは変更していない。公開照合後mandatory STOPする。

## 問題と修正

[旧Stage A報告](research_c_pr_aware_df_stage_a_execution_20261027.md)では、初期表現／DF予算PASS後に両fitter entryが`QUALIFIED_X0_CHANGED`で停止した。18/30 logical units、実optimizer0、retry/resume0の部分結果を保持する。

旧接続は、元J/Uをcodecでparameterizeして復元したJ/Uとx0を保存した後、fit内で**復元Uを再びmatrix logでparameterize**し、二度目のxに一度目のx0とのbitwise一致を要求していた。物理tensorの忠実なround-tripは、再packパラメータのbitwise冪等性を保証しない。

今回、分子配列を読まず、乱数seed固定の人工一般直交行列でこの旧接続のFAILを再現した。実H4の差の大きさやbranch寄与は追加測定しておらず、丸め差だけだったとは断定しない。

修正は以下の最小差分である。

| SOURCE | 変更 |
|---|---|
| `c1_ffsim_corrected_codec.py` | opt-inの`prepared_initial=True`を追加。資格付け済みx0を既存ffsimの`df_tensors_from_params(real=True)`でunpackし、保存J/Uとの整合を検査して、x0のcopyをそのままffsim pack出力として返す |
| `c_pr_aware_df.py` | PR-aware専用fitが上記opt-inを指定する。目的関数versionは維持し、初期化identityだけ`qualified_real_schur_snapshot_x0_v2`とする |
| `c_pr_aware_df_pilot.py` | 新scope `C_PR_AWARE_DF_PAIRED_STAGE_A_X0_V2`、新fresh output・初期化identityを照合。停止済みV1契約の起動を拒否 |
| 人工tests | 一般dense U／det±／π・branch近傍の実ffsim経路をoptimizer mockで検査し、旧二度pack FAILと新一度pack PASSを区別 |

最初のcodecは従来どおり`real_schur_column_gauge_v1`で、元J/Uのtensor・density projector忠実性を検証する。実行時の初期sector／general-K／raw-CAR gateも維持する。fit内では二度目のlogm・Schur・codecを呼ばない。pack/unpack・value/gradient・L-BFGS-Bはffsim／既存接続を再利用し、新optimizerやcodecは実装していない。

prepared snapshotについて、x0のfinite binary64・長さ`R*norb²`、Jの実対称性とunpack Jのexact一致、Uの有限性・直交性・unpack Uとの既存codec allowance `1e−10`内の一致を確認する。optimizerへ渡るx0は保存x0とexact一致させ、diagnostic hook前後の改変検査も残す。不一致なら明示的FAILとし旧codecへfallbackしない。

これはoptimizerに入るx0同一性の緩和ではなく、別の再parameterization結果を要求する条件を取り除く修正である。新たな物理誤差allowanceやpenalty条件は導入していない。

既存adapterのdefaultは`prepared_initial=False`で、旧repack経路・SETTINGS・hook・finally復元の挙動を保つ。既存H2/H4/H6 callerは変更せず、旧codec接続15 testsで非回帰を確認した。新branchでは共有adapterの新versionをSOURCEへ固定し、旧branch／commitの科学SOURCEや旧runのmanifestを書き換えない。installed ffsim、SciPy、既存venvは変更しない。

## 検証結果

| process／環境 | 検証 | 結果 |
|---|---|---:|
| fit env | 既存PR-aware目的9＋worker2＋新prepared-x0試験12＋既存corrected-codec15 | 38 PASS |
| evaluation env | 人工30-call end-to-end、初期／終了予算・表現STOP、未資格null、fresh拒否、byte preflight、RTE係数の既存接続 | 10 PASS |
| evaluation env・別process | 既存general-K／独立CAR回帰 | 13 PASS＋29 subtests |

計61 unique tests＋29 subtests。数値optimizerは全試験でmockしnit=0。既存ffsimのfitter・目的関数／gradientは人工配列上で呼び、optimizerに渡ったx0のexact一致、目的値と有限差分gradient、J/U復元を検査した。新prepared試験は2／4／6次元・複数frame、det±、共役π回転とπ近傍を含む。人工H4-shaped workerは一般dense Uを使い、prepareでcodec一回、fitでcodecゼロ回をassertする。

異常試験はx0のshape／dtype／非有限値／値変更、J・U不一致、U非有限、hookによるx0改変を含み、いずれもoptimizerに入らずSTOPした。旧実データを使った再解析や、その失敗が科学的に解消されたとの認定は行っていない。

実装commit：`5a185964090f2cb8d19d445d41cf520c4d8ed2c7`。科学SOURCE29件は[新manifest](../../artifacts/research_c_pr_aware_df/2026-10-27/technical/SOURCE.json)で固定。試験log・passive結果・[technical STOP](../../artifacts/research_c_pr_aware_df/2026-10-27/technical/STOP.json)は[technical](../../artifacts/research_c_pr_aware_df/2026-10-27/technical/)に保存した。localで実行して公開した技術証拠であり、CIや外部再現の結果ではない。

passive preflightではSOURCE29、保存初期入力3、固定ffsim2source、fit／evaluation package版を照合した。旧契約・報告・raw・STOP等45ファイルのbytes不変を確認し、新scopeが旧契約を拒否することも確認した。実分子NPZはhashだけを読み、配列はdecodeしていない。

## 新しい未承認の限定科学契約

[V2契約](research_c_pr_aware_df_pilot_contract_x0_v2_unapproved_20261027.json)。旧契約は変更しない。次の科学実行について、別途の明示許可が必要である。

| 項目 | V2で維持／固定した条件 |
|---|---|
| 系 | 保存直鎖H4・隣接1Å・STO-3G、Nα=Nβ2／sector36 |
| 初期値 | 旧案と同じ保存CDF8 J/U、元input・export hash不変 |
| DF／split | 通常number CDF、rank8、一般対称J／実直交U、元frame順prefix LD2 |
| 目的／optimizer | γ0／.1、ε1e−8 Ha、loss floor1e−14 Ha²、同一初期scale、既存L-BFGS-B options=None |
| 元規約 | 元g補正κ、核scalar一度、spin／frame順を維持 |
| 数値条件 | 旧DF budget .00019875 Ha、既存engineering guard、sector/general-K gateとexact mass1%基準を維持 |
| 変更 | 単一qualified x0を直接引渡し、新SOURCE／scope／outputに固定 |
| 科学call | 新範囲最大30、各arm1 fitで最大2。旧18を取消・移管しない。将来別許可の新範囲が完了した場合のみ累積最大48 |
| fresh output | `artifacts/research_c_pr_aware_df/2026-10-27/run01`。今回未作成。旧2026-10-26/run01は再利用しない |
| retry／resume | 新範囲内0／0、旧runのretry／resumeも行わない |
| STOP | SOURCE／input／x0／非有限／表現不一致で依存処理STOP、有限optimizer不成功や終了予算未達は診断して未適格、追加fit／fallbackなし |

30-callの分類・順序は旧案と同じで、prepare6＋paired fit各5＋sector／比較14。unpack整合検査は各fitter entry内の初期化検査で、実H4に対する追加評価callを隠しているものではない。目的関数・内部nfev/njevの記録・科学成功条件は変えない。

科学実行時は両環境のpassive preflightと空き資源確認を再実施する。独立2 fitは既存runnerの別processで並列化し、小行列BLASとJAXの競合を自job内だけ調整する。CPU／RAM／worker／wallの人工上限を追加せず、共有設定・他job・shared cacheには触れない。

## 残る制約と次の担当

今回解消したのは人工入力で再現したx0接続上の不具合である。修正後の実H4 optimizer収束、両終了DF予算、exact tail massの1%削減は未検証であり、保証しない。物理DF gateはengineering数値pilotで、厳密丸めenclosureではない。Stage AがPASSしてもPR減衰・PF/RTE誤差・native費用・shots加重総資源・科学的新規性は未確立のままである。

ユーザー／研究CがV2の新SOURCE・契約について限定科学実行を許可するか判断する。許可後に限りCodexがfresh outputへ最大30-call／2 fitを実行し、結果を公開してSTOPする。今回の「修正」依頼では再実行・追加科学レビュー・Stage B/Cを開始しない。旧結果・失敗証拠・grant／ledger／marker／STOPを保全し、公開照合後mandatory STOPする。
