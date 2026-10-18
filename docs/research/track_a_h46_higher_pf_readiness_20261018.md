# Track A Codex 2：H4/H6 高次PF・q比較・独立確認の技術準備

2026-10-18。状態：**準備完了、実科学計算は未承認、mandatory STOP**。
専用branch：`track-a-codex2-h46-higher-pf-readiness-20261018`。
実装SOURCE：`ed14d951a4761838f25aeb16a653b55aba499c34`。結果commitではなく、今回の純粋関数・adapter・runner・人工testのSOURCEである。

## 今回の到達点

利用者指定のPF研究repositoryを先に確認し、係数・次数検査と保存結果を再利用した。新たに作ったのは、PR固有のsaved signal→shot→paired native費用→q比較の接続と、独立確認の評価部である。実H4/H6のbias・shot・費用・順位・SEの再解析、新signal、係数最適化、新random trajectory、native compile、SCF/DF/state生成は**すべて0**。A_COST、理論費用比較、H8やgeometry系列を本作業へ取り込まない。

| 対象 | 再利用できるもの | 今回の実装・未確立事項 |
|---|---|---|
| A 係数 | PFの元8桁係数、既存projected候補・次数test、PRの保存decimal/binary64監査・70桁toy | exact rational監査、代数的root表現、formula ID分離。元高精度最適値・solver closure・新係数の採用は未確立 |
| B q比較 | H4/H6のq1/2/4/8 finite K4 signal24条件、同一系列のnative6群n8 | frozen shot rule、軸別適格性、paired RZ/CX・SE、q選択runner。残り18群の費用は欠測。実データの適格性判定・順位は今回未実施 |
| C 独立確認 | 元n8の限界、PF側のtrain/holdout管理の考え方 | 独立namespace、selection digest、SE/差/比の診断、標本規模の計画式。新しいnや母平均の優位性は未確定 |

[公開bundle](../../artifacts/resource_applicability/track_a_h46_higher_pf_readiness/2026-10-18/README.md) にmetadata監査、formula registry、4つの未承認plan・無効grant、SOURCE closure、人工testを置いた。

## 1. PF側の調査結果と再利用の限界

対象は [PF repositoryの固定main 20d64c2](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/tree/20d64c2fea1ed2d2188777df4f3d962ee94df486)。別branchの作業中checkoutは変更せず、Git blobを読んだ。8主要fileのSHA256と参照URLは `evidence_inventory.json` に記録した。

- `src/trotterlib/product_formula.py` のnew_2はcompact `(w0,w1,w2)` であり、PRの展開順 `(w2,w1,w0,w1,w2)` と一致する。元探索は非可換word係数の残差を用いたleast_squaresと候補選択で、float64最適化後のlongdouble評価は元係数の高精度復元を意味しない。
- [projected_reference_candidate](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/search_pf_cost_predictability_m2_m3.py) は `new_2_projected` を区別しており、8桁の切捨てを数値的に補正する設計が既にある。既存testも元new_2の立方moment欠陥を認めている。この課題をゼロから再探索しない。
- 保存された `PAPER_NEW4` compact値は `(-0.6581584493683974, 0.420087292300873, 0.4089919323833257)`。元8桁版とは別の数値候補であり、元の高精度最適解や厳密代数的証明として扱わない。
- **再現上の不足**：projectorがimportする `fourth_order_moment_residual_float64` と `solve_nonprocessed_4th_moment_coefficients` は、固定commitのimport先 `optimal_trotter.py` に定義がない。静的ASTとGit履歴・参照を調査したが、solverを実行したり、別repositoryの原本を修理してはいない。saved候補の存在と、その生成コード全体の再現可能性を分ける。
- 既存の有限時間窓、energy shift、QPE費用、分子holdoutの成果は設計に利用できる。ただしPRのfinite-RTE複素signal、Bによるshot増幅、ordinary ancilla native wrapperの費用とは評価対象が異なる。PF側の費用係数や最良候補をPRへ移送しない。

したがってAの次の選択肢は「既存8桁版の名称・限界を維持」「既存projected数値候補を別公式として承認後に検証」「厳密代数的条件を満たす別公式を承認後に定義」の3つ。今回どれかを科学的に正式採択してはいない。

## 2. A：数学的次数・丸めの仕様

`src/trotterlib/higher_pf_readiness.py` の `moment_audit` はdecimal文字列とbinary64の正確な有理数表現を別々に扱い、一次・三次moment、palindrome、負時間を出す。浮動小数点で残差が0に見えることを厳密証明にしない。

保存監査の立方momentはdecimalで `7.346136322761508E-9`、actual binary64で約 `7.3461363139E-9`。これは既存監査値の引用であり、新しいH4/H6解析ではない。既存70桁toyは有限qで4次に近い挙動、大qで2次への移行を示している。toyを再実行していない。

将来の厳密表現案は、`(a,b,1-2a-2b,b,a)` で有理数aを固定し、bを

`-6b³ +12(1-2a)b² -6(1-2a)²b +(1-2a)³+2a³ =0`

の実代数的rootとisolating intervalで表す方法。kernelは多項式を正確なFractionで生成するだけで、root探索・aの最適化・branch選択は行わない。a、root interval、実装の丸め精度、q-gridを将来の契約で固定する必要がある。厳密表現をbinary64へ丸めれば実装上のmomentは再び非zeroになり得る。Yoshidaの記号的4次・binary64実装・実測収束も同じ区別を守る。

`formula_registry.json` でS2、Yoshida、保存new_2、外部projected未採用候補、exact-family未採用案を別IDにした。今回のBは元3公式のみを対象とし、補正版を同名で混入させない。

## 3. B：保存値adapterと比較runner

### 固定する科学条件

H4/H6のlinear1Å/STO-3G、rank12/19、prefix3/5、保存Hamiltonian/DF/state/fragment ordering、T=.8、K4、q=1/2/4/8、ε=.001。per-axis ε/√2、α=.025、経験的allowance1e-9、logB margin1e-12、P=0。

1 outer step当たりRTE short steps32を、signed coefficientの絶対値に比例するlargest remainderで分配し、各piece minimum1を維持する。tieは元generation order。全runの総step数は **32q**。32を全qに割って予算を変える設計にはしない。

compilerはrz/sx/x/cx、optimization1、seed17、backend/couplingなし、ordinary control、explicit XXPlusYY/number Phase。Python3.12.3/Qiskit1.3.0等の保存profileを維持する。旧dense・symmetric-directional費用は混合しない。Gaussian実装、scalar相対位相、低/高control、全particle-number oracle、compile後vectorの1e-9検査について、修正済みnative SOURCEと保存18人工/222実数値PASSを再利用する。今回の新テストはadapter/mockの検査であり、新しいnative演算検証ではない。

### metadataで確認した範囲

| 系・prefix | 公式 | 保存signal K4 | 保存native費用 | 欠測q |
|---|---|---|---|---|
| H4 p3 | S2 | q1/2/4/8 | q4・n8 | 1/2/8 |
| H4 p3 | Yoshida4 | q1/2/4/8 | q2・n8 | 1/4/8 |
| H4 p3 | new_2 | q1/2/4/8 | q1・n8 | 2/4/8 |
| H6 p5 | S2 | q1/2/4/8 | q4・n8 | 1/2/8 |
| H6 p5 | Yoshida4 | q1/2/4/8 | q2・n8 | 1/4/8 |
| H6 p5 | new_2 | q1/2/4/8 | q1・n8 | 2/4/8 |

24 signal条件、6費用群（保存96軸）、18費用欠測群。これは件数・identityの照合であり、18群全てをcompileするという決定ではない。別承認のB_SAVED_ONLYで適格性を評価してから、必要な欠測群だけを取得する。q>8は未採用。q8が保存格子の最良になり境界の検討が必要なら、その時点で追加格子を別計画にする。

adapter `scripts/h46_higher_pf_adapter.py` は固定commitのJSONとbyte一致を確認し、SOURCE、provider、compiler、入力、prefix/q、n8、forced除外を結合する。H4 SOURCE 687679a4…の8file、H6 executor f0dd2a9a…の11file、実import provider89file、元入力4fileをGit blobへ照合した。providerはc31ef424…。両者のseed SOURCEは元H4 SOURCEを保持する。

`load_rows` は実値を純粋kernelのschemaへ写す接続部だが、今回は呼び出していない。synthetic fixtureでのみadapter→runnerを通した。metadata監査はbias・費用・SEを計算しない。旧STOP、run02 handover、失敗compile entry、既存grantを変更しない。

### 費用・選択・欠測の扱い

- `d0_frozen_shot_rule_v1.py` の整数ceil、headroom、B²会計をそのまま呼ぶ。biasは保存finite-RTE total_signed_biasの軸別絶対値。PF次数からbiasを推定しない。
- trajectoryごとに `G_i=N_real*C_cos,i+N_imag*C_sin,i` をRZとCXに計算する。両軸のevent digest/seedが一致し、SOURCE/input/preparation/compiler/formula/q/stageが結合していなければ拒否する。
- cost標本は通常sampleだけ。forced eventは拒否する。整数費用の平均・不偏標本分散はFraction、SEはsqrt(s²/n)の診断値。n1のSEはnull。
- MEASURED/MODELED/MISSINGを分離。modelは別表示でSEなし、実測qランキングへ混入させない。欠測はnull。INELIGIBLE、UNDETERMINED、log-domain-onlyも区別する。
- 同一system/formula内の4qを必須とし、点費用は正確な有理数で比較、tieは小さいq→candidate ID。適格性未解決、適格qの費用欠測やshot整数欠測があればfull-grid winnerはnull。available-subset point bestと区別する。
- 同一系の候補間の費用差を出せる。候補独立性を明示した場合のみ差/比のSE診断を出す。CLIの既定では独立性を認定せず、厳密区間や母平均winnerを出さない。

runnerは既定でscopeを表示するだけ。公開draftで `--execute` を指定しても、実値読込・output/marker生成前にexit2となる。将来承認されたsaved-only実行も、sealed SOURCE/input closure、独立grant、未使用outputを要求する。grant markerはexclusive createし失敗時も残す。例外/SIGTERM/SIGINTはSTOP保存、retry・sampling・compile・次Phase移行の経路を持たない。新native取得はこのrunnerでは起動できず、独立した取得契約を必要とする。

## 4. C：独立確認と統計的限界

探索で選んだcandidate集合・探索seed集合・結果SHAをselection digestへ凍結し、確認契約はそのdigestに結合する。確認時の候補変更、seed再使用、旧n8や別PR union6/n32の無断混合を拒否する。

SHA256のseed namespaceはcampaign/stage/candidate（formula/system/q）/replica/outer/piece/input hashを含める。cosine/sineには同じevent列を使う。起動前に元SOURCE/group/sample/outer/pieceから既存uint64 seedを再導出して新registry全体と衝突照合する。今回は新しい実trajectoryのseed列を生成せず、関数を人工引数で試しただけ。namespaceの相違だけで乱数の数学的独立性が証明されるわけではない。

公式間CRNは未採用。signed piece数・allocation・event lawが異なるため、同じsample番号は共分散の根拠にならない。使うなら共通base randomness→各候補の正しい周辺分布とjoint lawを別契約で証拠化する。

独立候補仮定で `SE(diff)=sqrt(s_A²/n_A+s_B²/n_B)`、`SE(A/B)` はdelta methodで用意した。いずれもplug-in探索的診断。小標本・重いtail・分母不安定では厳密coverageや母平均winnerを主張しない。分母0、SE欠測では統計結論を保留する。SE0も未観測eventの不存在を意味しない。

標本規模は将来ユーザーが目標MCSE、検出差、power、多重比較family/αを決めてから計画する。`n≈s²/τ²` またはnormal近似＋Bonferroniによる同数2群の差の計画式を実装したが、pilot varianceから得るplug-in値に過ぎず `adopted_n=null` を維持する。n8から根拠なくn32やn64を確定しない。追加sampleの自動実行もしない。

未観測s4・複数2次eventの費用を0補完しない。forced controlは期待費用標本ではない。全supportの費用上界、tail/varianceの根拠、必要な独立標本がなければ、母平均・未知希少層を含む優位性の結論は保留する。未確立のbias保証は経験的allowanceと別のまま。

## 5. 実行計画・予算・順序

全planは未承認、全grantはINVALID、科学execution count0。

| 単位 | science call / compile / sampling | 依存・STOP |
|---|---|---|
| A_COEFFICIENT | 採用候補・root表現・toy格子を別途固定。現時点の実分子/compile/samplingは0 | 元係数と別formula ID。solver原本不足を推測で埋めない。A終了STOP |
| B_SAVED_ONLY | 保存24signal・6費用群の限定数値再解析。新signal/compile/sampling0 | 別承認後に初めて適格性と費用を評価。不足ならfull-grid winner null。終了STOP |
| B_ACQUIRE_MISSING | 適格な欠測群m≤18、探索n_E未確定。trajectory m*n_E、cost wrapper 2m*n_E≤36n_E | B適格性・標本予算・native group/seed追加契約を固定後、別承認。control/診断compileは別台帳で事前登録。終了STOP |
| C_INDEPENDENT | 事前固定k候補、n_C未確定。cost wrapper 2k*n_C（例として各系3winnerならk6、未採用） | Bのselection digestと新規sample計画を先に凍結し別承認。終了STOP |

Aと元3公式のB_saved準備は独立。補正版を実H4/H6へ導入するならAの決定と新しいsignal/native検証が先に必要となり、元signalを流用しない。Cは選択固定後に行う。元3公式での軽量B_savedを先に承認する案が最小で、新係数採用や18群の一括compileを前提にしない。

### 並列化・資源

今回の純粋人工testは約0.16秒であり、多process化のoverheadが大きい。metadata監査も小さい既知fileに限定。重い科学処理は起動していない。

将来はH4/H6・formula/q・trajectoryを独立processへ分配する。immutable spectra/DF primitive/preparationは各process内でcacheし、cosine/sineのevent生成・wrapper共通部を共有する。Gaussian loweringのprocess-local patchはthread共有しない。compiler内部num_processes1は保存profileと整合させ、trajectory並列で速度を得る。worker数はfresh affinity/利用可能RAM/当該qのfootprintから決め、昔のworker上限を引き継がない。

新しいqでは回路長・replay保存量が増えるため、既存groupのwall/RSS/byteを出発点に、登録したq・sample数からprivate event replay＋2軸wrapperの容量を起動前に見積もる。現時点でn未決定なので絶対GiBやwallを確約しない。publicは軽量JSON/CSV・receipt、QPYは必要時のみprivate spool。必要容量未確定のまま取得契約をsealしない。shared OS/venv/他jobは変更しない。transient PSIや無関係worktree更新を理由に正常jobをkillしないが、実OOM、SOURCE/input破損、own process異常、科学不一致ではSTOPする。

## 6. 今回の検査・保全・公開

新規**32人工/mock tests PASS**、Python3.12.3。係数表現・多項式、凍結allocation関数、signed q順序/32q、mock scalar/control、既存explicit XY経路、shot整数/B²、非finite/欠測/model、paired軸、q/tie、SE/ratio、selection/seed分離、invalid grant、one-shot拒否、人工24条件のadapter→runnerを検査した。実native演算の独立検査を今回追加したとは主張しない。既存の18人工/222実数値を再計算していない。検査は作者自身が実施し、独立reviewではない。

専用cloneのみへ書き込んだ。既存PF checkout、元PR結果、SOURCE/input/Git blobは読取専用。既存grant/ledger/marker/lock/STOPは未操作。他worktreeは凍結せず、通常進行を妨げない。成果物を専用branchへcommit/pushし、remote確認の結果は最終報告・公開receiptに記す。認証が利用できなければ認証設定を変更せず、その事実と手動pushコマンドを報告する。

次の担当：**ユーザー**がA候補の扱い、および最小B_SAVED_ONLYの数値実行を承認するか判断する。**Codex**は別承認後に該当Phaseだけを実行し、結果を保存してSTOPする。Bの新native取得とCの標本規模・samplingは今回の準備承認へ含めない。
