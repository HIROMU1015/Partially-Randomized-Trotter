# C1：次期最小実験の比較・技術推奨（未採択）

2026-10-26 JST、起点 `3a8356a7843d3cb46bb8069955640381b490ac4c`。[最新統合・baseline監査](research_c_c1_h2_h4_h6_integration_update_20261026.md)の技術設計。

**次は案A：H4 XDF9の表現・own-ground／alias・26 PFセルを推奨する。** 保存順first-passingのrank8が最小総費用baselineかは未確認であり、rankを1増やすPF／shots上の利得とnative費用増の交換関係を検証する入口になる。単に小さい計算だからという選択ではない。現H4の約12%という主張のbaseline頑健性に直接関わり、既存のprecision／full512／CA接続を使って最終比較へ進める点を評価した。

本書の推奨と、Bの初期予算政策案は正式採択ではない。今回の科学call・fit・保存配列再解析は0。Aの最初の範囲だけ実装・人工検証し、Bの大規模runnerは作成しない。旧SOURCE・契約・STOPは変更しない。

## A：H4保存順XDF9

### 研究上の問いと入力

rank8→9の追加frameが、PF／必要shotsを変えてnative費用増を相殺するか。保存DF norm＋guard `2.0947134659228792e-7 Ha` はCA/8内だが、PF・shots・総費用の改善を含意しない。rank8は保存順1..9の最初のDF予算PASSであり、全XDF方式の最小rankでも費用最適rankでもない。

元P-H4は公開commit `86b8b002704aced8ca239414080f53793de81658` の `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。保存λ/Gの元順prefix9、元g補正κと核scalarを使う。並べ替え・新DF分解・fitはしない。

DF値はSOURCE `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5` の消費済み356-call実行の[prefix_9](../../artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/run01/prefix_9.json)から引用する。新しいDF normは0。prefix8・10のHamiltonian／ground／PF／回路／shotsをprefix9へ転用しない。

### 最初の固定範囲：95 logical units

| 科学call | 最大 | 再利用と評価範囲 |
|---|---:|---|
| saved_input_load | 1 | whitelist原packet＋DF row／STOP／SOURCEの3 JSON。fit export等は読まない |
| saved_DF_gate | 1 | rank9・予算・allowance・来歴の保存PASS確認、norm再計算なし |
| sector_context | 1 | 既存PhysicalSector／matvec contextとSectorCAR(4)、sector36 |
| input_identity | 1 | spin lift・κ・scalar・basis／bit／permutation・frame順 |
| xdf_prefix_model | 1 | 既存λ/G prefix9 DFHamiltonian／linear operator |
| xdf_matvec | 36 | 保存prefix9 sector targetの36基底action |
| raw_factor | 10 | base κ+cと元順9 square因子を既存CARで構築 |
| K_model | 1 | 固定Gの9 inner diagonalization、J=2ληηᵀと一般K adapter |
| representation_gate | 10 | base＋各frameのraw CAR対独立I/Z/ZZ展開 |
| sector_sum_gate | 1 | raw／K合計と保存matvec targetの照合 |
| ground_eigenpair | 1 | XDF9自身の36次元eigh、ground／spectrum保存 |
| ground_alias | 1 | eigenpair残差・正規化・H4登録window |
| sector_kernel_reference | 2 | s1／4、各base＋9 frameのordered S2、19 expmを内包 |
| PF_round_assessment | 26 | 各sのm0..12、既存sector 2-norm／反復二乗法 |
| candidate_summary | 2 | 登録十分条件の全13round分類 |
| **合計** | **95** | fit／DF norm／native／合成／CA／shots／費用0 |

[未承認契約](research_c_c1_h4_xdf9_pf_contract_unapproved_20261026.json)で既存数値条件を固定。component／ground残差1e-10、正規化1e-12、rtol0、alias window[-10,10] Ha、β_totalπ/6、β_statπ/12、PF operator誤差≤sin(π/12)。保存DF allowanceは`binary64_operation_scale_guard8_v1`。新規閾値・guardの採択や事後調整はしない。

SOURCE／入力／表現／ground correctness FAILは依存callを停止、開始済みtaskを回収してpartial evidenceを保全する。alias不明ならPFに進まない。ある候補のPF十分条件NOT_METは`INFEASIBLE_REGISTERED_PHASE_ROOM`として他の登録候補を継続し、物理的不可能とは呼ばない。値が欠ければ`UNRESOLVED`。全PASSでも`POTENTIALLY_FEASIBLE`だけで、CA達成とは呼ばない。

独立worktree、fresh `artifacts/research_c_c1_h4_xdf9_pf/<execution-date>/run01`、retry／resume0、既存run・残call再利用なし。2つの独立s taskを既存resource_planで利用可能CPU／RAMに合わせ、複数worker時はlocal BLAS1で過剰並列化を避ける。worker上限2は登録独立task数によるもので、恣意的な低capではない。36×36、2 S2 kernel、26 normが主要処理。full256／controlled512回路はこの範囲で作らない。固定wall予測・speedupは未測定。

### PF後に別途必要な資格

1. XDF9の新しいnative局所10因子・S2 s1／4を共通precision Gaussian／stable-phase P0、shared／fenced serial policyで構築・compileする。full512でraw／compiled対独立reference、actual adjoint、X/Y／head／readoutを検証し、全13roundへ合成誤差を伝播する。
2. XDF9専用のPFと合成profile、DF norm＋guardから同じfailure配分でNを算定し、固定N margin／energy／条件付きserial費用を接続する。XDF8や10のshotsをそのまま採用しない。
3. CDF8は元input・fit export・target・κ/scalar/frame/time／PF・precision provider・P0・shared／fence・phase／failure／horizon・serial指標が同一のときだけ、現行SOURCE862a97d…のnative／合成とSOURCE18045333…のCA／shots／費用を再利用できる。変更があれば影響範囲を両armで公平に再資格付けする。

この後続範囲のcallは、PF結果と必要な新回路／reference範囲を確認して別契約で固定する。今回の95-callに無条件に含めず、数値評価を始めない。一般相対Gaussian融合などを追加採択しない。

XDF9の最終費用が8より小さければ、first-passing baselineとresource-oriented baselineは異なると示す材料になる。CDF8との差をどこまで縮めるか／反転するかは実測前に予断しない。改善しなければ**登録8対9の範囲で**現baselineの頑健性を補強するが、XDF全体の最適性は示さない。PF不適格の場合も「DF精度だけで最終適格性を予測できない」という切り分けが得られる。

## B：H6 CDF13・修正codec固定1fit案

[前回B設計](research_c_c1_minimal_experiment_design_20261026.md#案b-h6-rank13修正codecによる最小1fit)のSOURCE／call分類を再利用する。新H4結果を受けてもBの科学条件は未採択で、rank14の既存契約を緩和しない。

| 固定案 | 内容 |
|---|---|
| 対象 | 直鎖H6・1Å・STO-3G、Nα=Nβ3、sector400、rank13だけ |
| 元入力 | H6-run01/input.npz、SHA256 `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447` |
| 初期候補 | H6-run01/XDF_full.npz、SHA256 `50623b80baa69a8fece6d0923ad4128634bfb12ba46ff1269d86bf5aa1316abd`、actual18の元順prefix13 |
| initializer | ffsim0.0.83の既存非最適化explicit Choleskyをtol1e-8／max_vecs13で1回再現し、rank・J/U・符号・順序のidentityを先に確認。不一致STOP |
| codec／x0 | real_schur_column_gauge_v1、J／projector／tensor／sector保持、実fitterに渡すx0 exact一致、旧codec fallbackなし |
| fitter | 通常number CDF、既存JAX x64目的関数・解析gradient、L-BFGS-B、options=None／callback=None／jac=True、元κ／scalar／frame順 |
| 新しい予算政策（要採択） | codec fidelityは必須correctness。**開始DF予算は診断値**で、予算外も記録して1fit可能。終了norm＋guard≤CA/8を適格性判定 |
| 停止 | 最大1fit、restart／best-of／retry／resume／rank変更／初期値fallback0。optimizer successとDF予算を独立表示 |

保存XDF13はsector誤差 `.0004632769171816072 Ha`＋guard `1.081524262161987e-6 Ha`で予算外。旧CDF13も誤差`.00081132667707527 Ha`＋guard`1.131698121763783e-6 Ha`でNOT_MET。旧fitはsuccess/status0、fun2.1066494419957358e-7、nit349／nfev373だがraw x/jac／njevが未保存。旧codec問題がrank13で与えた量的影響は未測定であり、修正後の可否は未判定。

既存rank14 runnerはinitial budget PASSをfit前条件とするため、rankだけ替えて起動できない。Bには独立entry・mock接続試験と、開始予算を診断とする**新実験条件の採択**が必要。今回そのrunner／契約を変更しない。

codec component max1e-10、tensor F≤1e-10·max(1,||g_before||F)、objective照合1e-12＋1e-8·|expected|、qualified x0 exact一致を維持する案。sector400評価は既存`SectorCAR`／`rank_assessment`、guardは`sector_operation_scale_guard8_proposal_v1`。tensorとsectorの増減、optimizer終了条件、生x/jac、開始／終了gradientを別々に保存する。

call案は**最大21**：保存load／identity2、非最適化initializer再現identity1、codec／tensor assessment3、既存fit＋前後value-gradient等6、sector snapshots／context／physical4、before／restored／final assessment3、初期codec同値gate／終了decision2。前回20-call rank14設計にinitializer照合1を加える。最大1fitの内部iterationを1unitに含む既存定義を保持し、PF／native／CA／shots／費用は0。これは未実装の設計上限で、実行可能な完成契約ではない。

6軌道・13 frame、sector400 sparse／matrix-freeを再利用しphysicalを一度共有する。4096²／8192²密unitaryは不要。修正rank13のiteration／所要時間は、旧349iterationや修正rank14の1iterationから予測しない。最終予算PASSでもCDF13自身のgeneral-K／ground／alias、PF、precision native、全sector-input合成、CA／shots／費用は別資格。既存XDF14はtarget／条件が不変なら対照として再利用できるが、CDF14を13の証拠に代用しない。

NOT_METはこの1fit条件の未達であってrank13一般の不可能性ではない。有限unsuccessful optimizer returnは登録終了診断だけ、例外・非有限・codec／objective不一致はpartial evidenceを保存してSTOP。初期値を自動採用せず、追加探索をしない。

## 情報価値と取得費用の比較

| 観点 | A：H4 XDF9 | B：H6 CDF13 |
|---|---|---|
| 問い | first-passing XDF8が費用比較baselineとしても強いか | 忠実初期化なら1frame圧縮してDF予算へ届くか |
| 既存成果への影響 | 現H4約12%のbaseline依存性を直接検証 | 現H6約2%劣位を変え得るが、1fit PASSだけでは費用結論は変えない |
| 成功／未達の情報 | 8→9のPF／shots対nativeの交換関係。PF未達もDF精度との非同値を示す | corrected13の単一fit可否、tensor目的とsector適格性の関係。一般不可能性は示さない |
| 実装負荷 | XDF8の薄いrank依存adapter。人工14＋適合回帰を今回実施 | 既存rank-generic corrected fitを使えるが、独立13entryと新しい開始予算政策のmockが必要 |
| 最初の計算 | 95 units、36次元、fit0、26 PF norm | 21 units案だがJAX1fit・400 sector norm診断あり。logical unitsはwall時間の比較尺度ではない |
| 後続負荷 | 成熟H4 full512と保存CDF8の再利用。新target9のnative／合成／N／費用は必要 | 新CDF13のPF・native・800入力のsector-complete合成が必要。前H6合成は既報86.2分で13への転用不可 |
| 再現性／公平性 | 原packetと保存prefix9が固定済み、両arm共通precision条件を維持しやすい | 入力・codecは固定可能。initializer13同一性と初期予算政策が新しい条件 |
| リスク | rank増でPF／N／総費用が改善するとは限らず、厳密enclosure未資格 | 初期予算外、収束未知、tensor減少でもsector予算未達の可能性。圧縮の費用効果も未測定 |

以上からAは、現行比較の強いbaselineという直接的な問いを、最適化不確実性なしに段階的に解消できる。Bも圧縮crossoverを問う価値があるが、最初のDF PASSから最終資源比較までの追加検証が大きく、今回は次順位の未採択設計として残す。

ユーザー／研究CがAを正式採択し固定95-callの科学実行を明示許可すれば、Codexはその範囲だけ実行して公開後STOPする。A/B双方の自動実行、重要な科学レビュー、PF後のnative／shots／総費用への自動進行はしない。
