# H4/H6 標準Yoshida4：独立確認の実行準備

**Phase A・Bを完了した。既存独立S2のH4 25軌道・H6 23軌道を再利用し、新しい標準Yoshida4をH4 45軌道・H6 32軌道だけ取得する固定契約案を準備した。科学実行・実データの新会計・有効grantは0。Phase Cの一回の明示承認を待つ。**

これはε=.001での固定構成S2/q8対標準Yoshida4/q4の確認である。次数だけの因果効果、母平均順位、厳密bias証明、全誤差評価を対象にしない。旧strict COMPAREはBLOCKEDのまま。

## 固定識別子と科学条件

| 項目 | 値 |
|---|---|
| 新しい実行SOURCE | `a94735946e8cf15ffea5c93ae99000f7ab83d23a` |
| private exact契約SHA256 | `6d08c2531bc009042fb58926b7eaa5d45e26348457a865783d825a2af97df794` |
| 入力manifest SHA256 | `de1996e3d9317074c2a8a8ad0037d87cf51f0dcf3baf94cde1b0d20348ad9e80` |
| 主比較・固定NのSOURCE | `eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23` |
| 独立S2結果commit | `3eca46148070ba4d4b0f5815ef45943407cc959e` |
| 再利用するnative・統計SOURCE | `c028f1c2885afdba65ef8c62c075738aadd7323d` |
| 元科学SOURCE | `8825391ff73b531be5834bf852b9bf2479d1799b` |
| provider SOURCE | `217c971dd537b1cc7fbc13742bc0f6c263ff7f45` |

linear H4/H6、隣接1 Å、STO-3G、tol-only DF1e-8、rank10/19、T=.8、ε_signal=.001、α_axis=.025、P=0、usual signed bias。既存empirical allowance/reference/normalization政策を条件付きで再利用する。symmetric-directional control、状態準備を除くfull cosine/sine Hadamard wrapper、compiled RZ、Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed_transpiler17、backend/target/layoutなし。compiler fingerprintは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

| 系 | 公式・候補 | prefix/q | 標本 | 固定N_cos / N_sin |
|---|---|---|---|---|
| H4 | S2 `M5_MIN_72b305020499dd68` | 3/8 | 既存独立25 | 24249726 / 18445335 |
| H4 | 標準Y4 `M5_MIN_ab57ad5bd40dc583` | 3/4 | 新独立45 | 19240638 / 18807773 |
| H6 | S2 `M5_MIN_a08e8bfb6642eb93` | 5/8 | 既存独立23 | 22907931 / 27494282 |
| H6 | 標準Y4 `M5_MIN_37ff3428bb58f5cd` | 5/4 | 新独立32 | 21115055 / 21415717 |

PRは全てK4・ABS_TIME_PER_OUTER32。signed piece・順序・RTE配分は元SOURCEどおり。m5_bestを代用せず、Fullの新compile・S2再取得・新shot計算・候補探索は0。[公開可能な契約内容](execution_contract_DRAFT.json)に科学範囲、private契約digest、seed方針、受入条件をまとめた。公開JSON自体のSHAと上記private exact契約SHAは別物である。

## Phase A：既存S2の再利用

固定`paired_samples_118.csv`から対象48行を、候補・系・sample番号・cosine/sine pair・元結果SHA・c028 SOURCEで照合した。元成功終了監査、stage契約、保存固定比較、input再構築監査へ結合している。今回S2側へ取り込むのはこの48行だけであり、探索標本や旧独立m5標本は取り込まない。

Hamiltonian/state/rank、S2/q8/prefix3/5/K4/outer32、Gaussian preparation、native仕様・compilerが対応する。H4 snapshotは旧容器とのbyte一致を要求せず、保存済みの全科学配列hash・Hamiltonian/state一致記録を受け入れた。現在回収した再構築snapshotは独立確認で使われた`ed9b5a13…`そのもの。H6は元snapshot SHA`99440a59…`に一致する。preparationはH4 `317662c0…`、H6 `8f1a392f…`。

**今回、旧独立S2のraw全体や実使用seed registryを新たに回収・replayしたわけではない。** 受入済み公開pair CSVと固定された成功終了監査の来歴を再利用する。旧cohort・namespace digest・registry digestを記録し、新namespaceと区別した。元実行の完全な歴史的seed非衝突は認定しない。旧ε=.0005のG/SEは新評価へ転記せず、上表のε=.001のNで再会計する。各照合値と証拠範囲は契約の`phase_A`に収録した。

## Phase B：固定nの根拠と限界

新しいnの科学数値算定は行っていない。c028の保存済み`n_E_sensitivity_216.csv`で、同じ標準Y4/q4の**相対MCSE目標1%・分散倍率4**に対応する計画値がH4=45、H6=32であることを読取照合した。これを今回の採択対象とする。n=33のコピーや便宜的な32への固定ではない。

根拠となったpilotの保存単価分散は、H4 `2674985.0643939395`、H6 `27100618.484848484`。pilotの相対MCSEは各`0.005777496460266724`、`0.004873865450581225`。主比較の保存探索分散は各`4795580.104166667`、`31519615.064393938`であり、pilotと同一ではない。これらは保存値の引用である。計画倍率4は母分散上界ではなく、未観測tailを保証しない。

既存S2のn25/23は固定され、その標本不確かさはY4のnだけ増やしても消えない。今回の小さいε=.001での方向確認に対して、保存表のより厳しい1%計画を選んだが、新しいpower計算や差の検出確率は未算定である。保存主比較ではS2のGがY4より高いが、その方向を新標本で保証する設計ではない。不確かならINCONCLUSIVEとして終了する。実行前承認は、このn、pilot由来の限界、固定S2、非適応STOPの採択を含む。

新namespaceと無条件finite-RTE lawを用いる。元探索の2 registry・既知除外リスト、source由来の旧提案seed、今回の未実行旧準備packを限定的に読み、既知242,684値を除外した。新しい924 piece seedについて既知集合およびH4/H6間の衝突0。値とauthorityはprivateに保持する。旧移行確認の実使用registry未回収により、歴史上の全seed非衝突は未証明。希少eventの強制・棄却・条件付け・良いtrajectoryの選別は行わない。

## 再利用pipelineとexact予算

c028の`h46_pr_pilot_native.run_task`、`h46_m5_native.make_requests/compose`、`h46_new2_native.prepare/OccupationOracle`、saved-input loader、既存compiler、lossless archive、runnerのwait4/fork/終了監査を再利用する。元scientific bodyはbyte変更しない。adapterは新pack・cohortラベル、親context共有、起動制御、call監査、統計の入力結合に限定した。最初の無効準備packは保持し、wrapper生成前の予約を追加した最終SOURCEで新packを固定した。両packとも未実行・無効grantである。

| call境界 | H4 | H6 | 合計 |
|---|---:|---:|---:|
| 新PR trajectory | 45 | 32 | 77 |
| request sampling | 540 | 384 | 924 |
| RTE event（samplingの内部件数） | 5760 | 4096 | 9856 |
| native piece composition / partial-native build | 540 | 384 | 924 |
| native evolution | 45 | 32 | 77 |
| Hadamard wrapper生成 | 90 | 64 | 154 |
| compile | 90 | 64 | 154 |
| native low action | 45 | 32 | 77 |
| native high action | 45 | 32 | 77 |
| independent oracle action | 45 | 32 | 77 |
| uncompiled wrapper action | 90 | 64 | 154 |
| compiled wrapper action | 90 | 64 | 154 |

statevector simulationはlow/high/uncompiled/compiledの合計462 callであり、別の追加予算ではない。partial-native buildはpiece compositionの内部924 callであり重複加算しない。元runnerに無かったwrapper生成直前の予約を追加し、実dispatchの12 partial/2 wrapperを軌道ごとに照合する。

各系のcontextはbootstrap1/setup1/saved snapshot restore1/saved reference read1/occupation oracle初期化1/Gaussian prepare2/matrix-only tail coefficients1/tail spectrum1、**各9・計18 ledger entry**。既存関数内部の行列・対角処理を含む実setupであり、科学処理0のsetupとは呼ばない。親で一度だけ実行しworkerのcache missによる追加prepareを拒否する。新SCF・DF生成・状態solve・reference再計算・signal評価は0。

別の会計rootではbootstrap1/bundle load1/family constant1/paired statistics4/direction comparison2＝**9 event**。S2再会計2群、新Y4会計2群を同じ固定Nで処理する。新shot/allowance/fit/rankingは0。native台帳は77×38＝2926 entry（TASK77を含む）、context18、会計9、全体2953 entry。予約はnonrefundable、完了は別の結果・終了監査で受け入れる。

## 統計・判定規則

元`h246_confirmation_evaluation.paired_statistics`を変更せず使用する。各trajectoryの`G=N_cos*C_cos+N_sin*C_sin`、標本平均、paired covarianceを含む分散・SEを求め、整数/Fractionで重み付き分散の恒等式を照合する。公式間は別namespaceの独立標本設計として扱う。これは独立性の仮定に条件付けたSEであり、完全な歴史的seed検証の代わりにはならない。

`D=mean(G_S2)−mean(G_Y4)`と`ratio=mean(G_Y4)/mean(G_S2)`を報告する。差SEは既存`classify_confirmation`の`hypot(SE_S2,SE_Y4)`。familyはH4/H6の2比較、α_family=.05、両側Bonferroni正規近似の`z=Phi^-1(1−α/(2m))`を一度求める。区間下限>0ならREPRODUCED_APPROXIMATE、上限<0ならREVERSED_APPROXIMATE、それ以外INCONCLUSIVE。平均差の点方向は別labelとする。比の区間は計算しない。

正規近似は小標本・未観測tailで厳密coverageを保証しない。未観測stratumはUNOBSERVED・費用null、0補完なし。標本分散0も警告を付け、母分散0やtail不在としない。探索n33・既存独立S2・新独立Y4をpoolしない。保存Fullと探索のε=.001の結果はeebdbfcから文字どおり引用し、新Full会計・compileをしない。Full/PRの異なるqでの方向関係を、同一q対照や次数因果効果から区別する。

## 技術検査・資源・保全

予定runtimeのPython3.11.0rc1で人工24件PASS。mockのcontext9 call、partial12/wrapper2の実dispatch監査、会計9 eventを含む。未承認実入力の起動を数値runtime import前に拒否し、使用済みroot/予約の再利用、欠測・cohort・axis破壊を拒否した。さらに実packのmetadata変更11種を拒否。504 fileを封印し、回収した334 Git blobのSHAを照合した。metadata-only preflight PASS。[技術・保全・資源記録](technical_audit.json)、[元SOURCE一覧](SOURCE_reuse.csv)。これは一次担当の技術確認であり、別担当による独立科学レビューではない。

旧Python実行pathは現在不在だった。回収した旧provider、NumPy1.26.4/Qiskit1.3.0を利用可能なPython3.11.0rc1へ結合した。旧runtimeのbyte同一性を新たに認定せず、固定compiler/native仕様と全軌道・両軸の実数値gateを受入条件とする。旧S2のruntime/correctnessは旧成功監査の範囲を条件付きで再利用する。実runのCT01はまだ未実行である。

資源previewはaffinity64、idle約56.94 CPU、実効空きRAM約515.47 GB、disk約392.16 GB。現観測ではH4 15/H6 27 workerの並列案、保存容量proxy約136.24 GB。これは実行admissionや恒常capではなく、旧測定のRSS/保存proxyによる案であり上界保証もない。起動直前と親prewarm後にCPU・可視祖先quota・RAM/cgroup・disk/inodeを再確認し、独立2系を可能な範囲で並行実行する。固定compilerのnum_processes1とprocess内BLAS1を維持して多processを活用する。host PSI・一時的競合・無関係なworktree HEADではworkerを停止しない。

書込みは新専用worktree/private assetsだけ。旧SOURCE・結果・raw・grant・ledger・marker・STOP、論文主表/Figure2を変更しない。元raw/ledger全体の不要な再走査は行わず、回収済み保存証拠・既知read-only入力のSHAと書込み先の分離で保全範囲を明示した。共有venv・OS・scheduler・他process変更0。正式共通指示の実体SHAは`050e5dc7055a1c114c35eb17913f3ee050092b57444dcffb70949e896548ac28`、§8.1/8.4/10/12を適用した。

## Phase Cから終了まで

**ユーザー**：上記SOURCE・exact契約・入力SHA、n45/32、旧S2再利用、native正しさ検査、会計・近似統計・非適応STOPをまとめて一回承認または却下する。承認前に有効grantを発行しない。

**Codex**：承認後だけ新規grant/rootを一回使用し、H4/H6取得→全8 gate/軌道・CT01両軸・byte archive/ledger/native exit/reap監査→固定会計9 event→公開可能な集約結果・REPORT・保全監査・commit/push→mandatory STOPまで一括で行う。方向が期待と異なっても完了とし、追加sampling・候補変更・自動retryをしない。科学call消費後に失敗した場合はpartial/STOPを保持し、別承認なしに再実行しない。

現在は承認待ちで科学call0・有効grant0。[STOP](STOP.json)。詳細科学レビュー、最終freeze、旧strict解除、別分子系への拡張は開始しない。
