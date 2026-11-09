# A. PR＋RPE既存成果・重複整理

2026-11-09 JST／SERVER 2。静的調査。数値は指定固定報告・保存scalarの引用であり、新しい再集計・再ランキング・数値再解析ではない。[総括](REPORT.md)／[候補](IMPROVEMENT_CANDIDATES.md)／[最小検証案](MINIMUM_VALIDATION_PLANS.md)。

## 1. 調査の基点と範囲

既存の[三環境台帳][LEDGER]・[再利用地図][REUSE]、[11月8日横断調査][PREV]とその比較表を優先して利用した。旧台帳の129記録を再棚卸し・複製していない。指定5成果を原報告で確認し、必要箇所のSOURCEを静的読解した。その後の統合と関連成果を下表で補う。

| 役割 | 閲覧用固定commit／branch | 今回の扱い |
|---|---|---|
| Closed P3共同最適化 | `c5a3c102199fe97f4d5c21837c059ae2684f5ef2` | 最終正本は **extended/FINAL.json**。root FINALは旧段階 |
| Closed P3初期統合 | `2e38af3b12200c48c4f1b793638d223baff24e76` | FAIR比較を採用。予備FINALと分離 |
| Track C旧総括 | `100377dac8635fc49074147bfd8b42f52f2b7311` | D内部・basis・対角融合までの経緯と負結果 |
| Track B既存成果調査 | `a8299889de93c201bf21e727415d629f5e4b9f48` | 「H4全費用結果未発見」は同調査集合での状態 |
| [B/C接続調査][BC] | `4d48a6f2097ea81ff3b7c64114e30bef35b1b685` | 接続条件を再利用。後続P3接続を未着手へ戻さない |
| 最新Track C統合 | `b3c81c2677988c7b9491eeee31fbcc0f9b29b078`／`codex/track-c-o3-integrated-20261108` | 上記に3分子比較・相殺P1/P2を加えた総括 |
| Track D統合 | `141fa7624d74aa5deacbd752d09df516cad9206f`／`codex/track-d-consolidated-20261109` | C1人工・実H4の後続完了を反映 |
| D q64完了 | `fe361b206a6572142775ac01972545d312e5fb14` | 全8 path・32 postcheck完了。旧未完了記述を更新 |
| D長q資源T0 | `c583abba520a711895d21b7ec9590688349a7bd1`／`codex/resource-eval-track-d-longq-structure-20261109` | native stream計数の人工検証。P0価格は未資格 |
| O2統合 | `d28164b5d51e0ca5a0ade74ded8ff176b443dab8`／`codex/o2-results-integration-20261108` | sampling／測定法の重複・負結果 |
| 全H純RTE | `9b83b96b83f62142c9d87858dc24a94db7ac5d61`／`codex/df-full-h-pure-rte-light-20261109` | H4/H6固定Tの結果。18round RPE結果と混ぜない |

関連remote branch名はGitHub接続で検索し、上記主要6 branch（本作業branchを含む）のtipはread-only `git ls-remote`でも確認した。全branch全文・全raw・他SERVERの非公開資料の網羅調査ではない。別チャットの未公開／進行中状態はUNKNOWN。旧branch名の削除を成果消失と解釈しない。

閲覧commitと科学実行SOURCEは別である。例えばC共同最適化の科学SOURCE headは **e83968cca8a69cc1d6d4c9c9861bed452f74800f** で、段階履歴・公開モデル・raw indexは[C-SOURCE]にある。今回の文書commitへ実行SOURCEを付け替えない。

## 2. 比較条件を混同しないための対応

全ての今回の候補は「元Hのexact groundを供給、準備費除外、同じenergy精度・成功確率」が目的である。以下の過去のscopeはその目的への証拠として使える範囲が異なる。

| 条件ID | Hamiltonian／state／split | 時間・精度・law | 価格・保証の境界 |
|---|---|---|---|
| C-H4 | Linear H4/1Å/STO-3G、保存Hermitian DF12生成順、元H ground、8 system qubits・N=4 sector70、PR LD3/D4/R218、Full LD12 | δ0=.02、18round m0..17、εE=1.5936e-4 Ha、failure≤.05、phase .4。K2=P3、最終h/rateはpolicy依存 | Qiskit1.3/compiler A、modular RZ/CX/size、depthはserial上界。保存H・ideal native・fresh bitsに条件付き |
| D-H4 | 同名分子、元H ground、number-CDF8、元順LD2、Nα=Nβ=2 | S2/s1/r32/K0、δ=h=.1、13round q1..4096、εE=.00159 Ha、failure=.01 | P0 opt1/seed17、whole wrapper。長q proxy・actual成功の未資格を保持 |
| A-C0 | 同名分子、保存rank10、元H stateをgroundと仮定、共通T_eff/V_square、LD0 | δ=.05/r2/K2、7round q1..64、εE=.1 Ha、95%目標 | opt1/seed7。engineering allowance、価格MC32/round、包括保証not_certified |
| A-B2/P1 | cutoff前H_phys ground、DF12/LD3、spin sector36 | δ=.02/r4/K2、18round、εE=.00015936001019904 Ha、固定予約位相予算 | opt1/seed7。C-H4とsource・state規約・予算・価格が異なる |
| RC | 同名H4、number-CDF8/LD2、候補ごとに近似Hと**own-DF ground** | K0/h=.1、13round、CA=.00159 Ha/failure=.01、Z再配置、各r再選択 | P0、ideal-exact-native proxy、actual CA未資格。元H共通ground比較ではない |
| B-toy | 3-system-qubit synthetic、p=(1/5,3/10,1/2)、x=5/7、state000。分子geometry/basis/DF/LDは非該当 | G9有限P5、軸精度1/200、familywise1/20。G10/Parentは別confidence/task | strict RZ1e-6、指定provider。分子RPE総費用へ外挿しない |
| Pure-H | H4/H6/1Å/STO-3G、tol-only DF10/19、保存state/reference、Dなし、一体項もsampling | T=.8、H4 q16/R16/K4、H6 q32/R32/K6、signal座標精度別の固定点 | opt1/seed17、各n9価格、経験的shot。energy RPE保証・最適化比較ではない |

exact groundの供給は、未知E0をdecoderへ渡すことや任意の古典期待値を無料取得することとは別。設計・検証にstate情報を使う場合はそのアクセスと費用を明示し、強Full／Truncatedにも同じ機会を与える。

## 3. 系列別の到達点と重複

| 実際の研究内容 | 検証済みの範囲・結果・負結果 | 未検証／依存・重複・状態 |
|---|---|---|
| Track A：DF/Pauli表現・分割、M1 | A-C0全7roundではDFのq64単価CX12805.375は強PF2183264より小さいが、attenuation約3.18e-11でshot/axis約1.12e24。条件付き総CXはDF2.86880473e28対強PF9.82700813e9。[A-C0] | 安い単価でも総費用は負結果。固定r/Kの結論であり全PRのno-goではない。旧F1/F2・mixed辞書のcorrectness／単回価格と全RPE保証を区別 |
| Track A：core＋保持residual M2 | Hermitian-B規約でC_H+R_H=H_H、full256/sector36、微小360 nonidentity項保持を検証。[A-M2] | **operator-only PASS、物理model適格性UNASSESSED**。PF/RTE・価格・N未取得。残差の小ささから費用勝利を認定しない |
| Track A：B2元H接続 | 保存18round B2は7,805,784 shots。強Full S2 s1/s2・Y4 s1は各18,540 shots、branch/energy条件付きPASS。[A-B2] | 価格比較未完。5 STOP armは未再資格。streaming/fingerprintは技術高速化で量子コスト改善ではない |
| Track A関連：旧Pauli/DF資源研究・全H純RTE | 旧台帳にB2/B3、Pauli H2/H4/H6、長時間・高次PFを保持。新Pure-Hは一体項もRTEへ入れてDを空にし、固定点のbiasは丸め水準、normalizationとshotは増加。[PURE] | 固定T=.8/各n9で純RTEは保存強PF等より高価。r/K未再最適化・rare費用未取得で一般no-goではない。B3のLD0は一体項をDへ残し、純RTEと同一でない |
| Track B：形式return／G6–G10 | Q_i²=Iの形式word集約、有限bit補正、Closed P3/P5。B-toy G9の十分N/axisはordinary1,246,046→Closed P5 917,129、2軸T切片348,759,216.423→255,860,636.626。[B-SURVEY][O2] | General FullとClosed P5は同ideal accepted ensembleの別生成法。P5をK2へ移すとtargetが変わる。G10は次数内比較、分子同精度順位ではない |
| Track B：Parent／IS／非加法価格 | Parentの固定toyにはT低下例。S2の3,969 pairではqBがqAより僅かに低TだがP0より約.01296807%高T。単価減をN増が相殺。[O2] | Parent D＋E終了、C1凍結、安価なwhole-path score取得未特定。新法への再命名・同じpair表再探索はしない |
| Track B→Track C：Closed P3 | 同じ有限P3、πw=係数、全path weight、range、root/angle/native diskを接続。固定policy FAIRはRZ1.965196%減、trial17186→17028。[P3] | 旧B調査の「H4未発見」を全repositoryの未実施と読まない。m3接続は完了。元B C1は324 labels/13 basis/m5/tail-only、Cは218/P3/D＋R全体で別入力 |
| Track C：D/R PF・ground BCH | 正確D/Rの5段族最良a=1/4は半刻みS2二回と同値。混合2MNはoperator norm改善でもground bias/shot改善なし。D内部h² biasとmixedのground期待値相殺は登録r固定でRZ26.195%、CX30.687%減。[C-OLD] | D-only Y4、高次partialの負R時間による減衰は負結果。特定groundの主要energy補正を消したのであり一般作用素4次ではない |
| Track C：basis・対角回路 | always-partialは共有を壊し悪化。singleton fallback→625共同isometry catalogueでRZ3.355%減、serial depth+.696%。既知control後のRTE固有fusion RZ利益は約.135%。[C-OLD] | 既知Pauli control／Gaussian／QR／parityの再発明ではない。Fullへも同機会。異basisの共同completion間を直接結ぶ追加効果は今回確認したC builderでは未評価（§5） |
| Track C：Closed後共同選択 | 最終両P3に183 PR条件、Full51条件。同じ18round構成を選択。Closedは旧同holdout比RZ6.03267655%、最適化canonical比2.11497621%減。[C-OPT] | h/resolution/α・整数ceil交換は実施済み。116条件は資格あり・価格未検証。有限座標探索でglobal最適性なし |
| Track C：Truncated PF | 元H groundを維持しHdrop＋PF誤差を会計。H2/H4/H6有限menuでは順にPR／Truncated／Truncatedが低RZ。[C-NEW] | H4は相殺を弱めた保守Bでも低RZ。H6はcombined AでTruncated<PR、BではTruncated>PR。全系一般化・相殺の任意state移送は不可 |
| 新A Reference・CT | 旧C-H4 canonical/compiler Aで強4-CX B′比CX削減≥4.681782%、native DAG depth削減≥40.995952%。RZ≥.150597%増、追加8qubit、shot8598→8606。m7/q128の2保存path・ε1e-11では1-shot T約1.87%減。[PREV] | 最新Closed＋joint＋fusionへの接続、全18round総Tは未資格。m17 word/guardの負結果を実際のRPE失敗や新合成必須の証明としない |
| Track D：Reference・cut・C1 | D-H4の13round主A点modelではCX11.1671%、depth54.6549%、RZ1.1200%減。C1実H4/q4一pathでは両方式RZ−23/CX−30、depth Strong−178/Ref−81。[D-C1][D-TOTAL] | C1の相対CX差は不変、depth差は97縮小。旧調査の「人工C1未評価」を再提案しない。cut selectorの事後min・価格oracle無料化不可 |
| Track D：q64・長q価格 | q64全8 path/32post/pairedχ完了。有限平均でRef CX10.962%、depth54.588%、RZ.987%減。[D-Q64] | 全law／q≥128／actual RPE未資格。既存13roundは約99.4–99.5%がq≥64 proxy。q64完了を全費用保証や再fitへ自動変換しない |
| 研究C：PR-aware/ENTRY DF＋Z再配置 | RC own-DF比較はCDF0/r64 RZ231951425282.5、P/r16 87514417699、E2/r16 75621395471。[RC] | P/EはNOMINAL_MODEL_ENTRY_LOWER、CDF0/EはUNRESOLVED_PRECISION_TARGET。元H ground・同精度のC-H4へ率やNを移さない。Track Cと別系列 |
| 研究B/B2-D、M1-S、C2 | 研究Bはsynthetic2mode/N1/prep込み単一signal、高次D／outer／RTE・強Full対照を検証し一時保留。M1-Sはshared± coherence平均と適用域付き理論・保存H4診断。C2-BはFull Y4の有限合成policy差.514072%、最適性gapが残る。[RB][M1S][C-OLD] | Track Bと研究Bは別。M1-S平均はP3 first momentと別、全RPE費用未確立。C2-Aの3round族交差なし・C2保留も保持 |
| O2推定器／周辺O6 | CV学習費・Born noise、単純LCU成功率補正後の負結果。最新非同値性選別は新たな実装推薦候補0。[O2][O2-DISTINCT] | 安価summary／測定accessが具体化するまで同じIS/CV/LCUを再実装しない。O6はideal unitary oracleとfinite weighted PR平均のgapが未閉鎖 |

## 4. 最新Closed P3の差を説明する保存値

以下は全てC-H4の[C-OPT]から転記。MC SEと量子測定の成功保証を分ける。

| 方式 | trial合計 | 総RZ | RZ MC SE |
|---|---:|---:|---:|
| 旧固定Closed（同じ最終holdoutで再評価） | 17028 | 1.387608544e12 | 7.110533e9 |
| 最適化Canonical P3 | 16946 | 1.332071607e12 | 4.943179e9 |
| 最適化Closed P3 | 16791 | 1.303898609e12 | 4.834427e9 |
| 強Full PF | 14225 | 5.647934021e11 | 0（決定論価格） |

Closed/Fullの保存比は **RZ2.308629、CX1.802077**。旧Closed→新ClosedではD RZは3.373971816e11→3.439699811e11と増え、RTEは1.050211269e12→9.599285356e11と減った。m17はh=.32、n1024→768、microstep8388608→6291456、log range .270397071→.360529421、rho下限 .762978948→.697246003、trial[253,253]→[303,302]。normalizationを悪化させてもevent削減が勝る例である。

旧bottleneckの「RTE basisがRTE RZの91.44%」はその段階の構成比。最終Closedへ同じ比率を代入しない。ただし最終にもRTE費用9.599285356e11が残るため、basis価格とrange/Nの両方を見る候補には直接の動機がある。ここから新たな節約上限・必要削減率は算出していない。

さらにC-H4のTruncated PFはcombined評価AでRZ3.337772238e11、保守Bで3.864575491e11。Fullだけに対する接近を、最安既存対照への勝利としない。[C-NEW]

## 5. SOURCEの再利用先と未接続部分

SOURCEはimport／実行せず、Git本文・関数定義・必要箇所のASTを読んだ。下表は流用の入口であって、新条件での資格認定ではない。元のtestやrunnerは今回起動していない。

| 資産／版 | 再利用可能な入口 | 必要な差分・適合条件 |
|---|---|---|
| C `c5a3c102` | `track_c_closed_p3_resources.{paired_paths,closed_native,qualify}`、`track_c_closed_p3.{application_events,flattened_runs,build_events}` | 同じ218成分・P3・256-bit law／weight／scalarを保持。`build_events`は最大same-basis runごとにbasis inverse→diagonal→basisを連結。異basis間relative joint bridgeの専用選択はない。compiler A既存相殺はbaselineに残す |
| C `c5a3c102` | `track_c_joint_isometry.{catalogue,append_run,build_joint}`、`track_c_diagonal_fusion.{emit,build,d_fused}` | 各列projector・support mapping・actual completionを保持。same-basis sharingを壊さない。未知集合はfull fallback。relative合成で安くなるかは未検証 |
| C `c5a3c102` | `track_c_closed_p3_joint_opt.{inventory,point,optimize,freeze,finish}`、`track_c_closed_p3_joint_opt_alpha.discrete_allocate`、`track_c_bottleneck.{allocate,rigorous_shots}` | 既存menu／α／共分散／独立holdoutを再実装しない。`rigorous_shots`はdirected Hoeffding十分式。新統計法へ旧保証を付替えない |
| D `141fa762` | `c1_h4_strong_baseline.StrongBuilder.{transition,sequence}`、`track_d_head_tail_h4.{certified_basis,splice_join,prepare}` | 既存G_follow†G_previousとanalytic vacuum gauge。full-frame D-H4からpartial completion C-H4への直接互換は未認定。新Gaussian合成器を作らない |
| 新A／D | `pr_reference_bprime.build_saved/rewrite_rte`、transfer/reference core、D whole-path/checker | uncontrolled factorization、actual inverse、vacuum χ、signed/scalar・X/Y・追加registerを保持。controlled nativeからcontrolを剥がす操作で代用しない |
| C truncation `da8b2c3f`→`92495a04` | `track_c_df_truncation*`のpublic replay/keep/native/qualification、`track_c_df_cancellation_*` | 元H、Hkeep、PFの三者を区別。source・input・ground bindingを使い、combined Aと個別Bを分離。三者D/R/dropのPR統計は別途必要 |
| A M1/M2／RC | matched provider、retained residual mapper/CAR、RC J/U export・Z再配置・fit/price pipeline | M2 model未採択、RC own-DF promiseを元Hへ変える必要。旧fit再実行や別Hamiltonian/stateへの無断差替えはしない |
| B G6/G7／O2 | `ReturnKernel`、G7 m3/m5 generator、finite-bit helpers、Parent/whole-path moment/range会計 | Closed P3実装済みを再作成しない。m5は新target。whole-path scoreのcheap conditional取得は既存kernelから得られたとはいえない |
| [D T0][D-T0] `c583abba` | `src/trottertracks/track_d_native_resources.py`、`tests/check_track_d_native_resources.py` | 確定native streamのcount／3 depth／max-plus transfer。人工29PASS。**NATIVE_EXACT＋P0_UNQUALIFIED**、未compile全pathのP0価格・量子性能改善ではない |

対応testsは各原報告と既存[再利用地図][REUSE]にある。特にCの`test_track_c_closed_p3*.py`／joint関連、Dのhead-tail／reference／original-H accuracy checksを優先する。CPU cacheやstreamingの改善を物理回路費用削減へ算入しない。

## 6. 保持するSTOP・未資格

- 初期Track C opt2の微小角省略、C-N長q proxyの不適合、C-S range爆増、C-P固定processor無改善、C2-A/B/C保留、always-partial悪化を保持する。
- Track B C1は旧P5 worker停止と別P5R完走・凍結を分離。Parent D＋E終了、S2負結果、P_Uの限定反例を一般不可能性へ拡張しない。
- 研究B一時保留、M1-S資源路線未確立、P-D／BF/BM／PR-3等の旧STOPを別候補名で再開しない。
- 新Aの全18round総T、m17 strict guard／word／domain／emitter／shellの未資格・保留は維持する。旧十分証明の不足は実際の失敗の証明ではない。
- D q64の旧数値FAIL/STOPは後続8本完了でも消さない。32有限postcheckは全law／q≥128／長q外挿保証ではない。
- RC actual CA／whole-law、A-M2物理model、A-B2価格、Pure-H rare価格を未取得0で埋めない。

今回、これらを解除・再資格判定していない。新しい候補の採択・科学実行は[最小検証案](MINIMUM_VALIDATION_PLANS.md)に対する別のユーザー判断が必要である。

[LEDGER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv
[REUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[PREV]: ../pr_rpe_cost_refocus_20261108/REPORT.md
[C-OPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md
[C-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/artifacts/track_c_closed_p3_joint_opt/2026-11-08/SOURCE.json
[P3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e38af3b12200c48c4f1b793638d223baff24e76/docs/research/track_c_closed_p3_20261108/REPORT.md
[B-SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a8299889de93c201bf21e727415d629f5e4b9f48/docs/research/track_b_existing_results_survey_20261108/REPORT.md
[BC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d48a6f2097ea81ff3b7c64114e30bef35b1b685/docs/research/track_c_b_connection_20261108/REPORT.md
[C-OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/100377dac8635fc49074147bfd8b42f52f2b7311/docs/research/track_c_o3_research_summary_20261108/TRACK_C_O3_RESEARCH_SUMMARY.md
[C-NEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3c81c2677988c7b9491eeee31fbcc0f9b29b078/docs/research/track_c_o3_integrated_20261108/TRACK_C_O3_INTEGRATED_REPORT.md
[A-C0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md
[A-M2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/docs/research/track_a_o1_m2_hermitian_b_science_20261106/REPORT.md
[A-B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/55fa5f5eb121ee01ae963da7ab6e5bbe9a507173/docs/research/track_a_b2_p1_strong_pf_branch_shots_20261107/REPORT.md
[PURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9b83b96b83f62142c9d87858dc24a94db7ac5d61/artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/REPORT.md
[O2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md
[O2-DISTINCT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bf88b251fd09caa0a227972ee0f1b56c5888e6a/docs/research/o2_distinct_mechanism_screening_20261108/REPORT.md
[D-C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/141fa7624d74aa5deacbd752d09df516cad9206f/docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md
[D-TOTAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md
[D-Q64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md
[D-T0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c583abba520a711895d21b7ec9590688349a7bd1/docs/research/track_d_t0_native_resources_20261109/REPORT.md
[RC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/709813805d41e470676e311cd63a740af23bf33c/docs/research/research_c_three_method_z_relocated_comparison_20261108/REPORT.md
[RB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/docs/research/research_b_final_outcomes_and_pause_handoff_20261026.md
[M1S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
