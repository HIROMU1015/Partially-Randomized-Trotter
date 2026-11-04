# 研究C・B1：Whole-kernel P0一様synthesis評価方法の技術具体化

2026-11-03。現在のPR/r32用adapter・人工回帰・非実行の未承認設計・passive preflightを完成した。
**技術adapterはGO、一様coverageと有用な実H4誤差上界は`P0_UNIFORM_NOT_ESTABLISHED`。**
Stage B科学実行契約は`executable=false`、calls/family数はnull。Stage C・重要科学レビュー・新実H4計算は行っていない。

今回閉じたのは、既存C2-A/ZSXX全domain composerへの現在grammar接続、構造的node/rewrite上限、
unchanged whole-P0の人工診断、full512/actual inverse/shell/repetitionへの接続である。
未取得の一様数値誤差を0に置いたり、有限familyの一致を全lawへ昇格したりしない。

## 対象・固定資料・保全境界

Linear H4・隣接1.00 Å/STO-3G、Nα=Nβ=2、sector36、number CDF rank8、元frame順LD2、partial ordered symmetric S2、
s1/r32/K0、h=δ=.1、m0..12、own-DF ground oracle、状態準備費用除外。Native対象はcontrolled full512、
P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation_degree1/qubits_initially_zero=false、
MPFR160 declared phase、既存Gaussian/relative融合、actual inverse、head/X/Y shell込みfenced serial。
候補SHA256は`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。

- 起点：[Stage A結果](research_c_iid_grid_stage_a_execution_20261103.md)、commit `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53`。
- Stage A科学SOURCE `ddd8333488e07a3b3af9095b5cb193ef9f90b3c3`、38/38 COMPLETEのrun01/STOPはそのまま保全。再実行0。
- [全law準備](research_c_all_law_signal_ca_readiness_20261103.md)、`3a784888a188db946681a6143c5e18479ddf284a`、実装 `ff80264447158c02d1f5d2f075e06f2ee8c970a4`。
- 正式共通指示原本 `research_project_instructions_unified_2026-10-25.md` の§8/§8.4/§10を実際に読み適用した。
- 独立branch：`codex/research-c-p0-uniform-method-readiness-20261103`。
- 独立worktree：`/tmp/prt-c-p0-uniform-method-readiness-20261103`。元dirty checkoutおよびB2領域は作業場所にしない。

## 再利用の確認と最小差分

以下の実sourceと対応する検証経路を確認した。古いr16/s4/h1/40の科学条件・許可は転用しない。

| 既存資産 | 今回の利用と限界 |
|---|---|
| `c1_h4_native_fair` | `compile_checked/decode`、full512/phaseを保持したP0。単一回路の検査から一様性は得られない |
| `c1_h4_full_synthesis` | `segment/propagate`。直接compiled/reference Frobeniusと両側norm-growth、shell/repetitionを再利用 |
| `c1_h4_strong_baseline` | `StrongBuilder.sequence/transition`、向き`V_following† V_previous`、analytic vacuum phase。Relative融合はhead内 |
| `c1_stage2_native` | `runs/combined_diagonal/actual_adjoint`。厳密な同一basisだけ共有、fsum、実際のgate inverse |
| `c1_h6_compiler_precision` | 既存MPFR160 marker/bind/cast。GateのEuler計算や全phase正規化を囲うものではない |
| `c1_h6_native_precision` | `gaussian`。8 RZ/最大28 XXPlusYYというtopologyと精度条件。Single-particle検査はfull512上界ではない |
| `c1_h6_synthesis_bound` | `circuit_identity/column_batch`。Phase/角度/配線を保持して重複をまとめ、batchサイズによらず列数を計上 |
| `c1_subspace_synthesis_bound` | `ColumnStream/repeat_bound`。全出力を保持。Sector制限には不変性条件が必要 |
| `c1_stage3_qualification` | 全supportと有限probeの区別、距離2の自明な包絡を有用なphase保証としない |
| `c_pr_aware_df_guarantee_strategy` | Captured Rust/ULP/有限scopeの資格拒否。固定回路事後検証は代替案として利用可能 |
| `c_pr_aware_df_stage_b` | `build_kernel`：両head＋r32 tail＋scalar/identity。Head/tail間relative融合なし |
| `c_pr_aware_df_stage_b2` | Full512 raw/compiled/actual adjoint、X/Y/shellの既存検査を再利用 |
| `c_pr_integrated_performance` | 完全なkernelを構築してからwhole-P0。保存native32は有限cohortの証拠 |
| `c_all_law_signal_ca` | `numerical_synthesis_rows`による現在s1/m0..12へのraw-complex伝播。共有moduleは無変更 |

さらに[既存ZSXX全到達domain設計](research_c_pr_aware_df_zsxx_uniform_domain_readiness_20261027.md)と
[既存保証方針調査](research_c_pr_aware_df_c2a_guarantee_strategy_readiness_20261027.md)を確認した。
`c_pr_aware_df_c2a.monotone_budget/symbolic_templates/compiler_coverage/trace_artificial`と
`c_pr_aware_df_zsxx_uniform.uniform_whole_bound/run_bound`が既に実装されていたため、同じengineを新実装しなかった。

追加は[B1専用adapter](../../src/trottertracks/c_p0_uniform_method.py)、[passive entry](../../scripts/prepare_research_c_p0_uniform_method.py)、
[人工test](../../tests/management/test_research_c_p0_uniform_method.py)の3 file。
旧`uniform_preview`はoccurrence=4×2^mを固定しているため、そのentryだけは現在条件へ流用せず、
genericな既存compiler composerと現在s1の`numerical_synthesis_rows`へ接続した。

## P0の実際の変換と入力領域

[Compiler/grammar記録](../../artifacts/research_c_p0_uniform_method/2026-11-03/technical/compiler_and_grammar.json)は
既存のpass graph検査で`PINNED_GRAPH_MATCH_NOT_NUMERIC_CERTIFICATE`。実H4回路は作っていない。
Installed Python source10 fileと公開Rustを照合対象にした。
[Rust一次source](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs)は
Qiskit1.3.0固定commit、[取得記録](../../artifacts/research_c_p0_uniform_method/2026-11-03/technical/external_source.json)にblobを保存した。
Version/source対応の確認であり、wheelの再現buildや一様数値誤差を認証したものではない。

| 経路 | 入力・変換範囲・残る条件 |
|---|---|
| 事前UnrollCustomDefinitions | 空EquivalenceLibrary、basis+uへofficial definitionを再帰展開。Opaque/annotated/control-flow/calibration等へ勝手に拡張しない |
| Init InverseCancellation | 固定のparameter-free inverse群。任意RZを近似比較して削除するpassとは扱わない |
| Translation | UnitarySynthesis/HLS/BasisTranslator。通常grammarにopaque unitary/HLS対象はなく、uをnative basisへ翻訳。数値parameter式・phaseは別誤差 |
| Opt1 1Q run | Event境界にfenceはない。CX等が区切るrz/sx/x runをRustで逐次行列積し、ZSXXで再合成 |
| Opt1 cancellation/loop | Target/error mapなし・basis閉包ならaccepted置換は短縮。CX消去で次のiterationに長いrunが生じ得る |
| 固定点検査 | Depth/Sizeがともに固定するまで反復。既存guard1000は超過時例外で、全word成功の証明ではない |
| Phase | 内部float Euler/definition phase、DAG加算、MPFR expression、最終bind/cast/remainder。Markerだけで前段float誤差は消えない |
| Inverse/shell | Compiledのactual inverseを別に構築。準備H/X/Y readout shellは別segment・fenced serialで、whole-shot再compileや新fusionを仮定しない |

決定論的な両half-headはkernel内に含む。既存伝播APIの`shells['head']`はancillaのH準備を指し、決定論half-headをもう一度加算するものではない。

既存SOURCE由来のrun domainは、理想RZ/SX/X productのdeterminantが`{1,i,−1,−i}`に入る4 cosetと、
run長依存のnear-unitary tubeである。Rewrite後のRZを次のrunへ再入力する場合も含める。
Euler poles、arg cut、θ≈0/π/2/π、RZ省略、SX/X、mod endpointの両側を除外できない。
`ANGLE_ZERO_EPSILON=1e-12`やfinite residual最大値は、全domainのoperator誤差定数ではない。

## 現r32の構造上限と誤差の接続

旧topology・Uの5-node翻訳witnessを再利用したSOURCE構造上限であり、実H4 gate数・費用の測定ではない。
固定standard definitionとequivalence graphの下で、XXPlusYY≤44、CRZ≤4、controlled RZZ≤106、P≤5、RZ/CX≤1 node。
8 mode Gaussianは最大8 RZ＋28 XXPlusYY。Tail最大64 Gaussian、head4 Gaussian＋4 relative、合計72。
Head diagonalは6、K0 centralは32、追加Pはheadのliteral zero-time scalar2・event phase1・total scalar/identity1を保守的に含める。

`N ≤ 72(8+28×44) + 6(8×4+28×106+5) + 32(2+4) + 4×5 = 107522`。

1Q node上限100914、CX上限6608。Zeroや同一basis sharing、init cancellationは減らす方向であり、旧r16のN=67731を流用していない。
Symbolic template traceはこの構造議論の回帰であり、有限traceだけを全parameterの数値証明としない。

既存`monotone_budget(N)`より、accepted rewrite R≤N、accepted入力gateの延べvisit≤N+5R≤6N=645132。
一つのrun長は1Q node上限以下、generic composerでは安全にNを使う。
成功して返る経路のiterationは`min(1000,N+1)`以下で、candidate matrix productの保守的上限は1000×100914。
**Rejected candidateの計算は出力を変えないため、その誤差をaccepted誤差へ加算しない。**
このcountは全wordでguard内に正常終了することや、有用なoperator誤差を証明しない。

条件付き式は既存engineのまま：

`eps_P0 ≤ eps_translation + 6N p/(1−Np) + N zeta(D_N) + eps_phase`（Np<1）。

pは既存binary64逐次積モデルのrateで、全domainのnode行列評価誤差aと基本算術条件に依存する。
zetaは全到達D_N・全branchに対する、phase込みの実Rust M→V差。
ζのlocal phaseを含む項と、追加のphase累積・正規化誤差を重複計上しない。
Raw Gaussian/relative/diagonal→exact saved-atom誤差、actual inverse、shell、数値referenceの未包含分は別に必要。
直接G→V誤差を使う場合はproduct＋ζを重ねない。

現在はa・ζ・translation・phaseの必要な入力がnullで、composerも`UNRESOLVED`を返す。
MPFR/Arbの精度を増やしたり、held output box・captured pair・ideal template imageを渡したりしても未観測Rust出力の一様性は得られない。
今回の第一目標は数値的十分条件なので、厳密包囲・wheelのbitwise一致を一律の必須条件にはしない。
それでも、全wordへの適用が正当化された数値誤差モデルは必要であり、sample ULPや残差から自動的に仮定しない。

## A/B/Cの比較、full512・inverse・signal

| 方法 | Coverageと有用性の判定 |
|---|---|
| A：構造的一様法 | Grammar/count/domainと条件付きcomposerは存在。全run数値モデル・初期phase/translation・全word成功が不足し、実H4一様boundは未成立 |
| B：有限family診断 | 境界・phase・fusionをunchanged P0とfull operatorで調べられる。有限familyだけでは全lawを覆わない |
| C：Hybrid（第一候補） | Aを主証拠、Bを実装整合に使う設計は可能。ただしBでAの不足を埋めた扱いにはしない。現科学実行はNO-GO |

任意のkernel operator boundとraw complex signal boundは別である。B1 adapterは既存`propagate`を介して
kernel＋head/X/Y、both-norm growth、n=2^mを伝播する。異なるwordの各occurrenceにも同じ一様boundがあればtelescopingが適用できる。
Actual inverseの誤差・normも含むkernel guardを要求し、理想adjointとの置換で済ませない。
Control1側のscalar phaseは相対位相になるため落とさず、phase fitはしない。

Full512を選べばsector外入力と全leakage出力を保持する。36 sector／control込み72列だけを選ぶ場合、
reference不変性と実際の全出力を保持した制限作用boundが別に必要であり、full512と同じ資格にはしない。
H6 complete-column法も「その入力基底と回路」の数値boundで、全wordへは自動拡張しない。

`conditional_rows`はsupplied uniform argumentのscopeを検査して既存engineを呼ぶが、schemaは証明の正しさを認証しない。
返り値の`actual_H4_qualification/actual_CA_achieved/rigorous_CA`はnull。
Synthesisだけで`sin(pi/12)`を超えれば有用性を拒否し、下回ってもPF/RTE・attenuation・reference・categorical・DF・統計を含むRPE PASSとはしない。
現在の実H4 phase roomやshotsを再評価しておらず、候補・r/h/K・compilerを結果に合わせて変更していない。

## 未承認設計・logical calls・代替案

[未承認設計](research_c_p0_uniform_method_contract_unapproved_20261103.json)は`P0_UNIFORM_NOT_ESTABLISHED`／非実行。
科学entryはpermission flagがあっても`NO_H4_EXECUTABLE_CONTRACT`で拒否する。新しい科学outputは未作成、retry/resume0。
未知のfamily数や評価アルゴリズムを埋めないため、science call上限はnull。

将来F個の有限診断wordを別に固定したならbuild F・whole-P0 compile F、重複を`circuit_identity`でまとめる。
G個の異なるfull512 mapをstate-actionで評価するなら、既存H6 `column_batch`は**512G logical columns**を計上し、
batchサイズやworker数で科学call数を減らさない。Raw/reference/inverse/shell・証明処理の対象GやFは未確定で、今回総数は出さない。
9 basis・4 directed head transition・216 atom supportは入力grammarであり、最終P0の閉じたfamily数ではない。
旧19/47/1231/38/16-call予算は転用しない。全law入力が得られた後の13 round伝播やStage Cも未実行。

代替は既存`fixed_certificate`による保存回路の事後検証、事前固定IID holdoutによる期待誤差/tail評価、または新compiler政策。
前二者は有限instance／確率的な主張で、一様保証とは異なる。旧文書のB1/B2/B3は比較案の名称で、今回の担当B1/B2とは別である。
現candidate用の入力・scope・calls・別許可なしに、旧19-callや保存native32を再実行しない。Compiler/backend変更も今回採択していない。

## B2とのinterface

今回B2のworktree・SOURCE・出力・cache・lock・processへ書込みや操作をしていない。統合は双方の公開後。
候補SHA、元科学SOURCE、Stage A SOURCE、h/r/K/s、ordering、phaseとtargetを揃える。

- B1のtarget：exact saved-atom trajectory mapに対する、phase-sensitive full512最終P0/actual inverse/shell。
- B1の将来出力：m0..12の`quantity=synthesis,norm=raw_complex_signal,bound,kind,provenance,rounding_enclosed`。現値はnull。
- B2：saved-CAR bridgeと、bridgeに含まれないreference数値不確かさを別の13行で示す。
- Raw Gaussian/native構成誤差、atom→CAR差、reference評価誤差の所属を明示し、同じ項を二回足さない。
- PF/corrected-RTE/DF/categorical/PCG64/統計はB1 synthesisへ混ぜない。特にDFはenergy段階だけで一回。

## 人工検証・固定SOURCE・公開・STOP

[テスト記録](../../artifacts/research_c_p0_uniform_method/2026-11-03/technical/tests.json)：**新34＋既存218＝252 unique PASS、旧2件deselect**。
Known perturbation、cross-event run、既存relative Gaussian/controlled phase/actual inverse/head-X-Y、coherent repetition、
full512の非sector列とleakage、有限family拒否、loose bound不成立、SOURCE/input/compiler drift、fresh/retry/resumeを確認した。
Whole-P0人工診断は既存callback・compiler・segment・actual inverse検査を使った。

旧Stage Aの「まだfreshなのでpreflight PASS」1件は実行済みのため適用外とし、新testで再起動拒否とSTOP不変を確認した。
旧C2並列testの表出例外順序1件は既知の不安定性を再現したため、既存の両順序primary-journal test2件を採用した。
旧test・SOURCEは変更しておらず、初回観測と除外理由をJSONに残した。全旧suiteが無条件PASSしたとは主張しない。

- 実装固定commit：`5fc9e6a18e016bfa0725d18edd1c7fc4589a0075`。
- [SOURCE163 file](../../artifacts/research_c_p0_uniform_method/2026-11-03/technical/SOURCE.json) SHA256：`a5c4b5fa5bae245ccbaaea7bf128a28a3cb0dd20c03b5de171dd74fef21feac9`。
- 契約SHA256：`8ed526a910c9ccec86439121e0ae38dc151ee2e29a4b69ff9c1023fcb2bc39fd`。
- [Passive preflight](../../artifacts/research_c_p0_uniform_method/2026-11-03/technical/passive_preflight.json)：数値library importを遮断してPASS。保存input25 file・installed compiler Python source10 fileを照合。
- [技術STOP](../../artifacts/research_c_p0_uniform_method/2026-11-03/technical/STOP.json)：新実H4 science/array decode/build/compile/sampling/Stage C/重要科学レビュー0。
- [静的保全確認](../../artifacts/research_c_p0_uniform_method/2026-11-03/technical/static_verification.json)：旧SOURCE/raw/契約/STOPと元HEAD/index/dirty状態を保持。

CPU affinity32・available RAM約52GiBを確認。小さい人工回路とinvented512行列を使い、重い実H4検証はしていない。
将来は既存resource_plan/ColumnStream・immutable input共有・column/task並列とBLAS調整を再利用する。
CPU/RAM/worker/wallの人工cap、共有venv/OS/global cache/他jobの変更はない。実測していないspeedupは主張しない。
科学的な新結果・モデル採択・Claim資格の変更はなく、旧Evidence Map、研究概要、validation manifestの科学記録はそのまま保持した。
公開後、remote SHA/tree・報告本文・主要blobを照合する。

**Mandatory STOP。** 次の担当はユーザー：不足する全domain数値モデルを準備する技術方針、または別の有限/確率的scopeを選ぶ。
Codexはその具体的な次指示後に必要な設計を進める。現在、実H4一様評価の実行許可だけを受けても起動可能な契約にはなっていない。
B2との統合・新科学実行・Stage C・重要科学レビューは自動開始しない。
