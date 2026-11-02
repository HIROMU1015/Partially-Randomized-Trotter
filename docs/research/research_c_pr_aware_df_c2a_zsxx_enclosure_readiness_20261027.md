# 研究C：C2-A・ZSXX 2×2 validated numerical enclosure

2026-10-27 JST。起点`93d5d875ccb37bfb60775ad64be79a6048b75ed8`。

**実Rustから取得した入力・出力の固定組について、位相込みoperator距離のvalidated enclosureを実装した。全lawを覆う一様上界は未完成。** 人工41例、実行した人工1qubit whole-P0 2例とそのactual adjoint 2例を囲えた。新規52＋既存回帰109＝161 tests PASS、旧順序依存test 1件は理由を明記して除外した。実H4科学callは0。

[新契約草案](research_c_pr_aware_df_c2a_zsxx_contract_draft_20261027.json)は`BLOCKED_PARTIAL_VALIDATED_DOMAINS`、`executable=false`、科学call上限null。これは部分的な技術成果であり、旧C2-Aの実行不可契約、C1の点推定、C2-B/Bridgeの資格は変更しない。

## 1. 評価方法と有効性

[実装](../../src/trottertracks/c_pr_aware_df_zsxx_enclosure.py)はQiskit 1.3.0の実`unitary_to_gate_sequence(M,['ZSXX'],0)`、または同じRust emitterの`generate_circuit(EulerBasis('ZSXX'),theta,phi,lambda,beta,True,None)`を呼び、返されたgate角度とglobal phaseをbinary64のhexで固定する。Compilerを再実装しない。

固定入力Mの各成分をexact dyadicとして扱い、返された角度・phaseも同じくexact dyadicとして解釈する。Python-flint 0.8.0／FLINT 3.3.1のArb/acb ball arithmetic、192 bitsで、exact RZ/SX/X/P/RYの数学的operatorを独立再構成する。SX/Xは有理複素数、RZ/Pはvalidated複素exp、RYはvalidated sin/cosで評価する。自由なphase fitはしない。

差D=M−Vの各entryを囲い、

\[
\max_{ij}\operatorname{lower}|D_{ij}|\leq\|M-V\|_2
\leq\sqrt{\sum_{ij}\operatorname{upper}|D_{ij}|^2}
\]

の右辺をoutwardに計算する。Frobenius上界なので数値SVDの厳密性を仮定しない。最終upperは`upper().man_exp()`からexact rational文字列にし、floatへの丸めで過小評価・underflowさせない。例えばbinary64より小さい正の有理上界も0にしない。

この証明は**固定されたRust出力が表す数学的operator**について成立する。Rustのatan2/argが全入力で何ULP正確かは不要であり、返された値の誤り自体も距離に含まれる。一方、未観測の入力に対するRust出力の一様保証は得られない。後検証可能性と事前の一様誤差モデルを分ける。

`generate_circuit`の基準は、PSX/ZSXX parameter conventionに従う

\[
G(\theta,\phi,\lambda,\beta)=e^{i[\beta+(\theta+\phi+\lambda)/2]}
R_Z(\phi)R_Y(\theta)R_Z(\lambda).
\]

M→Eulerを省いたemitter単独試験であり、この試験だけでM→Euler変換誤差を検証済みにしない。全行列入力試験はM→実Rust出力を直接囲う。

ライブラリの信頼根拠は[固定0.8.0の説明](https://github.com/flintlib/python-flint/blob/0.8.0/doc/source/general.rst)、同版の`arb.pyx`／`acb.pyx`／`acb_mat.pyx`、[FLINTのmatrix API](https://flintlib.org/doc/acb_mat.html)。`abs_upper`、`upper`のoutward処理、exact `man_exp`、通常の`acb_mat_mul`をSOURCE確認した。`*_approx`は使用しない。[SOURCE抜粋・Git blob](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/upstream/python_flint_api_sources.json)とMIT LICENSEを保存した。FLINT/GMP、Python binding、hardwareの実装契約を信頼するvalidated-library計算であり、C実装やhardwareの独立formal verificationではない。

## 2. 算術モデルA／B／Cと手法選択

[算術モデル](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/arithmetic_model.json)／[環境・wheel hash](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/environment.json)。新しいtask-local venvにpython-flintだけを追加し、旧evaluation site-packagesはread-only参照。共有venv、OS設定、cache、他jobは変更していない。

| 区分 | 確認・採用した範囲 | 未確認事項と今回の扱い |
|---|---|---|
| A：基本演算 | finite binary64を正確なdyadicとして受領。実processのFE_TONEAREST、subnormal probeを確認。通常のIEEE加減乗除モデルは適用条件付き | Rust wheelのFMA/contraction・全経路のsubnormal policy・toolchainは未確定。任意runのgamma評価には未採用 |
| B：Rust/platform数学関数 | 固定Rust SOURCEとsystem libm/glibc 2.35へのlinkを確認 | atan2/arg/trig/complex abs・remainder等の全入力ULP保証は不明。FMA等もnull。Aや有限点Cで代用しない |
| C：高精度比較 | mpmath 150桁の独立SVDと9行列例のArb区間を照合 | 診断専用。これを囲い込み証明や全入力保証にしない |
| Captured pair enclosure | exact input/output bitsをArbで直接囲う | Bの一様精度仮定を必要としないが、未知入力・再実行は覆わない |

[Rust標準資料](https://doc.rust-lang.org/std/primitive.f64.html#method.atan2)も数学関数のprecisionを一律保証しない。参照した現行資料のversionをinstalled wheelのtoolchainと取り違えない。`sqrt`等も、どのplatform routineとコンパイル条件かを特定せず保証済みとはしない。

| 候補手法 | 今回の判断 |
|---|---|
| Hardware directed rounding | libm全関数の方向丸め保証にはならず、processの丸めmodeを変更しない |
| Exact rational | 入出力binary64とexportに採用。超越関数の出力には単独では不足 |
| Interval／ball＋validated elementary functions | Arbを採用。既存の小行列APIを利用し、新しい数値計算ライブラリは作らない |
| 条件付き解析評価 | 既存unitary telescope／R≤N、visits≤6Nの構造に再利用。局所一様定数は未取得 |
| Branch subdivision | 将来の全領域評価候補。今回の有限境界例・held-output boxはcompilerの区間像ではない |
| Independent high precision | Cの診断として採用。上界証明の代わりにしない |

## 3. DomainとZSXX分岐

[Domain一覧](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/domains.json)／[Rust branch inventory](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/branch_inventory.json)／[全人工入出力・exact bounds](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/artificial_enclosures.json)。

Rust SOURCEのactive ZSXX routeを調査した。`ANGLE_ZERO_EPSILON=1e-12`は分岐閾値であり誤差保証ではない。

| 数値経路・実判定 | 人工確認と数学的注意 |
|---|---|
| detのcomplex積・差→arg、entry abs→atan2、entry arg→Euler | I/X/Z/SX/iI、near-unitary、perturbed entryでM→実出力を囲う。Euler poleで角度単独の安定性を要求しない |
| `abs(theta)<atol` | theta≈0、閾値のbelow/center/above。P(lambda+phi)への近似 |
| `abs(theta−PI/2)<atol` | one SXとtwo SXの両出力patternを確認 |
| `abs(theta−PI)<atol` | lambdaのphase移動、phi−lambda、X shortcutを確認 |
| `abs(mod(lambda+PI))<atol || abs(mod(phi))<atol` | lambda側・phi側、exact phi=0、各cut近傍。角度反転とphase変更を同時に検証 |
| `abs(mod(theta+PI))<atol` | SX×2→X。SX²=Xはexact、中央P省略は近傍では近似 |
| RZ callbackの`abs(mod(a))>atol` | omit/emit両方。P(a)=exp(ia/2)RZ(a)なのでphaseも同時に扱う |
| `(a+PI).rem_euclid(2PI)−PI`とendpoint clamp | cut両側、−PIへのclamp。RZ(a+2π)=−RZ(a)をphaseと切り離さない |
| Rewrite採用条件 | basis閉包・targetなしでstrict gate数減少。不採用runは変更されない |

詳細にはSOURCE行、判定に必要な演算、gate列、phase変更、数学的恒等式、数値誤差源、人工case対応をinventoryに記載した。**SOURCE上の全分岐調査と、全数値domainの囲い込みは別**である。Branch instrumentationで全組合せ到達を証明したわけではない。

適用したのは次の領域に限る。

- Angle emitter 28固定tuple：7 cut familyの各3点、identity／π／generic／signed zero／phi flip／clamp／subnormal。
- Matrix→Rust 9固定点：exact dyadic unitary I/X/Z/SX/iI、signed zero、subnormal perturbation、near-unitary、微小entry perturbation。
- 2 reference region：matrix各実虚entry、またはthetaの半径10⁻¹⁵をArbで囲う。ただし**Vを固定したreference uncertainty box**であり、box内のXすべてについてcompiler(X)を囲ったものではない。
- 長さ1024の相殺RZ列と長さ768の非自明列。理想gate積G→人工NumPyが作ったbinary64 M→実Rust Vを分離する。Mがprivate Rust `matmul_1q`のbit結果と同じとは主張しない。
- 実行した人工1qubit whole-P0 2例とactual adjoint 2例。初期translationとMPFR160 phase wrapperを含むその固定例だけの証明。

Matrix centerはvalidated `||M†M−I||F upper≤1/4`を入場guardとする。これは人工verifierの対象範囲であり、実Rustが受け取る全領域を証明した閾値ではない。形状／非有限値はRust前に拒否。ゼロ行列など特異・guard外は`UNRESOLVED`。非有限Rust動作、任意の大角度、全near-unitary入力、すべてのlibm/FMA変種は未coverage。

## 4. 人工enclosureの大きさ

以下は上記人工captured pairに対する**validated upperの近似表示**。規範値はJSONのexact rationalであり、有限例の最大値を一様定数へ転用しない。

| 対象 | Operator upperの例／最大値 |
|---|---:|
| Angle emitter 28例の最大 | 約9.99834e-13 |
| Matrix→Rust 9例の最大（near-unitary） | 約4.01944e-14 |
| 長さ768：G→人工M | 約2.84673e-14 |
| 同：人工M→V | 約2.84564e-14 |
| 同：G→Vを直接囲った値 | 約1.00572e-15 |
| 同：triangle上界 | 約5.69236e-14 |
| 長さ1024の相殺列：G→V | 約5.90576e-55 |
| whole-P0：translation_phase | 約2.24626e-15 |
| whole-P0：cross_boundary_run | 約8.95531e-16 |
| 実actual adjoint：各例 | 約3.46383e-16 |

直接G→Vとtriangleの両方を加算しない。長さ1024の極小上界も固定のexact逆角度列だけについて成立し、任意runの誤差を表さない。

Phaseを意図的に削除したiI反例はoperator距離lower>1で`CORRECTNESS_FAIL`になり、controlled人工回路ではX/Y信号が(0,1)から(1,0)へ変化した。これは故障注入の期待結果であり、実Rustのcorrectness FAILではない。

開発中、768因子の逐次Arb積でinterval wrappingが膨らみ、validだが約4.86e31という無用なupperになった。検証referenceの括弧付けだけをbalanced積に変え、同じ192 bitsで改善した。Exact chronological product、実compiler、物理条件は変更しない。[初期出力](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/artificial_enclosures_initial_leftfold.json)と[経緯](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/development_note.json)を保全した。巨大upperは実誤差の観測値でも科学的FAILでもない。

## 5. Whole-P0への接続と不足

[伝播条件](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/propagation.json)。既存`monotone_budget`、`uniform_preview`、Bridgeのexact rational preview、既存`repeat_bound/round_synthesis`のunitary telescopeを再利用・確認した。

固定P0・basis閉包・target/error_mapなし・ZSXXのみ・parameter-free cancellation・calibrationなしで、採用rewrite R≤N、採用runのmatrix入力総数≤N+5R≤6N。これらは回数の証明に留まる。

各採用rewriteについて、exact ideal run Gと実際のreplacement Vのphase込みδを有効に囲めれば、exact unitaryを前後に掛けてもoperator normは変わらず、contextへの埋込みを含めてΣδで伝播できる。新`accepted_word_budget`は採用recordだけを加算し、回数条件、phase-fit禁止、直接boundのscopeを検査する。不採用candidateの残差は加算しない。採否が丸めで変わる事前保証には両可能経路を覆う必要がある。

Whole-wordの形は

`epsilon_translation + sum(accepted direct ideal-run→V bounds) + epsilon_phase_accumulation`。

局所δをG→M＋M→Vで作るなら一度だけ加算する。G→V直接boundと重複加算せず、6N×有限点sample最大にも置換しない。Near-unitary Mをunitaryと仮定して反復しない。Exact ideal gate product GとVを比較することでunitary telescopeの前提を維持する。

Composerは外部証明入力を認証する機能ではなく、`CONDITIONAL_ENCLOSURE`まで。Translation／phase未知ならnullを返す。Captured whole-P0 2例は、実行したその1qubit例の初期translationを含むが、全native wordの証明を代用しない。

未解決項目：

1. SOURCEから到達可能な全run長・角度・丸めmatrixのdomainと、一様ZSXX operator bound。基本演算／FMA、libm精度またはそれを避けるbranch-level enclosureが必要。
2. Official gate definitions、BasisTranslatorの選択rule、numeric parameter演算、controlled decomposition、phase補正、MPFR160最終bind/cast、回路phase累積の囲い込み。
3. Gaussian/relativeのfull-Fock vacuum phase、raw angle／scalar、CAR/expm/reference、head/X/Y shell、実際のinverse生成の別上界。
4. 上記を全wordへ適用した後のround反復。C2-B lawとCAR Bridgeは別項とし、観測残差を厳密reference uncertaintyへ昇格させない。

m12の既存unitary telescopeは小誤差域でkernel誤差を65536倍し得る。今回のe-15〜e-12の固定例だけから、任意run・多数rewrite・translationを含む全lawで有用になるとは判定できない。実H4のN、phase-room、shots、totalは新たに計算していない。

## 6. テスト・再利用・保全

[テスト集計](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/tests.json)／[新規52 PASS log](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/zsxx_tests.log)／[既存回帰109 PASS log](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/regression_tests.log)。

新規ではfinite branch/cut、strict bound export、subnormal、near-unitary guard、unknown runtime/rounding拒否、phase-fit拒否、phase反例とcontrolled X/Y、長いrun、whole-P0／actual adjoint、採用recordのみの伝播、missing/null、入力hash不一致、passive preflight、science execute option不在を確認した。Synthetic PASSを実H4の科学的成功と扱わない。

旧C2の`test_artificial_E2E_exact35calls_and_failure_stops`は、`as_completed`で原REPLAY例外とdependent `Stopped`のどちらが先に表出するかに依存する。今回はこの1件を明示deselectし、**両順序を強制する2テスト**でprimary journalのREPLAY ValueErrorとSTOPを検査した。旧runner/testは変更しておらず、非決定性を修正済みとはしない。[方針](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/regression_policy.json)。旧初回FAIL logとSTOPもそのまま保全。

| 既存資産 | 今回の再利用 |
|---|---|
| C2-A `trace_artificial/check_graph/compiler_coverage` | 固定whole-P0 trace／profileとSOURCEの検査。新compilerなし |
| C2-A `monotone_budget/uniform_preview` | 条件付き回数とnull/phase/full-word coverageの接続 |
| `c_pr_aware_df_stage_c2_audit`／`c2_bridge` | SOURCE来歴、独立reference／lawの分離、exact rational伝播 |
| B2 `SharedQiskitCalls` | artificial build/compile/actual adjointの既存mutex |
| `c1_h6_compiler_precision` | 固定P0／MPFR160 declared-phase wrapper。Policy変更なし |
| Python-flint／mpmath | validated enclosureと診断を分離。Arb/acb以外の数値ライブラリを再実装しない |

既存にはexact captured Rust outputをvalidated ballで評価する経路がなかったため、opt-in verifier／固定人工fixture／passive identity helper／CLI／testを追加した。旧sourceへのpatch・新Gaussian emitterはない。

[Passive preflight](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/passive_preflight.json)はSOURCE107、旧input/provenance26、固定Qiskit Python SOURCE14、package11を照合する。NPZ decodeと数値評価entryを禁止した状態でもPASS。Fresh科学outputは未作成。[保全確認](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/preservation.json)は旧src/scripts/PR-aware証拠1865件のうち1863件byte不変、2 READMEは入口追記のみ。旧SOURCE/raw/grant/ledger/marker/STOPは変更しない。

独立worktree／task-local cacheを使用。小さい2×2/1q人工計算と2独立suite processで十分であり、GPUを必要としない。Arb precision contextはprocess-local lockとworkprecで復帰する。利用可能資源は環境資料に記録し、CPU/RAM/worker/wall上限は追加していない。相当する実分子負荷のspeedupは測定・主張しない。

## 7. 新契約・Q1〜Q5・次の担当

- Branch：`codex/research-c-pr-aware-df-zsxx-enclosure-preparation-20261027`
- [SOURCE107](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/SOURCE.json) SHA256：`c02e3a5c19a748ca271a36cc3c9d77bb5584cb5c6dcfedd435c388e63a7822e0`
- [未承認契約](research_c_pr_aware_df_c2a_zsxx_contract_draft_20261027.json)：SOURCE、旧input、P0、precision192、rounding、評価定義、分類、fresh namespace、retry/resume0、STOPを固定。実行可能なcoverageが不足するため、科学call分類／最大数／H4 allowanceはnull。
- [技術STOP](../../artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/STOP.json)：新規実H4科学call0。科学承認の記録・証明書・grantは作らない。

**Q1：構成できた。** 実Rustから返された固定outputと明示reference領域について、phase-sensitive validated enclosureを構成した。単なる高精度点比較ではない。

**Q2：部分coverage。** 明示41例と人工whole-P0/adjoint各2例。全SOURCE分岐を調査したが、未入力のbranch近傍や全near-unitary領域、すべてのRust math-library実行は覆っていない。Held-V boxをcompilerのinterval像とは呼ばない。

**Q3：条件付きで接続できる。** Exact unitary telescopeと採用rewriteの直接boundをΣする。Initial translation／phase累積／一様run bound／raw-reference-shellの全coverageが不足し、whole-P0／全law資格はnull。

**Q4：局所事後検証には実用的な大きさが得られた。全lawの見込みは未判定。** 事前に全到達domainを囲うには、run積／branch／phaseの長さ依存モデルか、安全なSOURCE領域制限が必要。有限最大値や構造回数だけでは代用できない。実H4のm12で有用かは今回判断しない。

**Q5：現時点で新しい実H4科学実験は推奨しない。** 次の最小作業はtechnical Step Aの全到達domain・run長依存上界とStep Bのtranslation/phase enclosureに絞る。その後に必要な有限primitive/reference A1とwhole-P0 A2だけを固定する。現契約は実行不可で、call上限を架空に設定していない。SOURCE／precision／retry/resume／STOPは固定済みである。

次は**研究C／ユーザー**がこの部分保証と残る技術方針を確認する。**Codex**は具体的な次の指示後、上記不足に絞ったtechnical作業を行い、実行可能な科学契約を作成できた場合だけ別途明示許可後に実H4へ進む。今回のcommit/push・remote/SOURCE/成果物照合後は**mandatory STOP**。新規実分子計算、C1再解析、sampling、shots/総資源、別分子、GPT詳細科学レビューへ自動進行しない。
