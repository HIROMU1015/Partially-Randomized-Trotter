# H4 K静的保証調査・L production実装

2026-10-23。専用branch `codex/non-df-h4-kl-production-20261023`、基点 `669a6ee576a032e8bf54d7ff47d570cc70d7eca5`、直近adapter SOURCE `14c0f8b0adda6160d507807334b82da0531a0e35`。**新技術SOURCEは `eb195bf2dedd0aed9d80ad7f84948b59ea53d854`**。C科学SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`、元科学SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`、T SOURCE・結果・旧grant/STOPを変更していない。

**Kは限定的なexact identityとcompiler全体の証明を分離し、旧programの保証をMISSING_EVIDENCEのまま維持した。Lは既存Cの数値・監査helperを流用して、128 bit固定の局所gate診断とone-shot runnerを実装・封印した。実H4のL数値実行は0、科学grant発行0。** 新compiled系列、正式受理A、N/G、詳細科学レビュー、最終evidence freezeは開始していない。

## 固定資料・運用・流用

GitHubで基点branchが669a6ee57を指すこと、基点commit/tree、14c0f8b0a commit/tree、必須23 fileのbytesを実確認した。大きい旧gate_bindings.jsonはcontents APIのcontentが空のため、固定blobを取得して全bytesを照合した。[GitHub記録](../artifacts/non_df_h4_kl_production/2026-10-23/github_verification.json)。新SOURCEの公開状態は別に扱う。

最新版共通運用指示は、ユーザーが追加した `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-10-23.md` の§8.4・§10を読み適用した。[受領SHA](../artifacts/non_df_h4_kl_production/2026-10-23/instruction_receipt.json)。当初の所在未確認を解消し、旧版を運用正本として併用していない。

[流用・適合性記録](../artifacts/non_df_h4_kl_production/2026-10-23/reuse_audit.json)：

| 既存資産 | 流用した機能 | 今回の差分 |
|---|---|---|
| h4_ball_evidence.py / C SOURCE | exact binary64 import、Acb matrix/phase、Frobenius上界、dyadic endpoint | full-H・reference・law・bias関数を呼ばず、2×2/4×4だけを使用 |
| h4_enclosure_prototype.py | exact rational product_error_bound、tiny exponential oracle | 2因子のexact合成後、各段で128 bitの外向き丸め。長い分母増大を抑制 |
| h4_certificate_control.py | fsync/exclusive save、SHA、locked非返却CallBudget、owned subreaper witness | L専用authority・予算・固定128を追加。隔離witness内で新receiptのscopeラベルだけをLへ適合 |
| 前回verify_h4_evidence_binding.py | 既知4case・固定runtimeのbounded compiler調査 | passごとの適用条件・位相・精度・未証明事項を追加 |
| 既存T67＋binding33 test | schema、phase/control/basis、scope/SHA、二重計上等 | 原本を変更せず、新L48 testを追加 |

既存C runnerはfull-HとC固有の精度schedule・grant/予算を扱うため、そのままLとして実行できない。新kernel/controllerはこの差分だけを別SOURCEとして追加した。別研究の結果・grant・科学承認を流用したものではない。

固定科学対象は従来どおりlinear H4/隣接1.00 Å/STO-3G、8 system qubit、Nα=Nβ=2、Pauli184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body、S2/T=.8/q8/r4/K4、4方式×5精度×両軸。H2のS2/Yoshida4系列と混合していない。

## K：SOURCE-level保証の成立範囲

[拡張調査JSON](../artifacts/non_df_h4_kl_production/2026-10-23/K01/K_report.json)、[pass対応CSV](../artifacts/non_df_h4_kl_production/2026-10-23/K01/K_pass_map.csv)。前回調査を流用し、固定Qiskit1.3.0、rz/sx/x/cx、optimization1、seed17、元wrapper/compilerを維持した。候補SOURCE経路と、旧programで実際に実行されたpass列を分離する。後者はreceipt不在のためUNKNOWN。

| 変換 | SOURCEで確認した条件・数学的意味 | 残る保証 |
|---|---|---|
| InverseCancellation | 設定されたself-inverse/inverse pairに対する隣接消去。exact A†A=I、Pauli/H等の恒等式はTで確認済み | Pythonからaccelerated matcher/DAG処理を呼ぶ。全SX/T/CX等のpair、実matching・mutation、旧pass instanceをTだけで証明していない |
| Optimize1qGatesDecomposition | 1Q runを集約しEuler gate列を選択。global phaseを新DAGへ保持するSOURCEがある | binary64のrun matrix、accelerated Euler分解・角度選択・丸めの一様保証がない |
| Controlled decomposition | definitionをunrollしglobal phaseを集める。control後はrelative phaseとして保持が必要 | 実分解・pi/angle/phase蓄積の数値誤差、全control-state/ancilla条件が未証明 |
| BasisTranslator/equivalence library | basis集合を探索しequivalenceに従ってDAGを書き換える | 全rule・parameter/global phase・accelerated変換・旧探索結果の網羅がない |
| HighLevelSynthesis | high-level operation/target/plugin等に応じたlowering | plugin選択・前提・分岐・数値処理・旧traceが未取得 |
| UnitarySynthesis | translatorが候補経路を含む。exact/approximationを区別する必要 | 実際の呼出有無、approximation_degree/数値kernelの保証がない |
| DF basis/control/wrapper | 逆順inverse、diag(I,U)、H-CU-H / H-CU-Sdg-Hのexact定義 | 丸めlocal行列→理想gateはLの対象。それでもliteral分解→postcompiledのK保証は残る |

既知4caseの1,080 file名の調査結果を維持し、完全なpostcompiled QPY/QASM/DAG・pass traceを確認できていない。全filesystemに存在しないとは主張しない。旧compiled価格とevent/trajectory join、514軌道/1,028 wrapperの固定state probability診断は経験的証拠として保持する。SOURCE/seed/設定一致や局所word equalityから、旧program完全一致・全compiler等価性を推論しない。

Kの分類：

1. **SOURCEと既存lemmaで確認できるもの**：exact Pauli/H/inverseの恒等式、phase込みcontrolの定義、理想wrapperのRe/Im対応。runtime pass全体の証明というscopeではない。
2. **保存資料の接続が必要なもの**：対象gate/pair、equivalenceとDAG instance、phase/basisの具体的な適用対応。旧program/traceが回収できれば対象を限定できるが、それだけで数値保証は完成しない。
3. **新しい数値保証が必要なもの**：Euler/synthesis・角度/phase・丸めlocal gateと理想gateの差。LはこのうちDF局所basisだけを扱う。
4. **旧program・trace欠落で未証明のもの**：実際のpass列、postcompiledの完全な演算子、全PR event familyへの適用保証。
5. **旧資源評価の保証見込み**：保存empirical費用・来歴は保持できる。厳密compiler保証は現時点でMISSING_EVIDENCE。今回のKだけで正式受理に進めない。
6. **最小代替**：まず特定された旧program/trace回収先や限定SOURCE証明を検討する。将来新program系列を選ぶ場合は別承認・有限workloadとQPY/hash/pass receipt保存を必要とし、旧program回収や旧資源評価の代替が成立したとは扱わない。今回は切替・compileをしない。

## L：production kernelと保証scope

[数値kernel](../src/trotterlib/h4_local_enclosure.py)、[authority/controller](../src/trotterlib/h4_local_control.py)、[runner](../scripts/run_h4_local_enclosure.py)を実装した。実H4のdescriptor/SHA/registry検査では、数値entrypointをmockで遮断した状態で148 unique hash、11 ordered basis、324 supportが今回実装の入力仕様に合うことを確認した。[静的入力検査](../artifacts/non_df_h4_kl_production/2026-10-23/static_input_check.json)。行列・trig・normの実H4数値評価は行っていない。

### 局所差と外向きnorm

RZはdiag(exp(−iθ/2),exp(iθ/2))。XXPlusYYはQiskit1.3のlittle-endian定義に従い、|01〉↔|10〉部分のoff-diagonalを−i sin(θ/2) exp(∓iβ)とする。Inverseはθを反転しβを保持する。Descriptor文字列は元のfinite binary64へround-tripし、そのexact整数比をArbへ取り込む。decimalの見やすい値に置換しない。

128 bit固定のArb/Acbでideal local gateを包含し、保存forward/inverseとのFrobenius上界を求める。この上界はspectral normの上界でもある。理想解析gateのunitarityと、丸め保存行列のunitarityを分ける。Unsupported gate、非有限、hash/arity/support不一致、証明不能bound、precision変更は拒否する。

### 基底・control・event family

各基底のforwardは記録順の左積、inverseは逆順のinverse操作として記録する。理想因子norm1、局所差dのtelescoping合成を用い、各段で128 bit外向き上端へ丸める。基底全体の差をf/iとすると、F D Jと理想U D U†の差はf+i+fiで上から覆う。D=Iのcontrol0、signed Z/ZZ/Iやunitary diagonalのcontrol1の両方を扱う。

PRのorder0/2/4は積＋rotationで最大5つのconjugation、r4/q8を固定し、deterministic S2因子も含むuniform operator上界へ伝播する。全324 supportのbasis・sign・qubit coverageを確認する。DF fragment λA²そのものをPauli involutionへ置換しない。

このL boundは**同じexact real angle/signを持つ理想局所gateとencoded matrixの差**である。System global phaseとrelative ancilla phaseを落とさず、符号付きidentityのcontrol=0/1を人工oracleで確認した。ただし、実際のdiagonal角度生成・phase蓄積、controlled gate分解、全compiler、state preparation、B_saved乗算・shot会計を証明したものではない。B_savedスケールは新しいL operator boundへ適用していない。同じ上界は両軸・5精度へ再利用できるが、今回はその40軸へ数値的に結合しない。

### Coverage ledger：単純加算を拒否

既存C totalはbias_distanceとcircuit_semantic_allowanceを含む。C semanticはbasis reuse、phase accumulation、pi丸めを包含する。Lのuniform pair/basis boundもbasis reuse効果を覆うため、少なくともその項が重なる。[coverage ledger](../artifacts/non_df_h4_kl_production/2026-10-23/coverage_ledger.json)に既存Cが覆うprimitive、新L、重複、未被覆を追跡した。

**現在の合成状態はCOMBINATION_BLOCKED。** 既存C aggregateを分解した個別上界が今回は新たに得られていないため、勝手にCの一部を差し引いたり、C total＋L uniformを単純加算しない。局所包含を保存できても、worker resultとSTOPは組合せ未解決を明示し、正式certificateはfalseとする。K未証明を理由にL実装を中止せず、Lの保証scopeを限定した。

## 実行制御・次回契約

[封印済みSOURCE/runtime](../artifacts/non_df_h4_kl_production/2026-10-23/SOURCE.json)、[seal照合](../artifacts/non_df_h4_kl_production/2026-10-23/seal_verification.json)、[完成した次回契約案](../artifacts/non_df_h4_kl_production/2026-10-23/L_execution_contract_DRAFT.json)。新code5＋流用code/test8＝13 file、既存の読み取り専用private backend runtime1,050 file、入力5 fileをSHA封印した。Python-flint環境や共有packageは変更していない。

| 次回の論理call | 最大予算 |
|---|---:|
| Forward局所包含 | 148 |
| Inverse局所包含 | 148 |
| 基底合成（scalar uniform伝播を含む） | 11 |
| Symbolic component coverage | 324 |

SCF/FCI/DF/state/reference/law mean/bias/sampling/RNG replay/compile/N_Gは0。新namespaceを予定するが、まだ作成していない。Contractのexecution_authorized/grant_valid/dispatch_enabled、formal_certificate_accepted、N_G_authorizedはすべてfalse。Approval/grantはnullで、有効な科学grantは発行・消費していない。

Runnerには以下を実装した。

- 起動前と終了後のSOURCE/input/runtime SHA、契約・approval・grantの束縛確認。各予約前にもauthority/STOPを確認する。
- 明示的な新規数値実行承認のscope/SOURCE/input/namespaceを確認し、別操作で新grant発行。Scientificとartificialのdomainを区別する。
- 排他的CONSUMED/LAUNCH/WORKER claim、使用済みgrant・重複namespace・STOP後の再実行を拒否する。
- C由来のflock付きledgerでcall前の非返却予約、個々のlocal/basis/supportのdurable保存、異常時partial/worker STOP保存。
- Detached owned subreaper witnessでnative wait/reapを永続保存し、launcherもwitness終了を回収。元C witnessのscopeラベルは新L receiptを作成する前だけ適合し、旧receiptを改変しない。
- 128 bit固定、retry/自動昇格なし。異常・authority変化・未対応gate・証明不能ではSTOPし、元run/grantを再利用しない。

現在許可されているCLI確認は`--check-contract`であり、実行やgrant発行ではない。次回数値実行には**ユーザーの明示承認と、このSOURCE/runtime/input/固定予算へ束縛された新approval・新grant**が必要。実行後もcoverage/K/wrapper/scale/予算の残存gate、正式受理A、N/Gは別段階である。

## 人工・回帰・異常系の結果

**T67＋binding33＋新L48＝148件PASS、FAIL/ERROR/skip0**。[最終テスト記録](../artifacts/non_df_h4_kl_production/2026-10-23/tests03.json)。局所一致/不一致、inverseのβ、非可換積・順序、phase、control branch、rational interval oracle、外向き合成、非有限・未対応拒否、coverage重複、消費済みgrant・authority前後、partial保存、STOP、harmless native exit0/3を検査した。

初回tests01は、長いexact rational積の分母が増大して整数の文字列変換上限に達する1 ERRORがあった。各段の固定128 bit外向き丸めに改め、exact rational oracleを包含することを検査した。Python全体の上限を変更せず、元の失敗記録を保持する。tests02は145 PASS、worker/domainの追加検証後tests03は148 PASS。

[人工2 gateの保存例](../artifacts/non_df_h4_kl_production/2026-10-23/artificial01/result.json)とartificial_native_0/3のreceiptも保存した。これらは明示した人工・mock入力であり、実H4の結果や新科学grantではない。実H4を人工domainで実行しないよう、workerはartificial markerと最大2 system qubitを要求する。

## 資源・保全・公開・STOP

小さいlocal matrixを148 hashへ重複除去し、256次元の演算を避け、scalar合成を使用する。今回はfirst-failure予約とfsyncを持つ小規模処理であり、process間転送や精度context共有を追加しない構成とした。CPU/RAM/worker capを設定していない。実H4のruntimeや性能優位は未測定。実測空きRAM約486GiB、filesystem約378GiB、観測時CPU/memory PSI avg10=0。共有環境や他ユーザーを変更していない。

旧2,430 fileをSHAで保全する。元H4 native未証明/原因UNKNOWN、既存C native PASSを分離し、今回の人工native receiptを過去の科学終了証拠へ代用しない。旧N/G null、40軸TARGET_MET/正式受理0を変更せず、科学値の再解析・新しい実H4診断・compile・certificate受理・N/Gは0。

本成果をlocal commitし、有効な共通AGENTSのcommit/push許可に従い新branchのpushを試みる。[公開記録](../artifacts/non_df_h4_kl_production/2026-10-23/publication_status.json)に成否を記録し、成功時はremote SHAを照合する。認証失敗時には最終回答へ再実行コマンドを載せる。

**mandatory STOP。** 次の担当は**ユーザー**：固定SOURCE・契約案を対象とするL数値診断の実行承認を判断する。**Codex**：その承認後に新approval/grant・fresh確認を行い、148/148/11/324だけを実行・保存してSTOPする。Kの新compile、新compiled系列、正式受理A、N/Gへは自動移行しない。
