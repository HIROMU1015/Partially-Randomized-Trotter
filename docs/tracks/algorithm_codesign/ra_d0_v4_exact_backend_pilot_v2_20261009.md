# RA-D0 v4 Resource Guard / Exact Backend Pilot v2

2026-10-09 JST。Phase A：**`GUARD_V2_PASS`**。
Phase Bの監査判定：**`V4_EXACT_BACKEND_PARTIAL`**。
SoPlex 7.0.0 harnessを一回だけbuildし、初回のrational I/Oを通過した。
人工LPは23予定中16件を求解し、15件の証明を独立Fraction verifierで認証した。
16件目`100_digit_infeasible`はSoPlex `ERROR (-15)`で証明未取得となり、停止した。
残り7件（B2/B3-shaped六件を含む）はNOT_RUN。**retry=0、mandatory STOP。**

## 基点と独立性

- branch：`track-b-ra-d0-v4-exact-backend-pilot-v2-20261009`
- 基点：`9e6d37fe6e345b402ba507db75f77dbec0855199`
- worktree：`.worktrees/track-b-ra-d0-v4-exact-backend-pilot-v2-20261009`
- private領域：`/tmp/ra-d0-v4-exact-backend-pilot-v2-20261009`
- 実行根拠：[利用者指示snapshot](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/inputs/user_pilot_v2_instruction_20261009.md)

旧135 files、数学監査18 files、前回backend pilot28 files、計181 protected pathsは不変。
旧v3/T0/T0.1/T0.2/R1/R1.5/Track A、数学監査、前回backend pilotのsource・結果・分類・STOPを保持した。
前回のguardもprivate実行記録も上書きしていない。v2専用のcode、contract、execution ledgerを作成した。
新authorization、RA-D0 one-shot、registered optimizationは0。
既存のprotected索引を変更せず、[GPT handoff](ra_d0_v4_exact_backend_pilot_v2_gpt_handoff_20261009.md)と本書から新資料を案内する。

## Phase A：監視の変更と保証範囲

専用cgroupを権限昇格なしで作成できる状態ではなかったため、Linux `/proc`のPID集合による監視を採用した。
監視対象は、専用process groupとそのdescendants／adopted children、およびsupervisor自身。
外側controllerや無関係なprocessは対象に入れない。
各PIDを一度だけ集計する**`UNIQUE_PID_RSS_SUM_CONSERVATIVE`**であり、物理的unique memoryではない。
同じ共有pageは異なるPIDに現れ得る。25 ms samplingは観測間の瞬間的peakを保証しない。

CPUは`PR_SET_CHILD_SUBREAPER`と`os.wait4`で取得する。
直接childがreapしたdescendantのCPUはそのchildのkernel rusageに含まれ、
orphanになったgrandchildはsupervisorへreparentされた後に別途reapする。
`Popen.poll()`で先にreapしてrusageを失う方式は使っていない。
normal exit・guard killの両方を人工検査した。ただしcontroller/supervisor自身のCPUはこのchild会計に含めず、
全pilot CPUがexactに取得されたとは主張しない。
[Linux child-subreaper manual](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html)

出力はprivate領域全体、v2 artifact、v2 script directory、新docs/noteを実行前にscopeへ登録した。
TMPDIRを新private領域に固定し、compiler intermediate、binary、stdout/stderr、certificate、JSONを監視する。
重なったrootとhardlinkは`(st_dev,st_ino)`で一度だけ数える。
前回private領域のread-only source/dependency/libraryと181 protected filesは入力として除外した。
64 MiBは変更せず、aggregate samplingと各childのRLIMIT_FSIZEを併用した。
sample間の多file同時生成はovershootし得るため、瞬時のdisk quotaと同等とは呼ばない。

各工程のwallは、個別capとpilot全体残り時間の小さい方。
total wall=3600 s、LP=30 s、harness compile=1200 s、RSS=1536 MiB、output=64 MiBを保持した。
停止時はprocess groupと追跡descendantをSIGKILLし、subreaperによるreapと残存確認を行う。
STOP ledgerまたは同じ開始ledgerがあれば新launchを拒否する。

## Phase A：独立テスト

guard implementation、人工process、test specificationのSHA256を実行前に保存し、一回だけ実行した。
**211 checks PASS / 0 FAIL**。21の名前付きacceptance checksと、各sampleのPID/RSS整合、終了PIDの確認を含む。
これは211の異なる研究条件ではない。

| 利用者の要求 | 観測 |
|---|---|
| single child、正常終了 | PASS |
| child＋grandchild、複数child | PASS |
| parent終了後のorphan grandchild | adoption/reapとCPU回収PASS |
| wall timeout | 小capで発火、対象残存なし |
| RSS発火 | 小capで発火、対象残存なし |
| aggregate output発火 | 複数fileの合算で発火 |
| SIGKILL後の残存確認 | 全対象reap、残存0 |
| normal／killed grandchild CPU | kernel rusageの正寄与を確認 |
| process treeの二重計上防止 | 全sampleのPID一意性と直接和が一致 |
| 対象外process | 計上もkillもしない |
| total pilot wallの継承 | 残り時間を適用、期限切れではforkしない |
| retry拒否 | relaunchはfork前に拒否、旧ledger不変 |
| output root／hardlink重複 | 同じinodeを一度だけ計上 |

人工cap発火は事前に定めた受入テストの期待動作であり、production capを変更したものではない。
Phase A wallは3.390 s。Phase AはPASSのためPhase Bへ進んだ。
[guard contract / freeze](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/guard_v2_contract_v1.json)
[全checks](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/guard_v2_test_results_v1.json)
[v2 guard](../../../scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/guard.py)

## Phase B：結果前freezeとbackend identity

前回定義の人工23 fixtureをcoefficient変更なしで採用し、静的寸法・canonical Fraction・finite positive boundsを検査した。
静的fixture errorは見つからなかった。元定義が前回STOP後に保存されたことを保持し、今回は実行前に凍結した。
順序はB1三件、B2十四件、B3六件。最初に`rational_roundtrip`をecho-onlyで読み戻し、`optimize()`を呼ばず認証した。
その後、各fixtureは一回だけ求解した。反復benchmarkはない。

guard、harness、independent verifier、controller、fixture manifest、backend source/library、
compile command、caps、分類規約をPhase B実行前にSHA256で固定した。
全て終了後も一致した。
[execution contract](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/execution_contract_v1.json)
[固定23 fixtureと順序](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/synthetic_fixture_manifest_v1.json)

SoPlex 7.0.0、source `6657fb3b27044bad7bf2bb58b16de2461de82109`、GMP 6.2.1、Boost 1.74.0を使用した。
前回private領域のarchive、static library、source/headerをhash照合してread-onlyで再利用した。
新install/system/Python/runtime変更は0。private binaryだけを新buildした。
SoPlex static library SHA256は`ddba4a66c04725977d28fcc70526add55d657774a6f683caf4c52c79b14fb3d4`。
harness SHA256は`59196dd28bba8b25cc960257f1b255f4f50bc4819aa48a0712e557e125c8b1ae`。
[backend identity](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/backend_identity_v1.json)
[binary・shared libraries](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/post_build_runtime_identity_v1.json)

SoPlex Apache-2.0、GMP LGPL-3.0-or-later OR GPL-2.0-or-later、使用するBoost core/multiprecision BSL-1.0の
前回license inventoryを固定参照した。今回のlibstdc++/libgcc/libcのnotice hashも記録した。
外部source、package、binaryはGitへ含めない。

## Rational I/Oと証明規約

canonical numerator/denominator文字列をGMP `mpq_set_str`へ直接渡し、SoPlex rational APIに保存する。
solver内部からcoefficient、row rhs/lhs、lower/upper、objectiveを読み戻した。
`1/3, 2/7, 2^-60, 10^-18`、約100桁のnumerator/denominatorは初回echo-onlyでexactに一致した。
16 LPについても全入力のrational echoが一致した。floatからFractionを再構成していない。
objective offsetはdoubleのOBJ_OFFSETではなく外部exact Rationalとして保持し、取得したprimalの目的値へ加える。

7.0.0 sourceに沿ってREADMODE/SOLVEMODE/CHECKMODE=RATIONAL、SYNCMODE=AUTO、FEASTOL/OPTTOL=0、MINIMIZEを固定した。
harnessのv2-only変更はecho-only、等式lhsのecho、SIMPLIFIER_OFF。
sourceのproof取得経路を確認してsimplifierをOFFにし、結果を見て変更していない。

SoPlexのrow dual/Farkasは、upper-only inequalityの係数について符号を反転する。
**両証明で`nu=-raw_A, u=-raw_H`を実行前に固定した。**
Farkasのorientation探索は行わず、rational reconstructionも0。
根拠は固定source `src/soplex/solverational.hpp::_computeInfeasBox`の
`max(y^T A x) < selected y^T b`という符号規約を反転したもの。

独立Fraction verifierはSoPlexをimport/linkせず、saved neutral JSONだけから以下を計算した。

\[
Ax\le b,\quad Hx=f,\quad 0\le x\le U,
\]
\[
r=c+A^T\nu+H^Tu,\quad
L=c_0-\nu^Tb-u^Tf+\sum_j\min(0,r_j)U_j,
\]
\[
r=A^T\nu+H^Tu,\quad\nu\ge0,\quad
\nu^Tb+u^Tf<\sum_j\min(0,r_j)U_j.
\]

primalは全row、bounds、objectiveを認証し、dualは符号・finite box correction・weak dualityを認証した。
nominal objectiveをlowerとして使っていない。九optimal fixtureのexact duality gapは全て0だった。
`upper_active`ではbox correctionが`-2/7`となり、finite-bound項の必要性を確認した。
Farkasは矛盾する等式・不等式、finite upper、decimal/dyadic微小gapで六件取得し、strict separationを認証した。
`upper_induced_infeasible`ではbox minimum=`-7/8`を含む証明だった。
[primal](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/primal_audit_v1.json)
[dual](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/dual_audit_v1.json)
[Farkas](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/farkas_audit_v1.json)

## 停止結果とraw classificationの不一致

| 範囲 | 結果 |
|---|---|
| B1三件 | 全証明PASS |
| B2十四件 | 十二件PASS、十三件目でERROR、残り一件NOT_RUN |
| B3-shaped六件 | 全件NOT_RUN |
| feasible/optimal | 九件primal＋dual認証 |
| infeasible | 六件Farkas認証 |
| `100_digit_infeasible` | ERROR (-15)、証明未取得 |

失敗fixtureは`lower=upper+10^-99`の人工LPである。exact inputのechoは一致したが、
SoPlexはOPTIMAL/INFEASIBLEではなく`SPxSolver::ERROR (-15)`を返した。
stderrは空で、primal/dual/Farkasは返されなかった。取得失敗の内部原因はこのログだけでは特定できない。
処理は約0.000432 s、guard failureやresource capではない。
新しいlogging設定、別solve、追加certificate再構成で原因を追試していない。

controllerはverifierがERROR statusを拒否した場合もgenericに`V4_EXACT_BACKEND_CERTIFICATE_FAIL`へ分類していた。
これは分類実装の不具合であり、実際に不正な数学的証明を取得したという意味ではない。
raw terminal、source、ledgerを変更せず保存した。
利用者指示§14 D（Solver failure）と§15 PARTIALに沿った**監査判定は`V4_EXACT_BACKEND_PARTIAL`**とし、
raw labelと併記する。結果を見てcertificate規約や実行を変えたものではない。
[raw execution output](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/raw_phase_b_execution_output_v1.json)
[failure semanticsと分類根拠](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/failure_semantics_v1.json)

停止後はmixed-scales一件とB2/B3-shaped六件へ進んでいない。
15件のprefix認証を全domain PASSや、元B2/B3 classのfeasibility、RA-RTE優位・非優位へ読み替えない。
inner用Farkasはそのinnerのみの証明で、元classのinfeasibilityにはしない。
outerから元classへの結論には包含関係の証明を要する。今回はRA-shaped fixture自体が未実行である。

## Resource測定

| 項目 | 観測 |
|---|---:|
| harness compile回数 | 1 |
| harness build wall / child-tree CPU | 19.312 / 19.269 s |
| Phase Bの最大RSS合算 | 1,104,265,216 bytes = 1,053.109 MiB |
| compilation中の最大output合算 | 30,616,216 bytes |
| 16 LP subprocess wall合計 | 0.5048 s |
| 16 LP subprocess CPU合計 | 0.03248 s |
| internal solve range（ERROR case含む） | 0.00005171–0.00043238 s |
| Fraction verification range（ERROR拒否含む） | 0.00008472–0.00042847 s |
| Phase B child scopesのCPU合計 | 19.5595 s、supervisor/controller自身を除く |
| maximum executed denominator bit-length | 658 bits |
| certified最大variables | 5 |
| B2-shaped 28 / B3-shaped 22 variables | 未実行 |
| Phase B cap到達 | 0 |
| 残存対象process | 0 |
| retries | 0 |

input bit-length、fixture寸法、primal/dual/Farkas取得時間、verifier時間、guard測定、全saved input/outputを公開する。
同fixtureの再求解benchmarkは行わなかった。少数の小型LPから性能分布を推定しない。
samplingとRSS共有pageの限界、CPU測定scope、outputのsample間overshootを明記する。
[resource benchmark](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/resource_benchmark_v1.json)
[全実行／NOT_RUN](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/synthetic_results_v1.json)

## Backend採用への技術提案

**SoPlex 7.0.0をproduction backendとして採用する判断は保留。**
有理input、九primal/dual、六Farkasは実証できた。
一方、必須fixtureで証明取得に失敗し、RA-D0型28/22変数のbackend可否は未実証である。
今回をtechnical部分成功として残せるが、v4 productionへ進む根拠は揃っていない。

旧v3の55,275 main／110,550 total callsは参考に留め、v4件数や総実行時間へ自動適用しない。
continuous innerとB2 outerの難度差、保守的innerのfeasibility、proof取得費用、
outputとFraction検証のscalingは今回の小型prefixからは判断できない。
anchor-first、query recipe、baseline、denominator/tolerance、production requirementsは変更していない。

極小gap fixtureを必須のまま扱うか、利用可能なbackend scopeをどう定義するか、
追加検証に情報価値があるかはGPT側で判断する。Codexがfixtureを除外して続行することはない。
controllerのlabel defectもreview対象として保持するが、同pilot内で修正・再実行しない。
**資料公開後mandatory STOP。production implementation、新authorization、registered solveへ自動移行しない。**

## 証拠・provenance

[evidence manifest](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/evidence_manifest_v1.json)に
全新規pathのSHA256、181保護対象の照合、frozen source/contract/backend identity不変を保存した。
registered LP=0、production source changes=0、science/synthesis=0、
circuit/quantum matrix/trajectory/DF/molecule/NPZ/GPU/IS/CTS、新angle/precision、新authorization、旧marker変更は全て0。
過去の研究分類を再分類しない。
