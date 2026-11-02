# RA-D0 final source review v2 — registered solve未実行

2026-10-07 JST。基点`0ddf67756516e08f85fed1b987459a5e862676b7`から独立した
`track-b-ra-d0-source-review-v2-20261007`で、[利用者指示のbyte-exact原文](inputs/ra_d0_source_review_v2_user_instruction_20261007.md)
に従って実行契約とsourceを修正した。

**`READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW`。** ローカル技術確認にblocking issueは見つからなかった。
GPTによる最終source reviewは未了。authorizationは作成していない。
registered optimization、B2 minima、registered budget実値、B3解、strict witnessは取得していない。
科学実行・新条件・次stageの認可はない。mandatory STOP。

## 固定入力と証拠境界

| 入力 | 固定commit |
|---|---|
| v1 preparation | `0ddf67756516e08f85fed1b987459a5e862676b7` |
| R1 saved result | `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` |
| R1.5 saved attribution | `af3d014d0a0cfcbbd25bb544f6544652fec92942` |
| RA-RTE mathematical audit | `8a04c148a66d23dbc1f045086a95a5e19a6372dc` |

対象は保存済み2-qubit distinct-basis controlled、finite P3、p=(3/4,1/4)、x={1/8,1/4}、
既存precision={1e-3,1e-4,1e-6}。sigma=+1を設計側、-1を保存値の符号controlとする。
分子、geometry、DF rank/split、held-outは対象外。新angle・precision・合成・Hamiltonian取得はない。
R1のscience分類、R1.5のpost-hoc性、全旧STOP、Track A、共有APIを維持した。

原文・[入力identity](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/input_identity_v2.json)を別pathに保存した。
旧artifactを再生成していない。8 indexにはv2への案内だけを先頭に追加し、基点の本文をbyte-exactで残した。

## 数値baselineと包含

[numerical amendment](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/numerical_baseline_amendment_v2.json)
を[numerical kernel](../../../src/trottertracks/algorithm_codesign/ra_d0/numerical.py)へ実装した。

| class | 今回の意味 |
|---|---|
| B0_saved | R1の9 profiles/x。immutable anchor、元shot/resource参照、same-n budget sourceのみ |
| B0_ideal | finite-mean decompositionの数学embedding専用。B0_ideal⊂B1_ideal |
| B1_num | 一representation固定、三precision間の配分、共通dyadic lawとmembership allowance |
| B2_num | whole-representation latent z混合、共通dyadic lawとmembership allowance |
| B3_num | 同じ保存implementation columnsをdegree-localに配分、共通dyadic law |

B0_saved⊂B1_numとは主張しない。保存midpoint/coefficient roundingの会計とR1分類を変えない。
qは2^60分母largest remainder・index tie、yはnearest/half-up、負のnominal qはclipせずreject。
registered pipelineにはdenominator overrideを設けていない。旧synthetic丸めテストの小分母fixtureは維持した。

三precision group gに対しτ_g=(3+c_g,up/2)/2^60を固定した。
B1では|Σ_p q_gp−y c_g|、B2では|Σ_p q_rgp−z_r c_g|を、元coefficient intervalの両端で認証する。
z≥0、Σz=yをexact Fractionで確認する。zは物理samplerではなくwitnessなのでdyadic化しない。
取得手順はnominal zの非負sharesを一度だけ丸め後yへnormalizeする。失敗後のshares調整はしない。

数値包含の根拠は次の通り。

1. B1の固定representationにz=y、他representationにz=0を置く。追加groupはmass=0でτ条件を満たす。
   q/y、mean、confidence、resource、workspaceは同じなのでB1_num⊂B2_num。
2. B2の共有O0 aliasだけを、保存implementation identityが等しいcolumnへ合算する。
   dyadic qの非負性・総和1を保ち、Dq、d·q、全cost·q、workspace peakを変えない。
   B3にはmembership制約がないのでB2_num⊂B3_num。

これは定義と線形性による包含の説明で、全registered samplerを列挙した結果ではない。
人工fixtureでB1/B2/B3のcertificate保存とalias合算を確認した。
旧ideal embedding・sign auditを読み取りで再照合し、旧数値未確定statusは歴史記録として残した。

全classでξ≥||Dq−yt||_1のinterval upper、ξ≤y·10^-12、
(1/200)y−d·q−ξ≥κ_n,upを認証する。ell≥ln(10560)とκは旧100桁outward enclosureを継承した。
workspaceは期待値でなくactive-support peak≤1。
resource upperは2n[cost·q+(resource=1Qなら5/2)]。これは既存native additive会計でwhole-circuit最適値ではない。

[LP compiler](../../../src/trottertracks/algorithm_codesign/ra_d0/lp.py)のv2 innerは両端worst条件、
outerは外側条件を使う。B2_outerはdyadic性を外してB2_num全体を包含する。
有限変数上界を使うexact dual stationarity補正によりlowerを認証する。
outer optimizerをphysical B2 lawと呼ばない。
**primaryはcertified U_B3<L_B2,outerだけ。** nominal float optimumやtolerance差をwitnessにしない。

## Budget freezeと順序

[freeze policy/schema](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/budget_freeze_schema_v2.json)、
[bounded recipe](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/bounded_query_recipe_v2.json)、
[engine](../../../src/trottertracks/algorithm_codesign/ra_d0/engine.py)を固定した。

Phase Aは対象batchの全(x,n)でB2のT/CX/1Q inner solveを各1回行い、
固定q/y丸め→membership→mean→confidence→workspace→resource upperを確認する。
一つでもcertified implementationが得られなければ
`TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION`で全one-shotを停止する。
nominal solve candidateを認証したupperを使い、離散lawの厳密global optimumを得たとは主張しない。
single-resource solveのprimal/dualとcertification gapを保存できる形にした。

同nでconfidence-feasibleなB0_saved最大9件と、三minimum候補の**完全T/CX/1Q vector**最大3件だけを使う。
exact resource tupleでdedupし、同vectorの他二座標をepsilon-constraint capへ渡す。
異なるprofileの座標をCartesian productしない。

各batchの全budget/queryを`budget_freeze.json`へexclusive保存し、SHA256、input IDs、vector/query数を
receiptへ保存してからPhase Bへ進む。Phase AのB3呼出しはsource guardで拒否する。
Phase Bはhash/input/derived-queryを検証し、B2_outer lowerとB3 primal upperだけを取得する。
batch完了後にもfreeze hashを再確認する。結果からbudget/grid/settingsを変更しない。

**anchor-firstとfreezeの整合：batch単位で全点を先にfreezeする。**

1. P1の両x×9 anchors全18点をPhase Aでfreezeしてから、その全paired queriesをPhase Bで比較する。
2. 両xにanchor strict witnessがあればSTRONGで終了しcoverageを実行しない。
3. それ以外はanchor witnessのないxだけをP2対象とする。片方なら片方、両方なら両方。
   P2の必要coverage全点を別のPhase Aでfreezeし、その後だけP2のB3へ進む。

P1/P2はそれぞれimmutableな別freezeを持つ。P2対象選択は固定anchor gateだけで、P1 freezeの編集や
B3を見たbudget選別ではない。全737点のbudgetをanchor評価前に取得する方式は採用していない。
このbatch解釈もGPT最終reviewの対象とする。

solver infeasible flagだけでは分類しない。最大一回の補助LPで取得したFarkas rayをexact検証する。
取得／検証失敗はtechnical STOP。B3-only feasibilityはdescriptiveでprimary witnessではない。
未完了・cap・certificate失敗は、途中witnessの有無にかかわらずtechnical分類を優先する。
coverage-onlyはLOCAL止まり。完了後NO_REGISTERED_WITNESSは登録したpaired queryで認証witnessなしという意味だけ。

## 実行guardと出力

[execution contract](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/execution_contract_v2.json)に固定した。

| 項目 | 上限 |
|---|---:|
| grid | 737点（anchors 18、coverage 719） |
| main LP/点 | 3+12×3×2=75 |
| main LP全点 | 55,275 |
| 各mainに最大一Farkas auxiliaryを含むrecipe上限 | 110,550 |
| total hard call guard | 111,000（recipe guardも別途110,550） |
| processes / solver・BLAS threads / retries | 1 / 1 / 0 |
| per-LP wall / total wall / total CPU | 2 s / 3,600 s / 3,300 s |
| peak RSS / virtual address / output | 1,536 MiB / 4,096 MiB / 128 MiB |

[guard](../../../src/trottertracks/algorithm_codesign/ra_d0/guard.py)は呼出し前のcount拒否、retry拒否、
一auxiliary/main、monotonic wall・process CPU・peak RSS・compressed bytesを確認する。
POSIX RLIMIT_AS/RLIMIT_CPU、SIGXCPU、50 ms周期SIGALRMを併用し、HiGHS time_limit=2 s、threads=1、parallel=Falseを固定する。
cap subtypeは`D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP`で、primary分類はtechnicalとなる。
terminal STOP記録用に64 KiBを予約する。source reviewでGPU query、science matrix、合成等を行っていない。

Python signal handlerは長いC呼出し中には遅れて動く場合がある。
per-LPはHiGHSの内蔵time limitと呼出し前後のwall検査を併用し、超過した解を採用しない。
2 sちょうどでの強制killを保証する別process watchdogではない。process=1を維持する。
根拠は[SciPy 1.16.2 source](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_linprog_highs.py)、
[Python signal仕様](https://docs.python.org/3.12/library/signal.html#execution-of-python-signal-handlers)、
[resource仕様](https://docs.python.org/3.12/library/resource.html)。この実装限界も最終reviewに渡す。

[output schema](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/output_schema_v2.json)はcompressed JSONLと
freeze/receipt/terminal resultを定義する。全paired queryのID、x/n/tag、objective、budget ID、caps、
B2/B3 status、exact bounds、strict bool、certificate hashesを保存する。
完全vectorsはminima・strict witness・certified infeasible・返却済みvectorsのあるtechnical failureで保存する。
通常の非witness行はcompact recordだけで、discardしたvectorをhashから復元できるとは主張しない。
中断したsolverの未返却vectorは捏造せず、last task/reasonを保存する。
usageはterminal receipt直前の測定値で、最終書込自体にもoutput guardを適用する。

## Source/authorization gate

[runner](../../../scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py)は最初に[launch gate](../../../src/trottertracks/algorithm_codesign/ra_d0/launch.py)
を通す。現sourceにはauthorizationがないので、登録table／minima／markerの前に拒否する。
将来の起動には固定source Sの直接子A、authorization JSON＋任意receiptだけのdiff、clean HEAD、
HIROMU1015 remote、contract/source/input/runtime hashesと明示one-shot指示が必要。
別review後のAからexclusive markerを作ってからだけdevelopment solverを許可する。
sourceチェックは既存保護資料をmaterializeした実行worktreeを前提とし、欠けたpathを黙って省略しない。
source SHAの自己参照はmanifestへ埋めず、SのGit commitとAの親関係で固定する。
今回authorization・marker・registered resultは作成していない。

## 技術確認と未保証事項

[80 focused tests](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/focused_verification_v2.json)がPASS
（旧35件をbyte-exactで保持、追加45件）。Python 3.12.3 / SciPy 1.16.2 / NumPy 2.5.3の隔離venvを使用した。
人工LPへの実solver呼出しは最終検証5回。registered solver呼出しは0。
full test suiteは実行していない。これはローカル技術検証で、immutable CI・外部再現・RA-D0科学結果ではない。

各x 21 columns、全18 sign pairs、保存126 synthesis sequencesのidentity/count/pass記録が一致した。
operator errorやsignalを再計算していない。旧R1 result・marker・source・authorization、tool identityと旧v1 artifactの
SHAは[provenance audit](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/provenance_audit_v2.json)参照。
本改訂で取得したregistered budget実値、minimum、witnessは0件。

registeredで証明を取得できることやcap内に完了することは保証しない。
τはdyadic rounding由来の約10^-18幅なので、double LPのnominal membership誤差がcertificationを落とし得る。
規定どおりno rescueでtechnical STOPする。coverageのoptimistic n_minもB2のfeasibilityを保証しないため、
低nのB2 minimumで停止し得る。infeasible点を飛ばす変更は入れていない。
これらは未実行の可能性／設計制約で、registered結果を先取りした主張ではない。

GitHub上のこのsourceをGPTが最終reviewする。研究方針・RQ・新規性・追加検証の範囲はGPT側の判断。
通過後にも別authorization-only childと明示実行指示が必要。全outcome mandatory STOPを維持する。
