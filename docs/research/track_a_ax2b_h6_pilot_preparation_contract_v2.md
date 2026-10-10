# Track A AX-2B H6技術pilot：準備契約 v2

2026-10-10 JST。`DRAFT_NOT_AUTHORIZATION` / `H6_NOT_AUTHORIZED`。
[GPT独立レビュー](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md) §§13–17,21–23を受けた
[追補 v1](track_a_ax2b_post_independent_review_amendment_v1.md)。
旧[v1](track_a_ax2b_h6_pilot_contract_draft_v1.md)と旧machine-readable案は変更しない。
研究上の進行方針を採用し、source/synthetic準備を進める。入力生成・科学実行・資源割当は未認可。

## 登録技術scope

linear H6、1.00 Å、STO-3G、T=.8、12 system qubits、Nα=Nβ=3/dimension400。
指定serialized H_DF/normalized stateのfinite-time signalであり、ε_sig=.001は診断label。
state preparation、energy estimation、PR優位、N×Cによる資源順位は今回の技術完成条件にしない。
H6全露出はdevelopment。H8 input/truthへの接触、GPUは対象外。

actual DF rankをL、p=(L+1)//2、L≥2とする。

| cell | order | prefix | q | R/r/K | cost replica |
|---|---|---|---:|---|---:|
| H6_B0_S2_q2 | S2 discard | p | 2 | — | 1 |
| H6_B1_S2_q1 | global S2 | L | 1 | — | 1 |
| H6_B1_S2_q2 | global S2 | L | 2 | — | 1 |
| H6_B1_S4_q1 | global three-piece Yoshida S4 | L | 1 | — | 1 |
| H6_B1_S4_q2 | global three-piece Yoshida S4 | L | 2 | — | 1 |
| H6_B2_K2_q2_R4 | partial S2 finite-RTE | p | 2 | 4/2/2 | 2 |
| H6_B3_K6_q2_R4 | partial S2 finite-RTE | 0 | 2 | 4/2/6 | 2 |

5×4+2×2×4=36 wrapper、4 random trajectories/8 outer occurrences。
ordinary/symmetric_directional×cosine/sineを同じprepared trajectoryで対応比較する。
symmetric_directionalは**新契約**のtechnical cost primary。旧H4のprimaryを遡って変えない。
fresh trajectoryを各量子shotで生成する測定意味論と、paired cost用の共有を混同しない。
seedは旧v1文字列規則と0-based replicaを維持し、衝突はSTOP。結果を見て再選択しない。
one-bodyはprefix0でも決定論側、scalar/extracted identity/event phaseを保持する。
finite平均は数値作用であり、非unitary gateとしてcompileしない。

## 入力政策・reference・state

tol-only adapterは`truncation_threshold=1e-8`のみをdecomposerへ渡し、`final_rank`と分子config rankを供給しない。
mockでこのkwargsを確認した。実際の分子decompositionは未実施。
actual kwargs/L/truncation value、integral hashes、ordering、one-body correction、各Hermitization前後hash/差を保存する。
後段cutoffは0の準備案。許容値1e-10を超えるHermitization変更はSTOPし、無断で救済しない。
新adapterは既存分子builder/configを変更しない。H4 rank12との共通DF政策のmain比較は別に固定する。

future入力生成は予算・authorizationを別に持ち、生成後hash/sourceを固定してSTOPする。
HF初期vector、eigsh/which=SA/tol1e-12/maxiter1000/ncv40を技術案として維持。
solverと事後residualのmatvec/rmatvecを共通before-call counterで制限する。
収束・小残差をground-state証明としない。cross-spin/sector不成立では別sectorへ切り替えずSTOP。

referenceは400次元sectorの単一dense matrixをbounded matvecで構築する案。
共有matvecに依存する経路と独立occupation構成を区別する。
native actionは4096次元full vector、finite平均はsector作用。Gaussian中間をsectorへ投影しない。
全4096×4096 fragment行列・固有vector cacheは作らない。
新しいbounded helperの合成検証は、H6での正しさ・memory・収束を実証していない。

## 数値scopeと上限

総uの厳密certificate完成を技術起動の一律前提にしない。
technical gateだけの実行では`numerical_allowance_certified=false`、`accuracy_eligibility=UNDETERMINED`、N/G=null。
empirical uは後続条件付き解析で根拠・適用範囲・headroomへの感度を示す。u_boundへ昇格しない。

旧v1 caps：CPU/worker/BLAS各1、AS8GiB、phase1800/1800/3600秒、total7200秒、
output512MiB/log64KiB/diag1024、compile36/trajectory4/occurrence8、primitive2000/control256、
solver10000/reference total20000/per-action20000、deterministic actions100000/cell、
tail corrected+raw B2=24/B3=56、untranspiled1e6/transpiled5e6を**提案値**として維持。
入力生成のwall/CPU/output予算は未割当nullで、pilot input/reference phaseとは別。
Python import・起動もtotalに含め、phaseタイマーをcellでリセットしない。retry/resumeなし。

actual-rankでmerged global PF/unmerged directional/partialの時間集合を展開し、negative/half/undo時間を含める。
全sector列×全時間を機械的に課さず、構造的sector保存、独立small oracle、代表数値probeの役割とcoverageを結果前に固定する。
probe capに入らない場合はSTOPして計画を修正する。実行後の間引きは禁止。
native `block_instruction_bound`、tail event上界からtask別instruction上界を固定する作業は未完了。

新scale guard案：非finite値でSTOP、B overflowまたはraw attenuationがnormal以下へ入る場合STOP。
agreementは`absolute_error <= 1e-9 + 1e-10*max(1,input/intermediate norm)`のengineering診断。
reference比較/inner productへの適用方法と全stage norm traceはmolecular portで未接続。
この数値規則はuの証明ではない。scaleに合わない場合、R/rank/tol/seedやgateを結果後に変えて成功扱いにしない。

## 準備状況と次の固定順序

metadata CLI、schedule、tol-only mock、会計、独立small reference、caps/writer、fake-controller、dummy-watchdogを実装・検証した。
**H6 molecular backend・science launcher・authorization parserは未実装**。CLIは実行引数を受理しない。
残る順序は次のとおり。

1. H4-N/A/Eのmolecular検証portと監査を準備し、対象・上限を別実行指示へ載せる。
2. H6 input generationの別scope/budgetを確定し、明示指示後だけ入力生成、hash固定後STOP。
3. H6 molecular port、actual probe/gate plan、source/input/environment/CPU binding、別authorizationを固定。
4. 明示launch指示後の一回technical実行、partial/terminal保存、保存監査、STOP。
5. H4限定検証・H6technicalの結果をGPTへ渡し、本検証GO/STOPを判断する。

通常の保全的なsource/synthetic修正はCodexの範囲。研究意味論の変更・未解決矛盾は早期にGPTへ戻す。
準備完了を科学起動準備完了と同一視せず、H6 main/H8を自動認可しない。
