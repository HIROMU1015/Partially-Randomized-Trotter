# Track B G1 decision packet：一回結果・GPT G1引継ぎ

2026-10-09 JST。最終分類 **`G1_BACKEND_CLOSURE_PASS`**。
Phase Aは **`G1_STRUCTURE_PASS_WITH_DECLARED_LIMITS`**。
**run=1、retry=0、mandatory STOP。研究方針・backend採用・次stageの判断はGPT G1へ戻す。**

固定P₃の理想係数構造と、固定8人工LPのbackend証明取得を扱うlocal source-bound evidenceである。
登録B2/B3の資源比較、科学的優位性、新規性、production readiness、immutable CI、外部再現を意味しない。

## 実行・provenance

branch：`track-b-g1-source-preparation-20261009`。
実行HEAD/source S：`718cf6c1abb50c0398028ab39c24c994a4de2bd3`。
[固定source review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/718cf6c1abb50c0398028ab39c24c994a4de2bd3/docs/tracks/algorithm_codesign/g1_source_review_20261009.md)。

利用者の[G1明示指示原文](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/inputs/user_g1_execution_instruction_20261009.md)と
続行指示「ではこの指示に従って進めて」に従い、Sのまま一回呼び出した。
別authorization-only commit、新RA-D0 authorization、source修正は行っていない。
共有API・Track A・旧失敗fixture・既存結果・旧consumed marker/STOPも変更していない。

preflightではHEAD/remote=S、clean worktree、全source/input/protected/runtime hashes、marker/STOP/result未存在を確認した。
[preflight receipt](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/preflight_receipt_v1.json)に496 pathのhashを保存した。
実行後も全496 pathはbyte不変。固定された研究概要・索引も保持し、今回の結果はこの新文書・新dated note・新manifestに記録する。
一般的な要約・索引更新規則より、今回の固定protected files不変指示を優先した。

実行command：

```bash
/usr/bin/python3 -B scripts/tracks/algorithm_codesign/run_g1_decision_packet.py --source-commit 718cf6c1abb50c0398028ab39c24c994a4de2bd3 --execute-one-shot --instruction-file /tmp/track-b-g1-instruction-718cf6c1-20261009.txt
```

契約SHA256：`ab14e8944541691827980dadfe7a4529df44b979995785cd9d499dcc16162fe2`。
旧SoPlex binary SHA256：`59196dd28bba8b25cc960257f1b255f4f50bc4819aa48a0712e557e125c8b1ae`。
SoPlex 7.0.0、旧guard/verifier、設定・input・期待status・順序・capsは固定のまま。
build/compile/installは0。runtime最終同一性確認はPASS。

原文bytes SHA256：`7a22519809b8b5cc4faa927608c4009747612e7df449a635bb4485c2eb62b076`。
runnerが`read_text()`でCRLFをLFへ正規化した指示text SHA256：`2e7c509325d88de69647c401430517173cc94b90730c02697e0319016518dfe6`。
原文bytesはそのまま保存し、markerのtext/hashが正規化後の指示と一致することも確認した。

[one-shot marker](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/one_shot_consumed.json)
SHA256：`83688d6598183d3849ada03131db71079b23c00f972d4957915fd0e0deaeda9b`。
private markerはconsumedのまま。[STOP](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/STOP.json)も保持する。
raw marker/STOP/resultを改変せず、privateから250個の選択した新raw/ledger/spec/input/permit recordsをbyte一致で保存した。
privateの重複resultはコピーを増やさず、既存artifact resultとのbyte一致を確認した。

## Phase A：固定理想classの構造

sourceの7 degree vectorsをtextとして読み、四つの従属係数を独立に消去した。
従属4×4 minorは `x^8 / (18(x^2+2)) > 0`（x>0）、rank=4。
free parametersはs,r,bであり、導出結果は次と一致した。順序はO0,O2,P2,P3,A0,A1,A2。

\[
\gamma=(s,b,\mu+(1-\mu)s-\mu r-b,1-r-b,1-s,1-s,r),
\qquad \mu=\frac{x^2+2}{x^2+6}.
\]

非負領域は `0<=s<=1, r>=0, b>=0, r+b<=1, b+mu*r<=mu+(1-mu)*s`。
有界性とstrict interiorにより3次元を確認した。実際のmuは(1/3,1)、頂点完全性はより強い抽象域(0,1)で監査した。
一般的なpower/Bernstein sign certificatesを保存し、有限個の代入を一般証明の代用にはしていない。

全20 tripleの分類は、singular/non-unique 5件、infeasible intersection 3件、feasible vertex intersection 12件。
feasible intersectionsを重複除去すると6頂点になる。独立列挙後に提案リストと照合した。

| 頂点 | (s,r,b) |
|---|---|
| ordinary | (1,0,1) |
| PTSC-K0 | (1,0,0) |
| A | (0,1,0) |
| J1 | (0,0,0) |
| J2 | (0,0,mu) |
| J3 | (1,1,0) |

各頂点でdegreeごとのmean保存を確認した。A01–A10は保存reportで
`RESOLVED_FOR_STATED_IDEAL_CLASS`。反例・technical failureは報告されていない。

原B2のembeddingは `r=1-s, 0<=b<=s`。
`theta_O=b, theta_A=r, theta_P=s-b`によるordinary/PTSC-K0/A混合の断面であり、J1–J3をB2へ追加していない。
一般gammaを6頂点のconvex mixtureへ分解する3領域の重み式を保存した。
positive groupでは `pi_gp=gamma_gp/gamma_g` から任意の非負precision shareを復元し、
zero groupでは正の混合重みを持つ全寄与が0なので除算しない。

canonical component probabilityは `theta_l B_l/B`、`B=sum_l theta_l B_l`。
thetaだけで抽出する別estimatorとのsecond moment差も式として記録した。
これはsamplerの構成・抽出やcircuit実行を行ったという意味ではない。

理想matchingを元の数値K3全体、midpoint norm、別々のLRM decode像、量子化後のlawへ拡張しない。
固定precisionの6頂点だけでcapped optimizationを解けるという主張もない。
適用範囲は固定7 prototypes/P₃/x>0/固定dictionaryの理想係数classに限定する。

全sign certificates・minor・prototype・parameterization・20 triple・convex weights・precision/samplerの証明は
[Phase A raw report](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/raw/output/A_STRUCTURE_AUDIT.stdout.json)、
[20 triple表示CSV](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/boundary_triples_display_v1.csv)を参照する。
faces番号0–5は順にs=0、s=1、r=0、b=0、r+b=1、b+mu*r=mu+(1-mu)s。

## Phase B：人工backend coverage

Phase A PASS後、8 echo-onlyとその独立verifyをすべて完了してからLPを開始した。
入力係数・offset・matrix/rhs・boundsのrational readbackは全8件PASS。
固定順序、各key一回、solve後の独立Fraction verifyも全8件PASS。

| Fixture | Expected / obtained | 保存された証明 | 結果 |
|---|---|---|---|
| B2_inner | OPTIMAL | primal=dual=52/165、exact gap=0 | PASS |
| B2_outer | OPTIMAL | primal=dual=52/165、exact gap=0 | PASS |
| B2_infeasible | INFEASIBLE | strict Farkas separation=1/2 | PASS |
| B3_inner | OPTIMAL | primal=dual=39/154、exact gap=0 | PASS |
| B3_outer | OPTIMAL | primal=dual=39/154、exact gap=0 | PASS |
| B3_infeasible | INFEASIBLE | strict Farkas separation=1/2 | PASS |
| HP100_B2_inner | OPTIMAL | primal=dual=52/165、exact gap=0 | PASS |
| HP100_B3_infeasible | INFEASIBLE | strict Farkas separation=1/2 | PASS |

OPTIMAL 5、INFEASIBLE 3、certified primal 5／dual 5／Farkas 3、unverified 0、NOT_RUN 0。
statusだけで認証せず、finite-box correction、multiplier sign、reduced cost、exact gap/separationを旧verifierで確認した。
これらのobjectiveは人工証明照合用であり、登録実装の量子資源ではない。B3>B2の科学結果へ読み替えない。

HP100は正の可逆対角変数変換による登録済み人工入力で最大109桁の分母を含む。
この2入力のrational係数・boundsと証明取得は通過したが、変換はsolver側で簡約され得る。
任意の100桁問題、near-singular条件、10^-99 gap、productionの証明取得保証を意味しない。
旧`100_digit_infeasible`は再実行しておらず、旧ERROR/取得未完・旧分類/STOPは保持した。

## 保存値監査と資源

[原result](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/result.json)、
[保存field/provenance監査](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/saved_result_provenance_audit_v1.json)、
[8 fixture summary](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/fixture_summary_v1.json)、
[stage別resource summary](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/resource_summary_v1.json)。
監査は既存raw JSONとの一致、保存PASS/status/counter、hash、process identity、resource記録の照合だけであり、
LP/echo/verifier/構造監査を再呼出ししていない。

| 記録 | 値 |
|---|---:|
| marker基準wall snapshot | 2.761438 s |
| child CPU（wait4/subreaper合計） | 0.474606 s |
| peak observed RSS sum | 29,949,952 bytes（28.5625 MiB） |
| peak sampled execution output | 2,631,983 bytes |
| guard stages | 33 |
| structure / echo / solve / independent verify | 1 / 8 / 8 / 16 |
| cap / numeric / certificate failures | 0 |
| residual processes / retry | 0 / 0 |
| build / compile / registered science / synthesis / circuit / quantum matrix / trajectory / DF / NPZ / GPU | 0 |

wallはcontrollerの保存snapshotで、preflight・公開作業を含まない。
CPUはtarget子孫のkernel wait4/subreaper scopeであり、guard supervisor・outer controllerは含まず、新CPU hard capもない。
RSSは各sampleでunique PIDの直接和を確認した。共有pageの保守的重複、sample間peak、outer controller除外の限界を維持する。
outputはnew private/result/source/reviewのseed bytesも含むsampled inode-deduplicated値。旧read-only runtime/入力/証拠は除外する。
STOP後の証拠整理bytesはruntime測定へ混ぜない。
raw guard stdoutは設計上samples/wait4_recordsを省略するため、ledgerと共有fieldの一致を確認し、省略fieldもledgerに保存した。
全33 stageでfailure=null/exit=0/residual=0。保存PID/start_ticksとのread-only照合でも同一processの残存は0だった。

## GPT G1へ返す未決事項

今回の限定A auditでは6頂点完全性・理想precision復元が通過した。
理想係数自由度を3変数/6頂点混合で表す証拠はあるが、数値K3・dyadic law・資源cap付き最適化への適用は未確定。
固定8人工LPで旧SoPlex/Fraction系のcoverage closureは通過したが、production/backend採用や登録source設計は別判断とする。

GPT側では、(1)理想構造をv4数値sourceへどこまで採用するか、(2)この範囲のSoPlex採用可否、
(3)旧tiny-gap取得失敗と数値/sampler classの未解決点、(4)最小の次登録queryとmateriality、
(5)強いbaseline比較の必要性・順序を判断する。
CodexはこのrunからRA-RTEの採択・資源改善・独立新規性を判断しない。

[Evidence manifest](../../../artifacts/track_b_g1_decision_packet_result/2026-10-09/evidence_manifest_v1.json)に選択した公開pathとidentityを記録した。
source・contract・protected files・原result・marker・STOPは不変。**mandatory STOP、GPT G1 review待ち。**
production、registered B2/B3、新authorization、追加backend pilot、新baseline、科学計算は実施しない。
