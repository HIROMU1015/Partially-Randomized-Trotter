# RA-D0 v4 Exact LP Backend Feasibility Pilot

2026-10-09 JST。最終分類：**`V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE`**。
SoPlex 7.0.0の静的libraryを専用領域でbuildしたが、pilot harnessのcompile中にRSS guardが発火し、停止した。
**synthetic LP calls=0、registered LP calls=0、retry=0。backend採用は保留。**
cap到達後はguard修正、再build、solver実行、別backendへの切替を行っていない。

## 分離と証拠範囲

- branch：`track-b-ra-d0-v4-exact-backend-pilot-20261009`
- 基点：`beb82427d202f479cc2ba954480d73a51941e322`
- worktree：`.worktrees/track-b-ra-d0-v4-exact-backend-pilot-20261009`
- 隔離build/runtime領域：`/tmp/ra-d0-v4-exact-backend-20261009`
- 実行許可：今回の[利用者指示snapshot](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/inputs/user_backend_pilot_instruction_20261009.md)。新authorizationは作成していない。

今回の入力は人工LP定義のみ。既存candidate tableの係数はfixtureへ使っていない。
保存済み研究データの求解、budget freeze、source変更、科学的再分類は行っていない。
旧v3/T0/T0.1/T0.2を含む135 protected pathsと、基点数学監査の18 paths、計153 pathsのSHA256を照合し、不変だった。
旧consumed marker SHA256は`88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`のままである。
既存索引を含むprotected pathsを変更せず、本書と[handoff](ra_d0_v4_exact_backend_gpt_handoff_20261009.md)から新規資料を案内する。

## 実際に導入したもの

SoPlexの公式`release-700` tagを確認し、full commit
`6657fb3b27044bad7bf2bb58b16de2461de82109`のarchiveを取得した。
archive SHA256は`276290a48a177cb1793be19f169f0a1810d7c9903f7e8b05ffd4bb607cb9baf0`。
これは取得物のidentityであり、別署名や公式公開checksumを照合したという意味ではない。
採用候補は公式資料に対応する7.0.0に限定し、masterのversionへ追随しなかった。

公式7.0.0はplain Makefileによるbuildと、rational mode用GMP/Boostを説明している。
CMakeが見つからなかったためMakefileを使った。
[固定sourceのINSTALL.md](https://github.com/scipopt/soplex/blob/6657fb3b27044bad7bf2bb58b16de2461de82109/INSTALL.md)

| 依存物 | 実際の導入方法・identity | license |
|---|---|---|
| SoPlex 7.0.0 | 上記公式archiveをprivate領域へ展開 | Apache-2.0 |
| GMP 6.2.1 | Ubuntu `libgmp-dev_6.2.1+dfsg-3ubuntu1_amd64.deb`をprivate領域へ展開、static archiveを使用予定 | LGPL-3.0-or-later OR GPL-2.0-or-later |
| Boost 1.74.0 | Ubuntu `libboost1.74-dev_1.74.0-14ubuntu3_amd64.deb`をprivate領域へ展開 | 使用するcore/multiprecisionはBSL-1.0、packageの各file noticeも記録 |
| g++ 11.4.0 / GNU make 4.3 | 既存system toolを使用、導入変更なし | 既存compilerと依存libraryのnotice hashを記録 |

Ubuntu二packageのSHA256はlocal apt metadataの値と一致した。
`dpkg-deb -x`によるprivate展開だけで、apt/dpkg installation、sudo、system-wide変更、Python dependency追加は0。
GMP/Boostのversionは展開したheadersでも照合した。
既存compilerが読むshared libraries、compiler/cc1plusのbinary hash、license notice hashを保存した。
実行harnessはlinkまで到達しなかったため、harness binary SHA256とruntime shared-library identityは未取得である。
[導入inventory](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/backend_inventory_v1.json)
[build/runtime identity](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/build_runtime_identity_v1.json)

## Build結果とRSS guardの問題

初回の`make -j1 makelibfile GMP=true BOOST=true ZLIB=false QUADMATH=false ... USRCXXFLAGS=-O1`は完了した。
GMP/Boost includeとGMP static libraryはprivate領域を明示した。
SoPlex static library SHA256は`ddba4a66c04725977d28fcc70526add55d657774a6f683caf4c52c79b14fb3d4`。
続く初回の`g++ -std=c++11 -O1 -DNDEBUG ... harness.cpp ...`を、supervisorがSIGKILLで停止した。
両command全文、stdout/stderr、開始記録、resource recordを保存した。再実行はない。

作成したpilot supervisorに、RSS集計の不具合があった。

```python
peak = max(peak, tree_rss(process.pid) + tree_rss(os.getpid()))
```

`tree_rss(os.getpid())`は既にcompiler側のchild process treeを含む。
したがってこの式はchild treeを二重計上する。
**guard indicator=1,612,357,632 bytes（1,537.664 MiB）は真のunique peak RSSではない。**
guardは1,536 MiBの設定で発火したが、実際のRSSがその上限を超えたとは立証できない。
RSSによるSoPlexの利用不能や、数学的certificate failureとは解釈しない。
cap到達時のretry禁止に従い、監視コードも実行後修正せず、そのまま証拠として保存した。
修正案は「supervisorをrootとしたtreeを一度だけ集計する」であり、今回の採用・実行は0である。

| 測定項目 | 記録 | 限界 |
|---|---:|---|
| static library build wall | 7.071 s | library buildのみ |
| harness compile wall | 6.260 s | guardによる途中停止 |
| library build CPU | 6.967 s | supervisorのRUSAGE_CHILDREN |
| harness compile CPU | 0.001438 s | killed compiler grandchildrenが回収されず、総CPUを示さない |
| library build RSS indicator | 715,661,312 bytes | 同じ二重計上を含む |
| harness compile RSS indicator | 1,612,357,632 bytes | true peak RSSは未確定 |
| LP solve / primal / dual / Farkas取得時間 | 未測定 | LP calls=0 |
| independent Fraction verification時間 | 未測定 | certificate未取得 |

各childには1,536 MiBのRLIMIT_AS、process groupのwall timeout、file-size capを設定した。
RSS監視は25 ms間隔で、threadsの環境変数は1、makeは`-j1`。
実solver processは0であり、solver thread数・per-LP 30 s enforcementは実証していない。
全体3600 sの時計をbuild前に開始し、発火時刻までのwallを保存した。
64 MiB監視はbuild log領域を対象としており、全build産物を連続監視できてはいない。
事後にobj/lib、log、公開evidenceのbytesを計測した。外部source/dependency入力はevidence出力に含めていない。
全域のoutput cap enforcementも今後の未決項目として残す。
[resource benchmarkと限界](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/resource_benchmark_v1.json)

## 有理入力・証明：全てNOT_RUN

| 項目 | 今回の判定 | 理由 |
|---|---|---|
| Fraction→backend→Fraction exact roundtrip | NOT_RUN | harness未build |
| exact primal | NOT_RUN | 解取得0 |
| exact dual lower | NOT_RUN | dual取得0 |
| exact Farkas | NOT_RUN | ray取得0 |
| independent Fraction verification | NOT_RUN | backend certificate0 |
| 小規模LPの実行可能性・時間変動 | NOT_RUN | 求解0 |

APIの存在をPASSへ読み替えない。既知解からcertificateを作ってbackend出力を代用していない。
公式exampleのrational設定・primal/dual APIを参考にしたpilot専用harnessは未compile・未検証である。
[固定sourceのrational example](https://github.com/scipopt/soplex/blob/6657fb3b27044bad7bf2bb58b16de2461de82109/src/example.cpp)

計画上はcanonical numerator/denominator文字列を`mpq_set_str`で読み、SoPlex rational APIへ渡す。
solverのrational LPから全coefficient・bounds・rhsを読み戻してneutral JSONへ保存する。
objective offsetはdoubleのOBJ_OFFSETへ渡さず、exact外部Rationalとして加算する案だった。
`1/3, 2/7, 2^-60, 1/10^18`および100桁程度の値を用意したが、backendへの入力は0。

独立verifierはsolver/libraryをimportせず、saved JSONとstdlib Fractionで、
全row、bounds、objective、以下の式を確認する未実証実装である。

\[
r=c+A^\mathsf T\nu+H^\mathsf Tu,\qquad
L=c_0-\nu^\mathsf Tb-u^\mathsf Tf+\sum_j\min(0,r_j)U_j.
\]

\[
r=A^\mathsf T\nu+H^\mathsf Tu,\qquad
\nu\ge0,\quad \nu^\mathsf Tb+u^\mathsf Tf<\sum_j\min(0,r_j)U_j.
\]

dualの符号mapping、finite bound correction、offset、自由符号の等式multiplier、weak dualityは未照合。
Farkasは取得rayのglobal orientation二通りだけを比較する案で、rational reconstructionは0。
実際に得られるrayとその符号、degenerate/nonunique caseは次の検証が必要である。

## 未実行fixture

compile完了待機中に23件の人工fixture定義を作成した。solverへ提出したfixtureは0件。
定義のfile timestampは実際のguard発火より後で、停止結果を取得する前である。
結果前に固定した実行manifestや、実行済みfixtureとしては扱わない。
STOP後は保存用に定義をJSONへserializeしただけで、新しい求解・certificate評価は行っていない。

基本LP、unique/multiple/degenerate、active upper、等式・不等式の矛盾、
微小gapのfeasible/infeasible対（decimal/dyadic/100-digit）、mixed scales、
人工B2/B3のinner/outer/infeasible構造を含む。
B2は8 groups×3 precisionsにz三変数とyを加え28 variables、B3は7×3にyを加え22 variables。
人工normalization、degree matching、mean/confidence reserves、resource constraints、finite boundsを含む。
係数とbit-lengthを保存しているが、feasibilityや求解能力は実証していない。
[全未実行fixture](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/synthetic_fixture_manifest_v1.json)
[pilot専用code入口](../../../scripts/tracks/algorithm_codesign/exact_backend_pilot/fixtures.py)

## Productionへの技術判断

**今回の証拠だけではbackend採用はGOにできない。**
isolated依存展開とstatic library buildは可能だったが、最重要のrational I/O、primal、dual、Farkas、
independent verification、LP timingは全て未実証である。利用できるLP規模を数値で提示できない。
旧v3のmain 55,275 / total 110,550を新v4の件数へ適用せず、総時間も外挿しない。
continuous inner、B2 outer、証明取得、Fraction検証の費用差は未測定である。

GPTへの条件付き技術提案は、まずguardの重複RSS、grandchild CPU会計、全出力capの範囲をreviewすること。
修正した別pilotを行うか、setupにこれ以上の情報価値があるかはGPT側で判断する。
今回のcap・retry=0を上書きせず、別指示なしに修正・再build・backend実行をしない。
検証を通過するまでproduction implementation、registered query recipe、query数、anchor-first、baselineは変更しない。

inner用Farkasは対象inner LPだけの実行不能を証明し、元クラスの実行不能にはしない。
outerから元クラスへの結論は別途包含関係を要する。今回はいずれのcertificateも取得していない。
backend setup停止はtechnical inconclusiveであり、RA-RTEの優位性・非優位性の証拠ではない。

## 公開と停止

[evidence manifest](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/evidence_manifest_v1.json)に
全新規pathのSHA256と保護照合を保存した。
外部source一式、package、binaryはGitへ含めず、取得URL、SHA256、exact build commandと失敗logを公開する。
これらは停止に至った手順の記録であり、同じpilotのretry指示ではない。
研究共通API/production source変更0、新science/synthesis、circuit/matrix/trajectory、DF/molecule/NPZ/GPU、
IS/CTS/new angle/precision、authorization/marker変更は全て0。
**資料commit・push後mandatory STOP。次の判断をGPT側へ戻す。**
