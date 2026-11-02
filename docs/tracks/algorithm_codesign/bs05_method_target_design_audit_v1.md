# BS-0.5：method・target設計監査

2026-10-06 JST。**DOCS_ONLY_AUDIT_COMPLETE_AWAITING_GPT_REVIEW / mandatory STOP。**
受領判定は`REVISE_BLOCK_SYNTHESIS_DESIGN_BEFORE_IMPLEMENTATION`。
[GPT review原文](inputs/bs05_design_gpt_review_20261006.txt)の範囲で文書だけを監査した。
科学結果、実装承認、pilot契約、研究方針の採択ではない。

## 監査結論とGPTへ返す判断

**現candidateにはstandard sparse/operator LCUから独立したmethod deltaが定義されていない。**
一文での位置付けは「finite-RTEの有限Pauli多項式を既知のsparse演算で取得し、同じphase-aware辞書・残差条件のoperator LCUへ渡す構成」である。
現candidate専用armとnew-method claimを比較案から外す。全trajectoryを列挙しない、dense Mを作らない、phaseを保持することは、同じ処理を許したgeneric sparse LCUにも可能である。
これは同じsolutionが数値的に得られたというscience結果ではなく、入力・処理・最適化問題の同一性の設計監査。

研究モードの推奨は**design/application study候補への縮小をGPTに照会**。
finite-RTE taskでの適用域・多資源trade-offに、既知の制約緩和以上の追加知見を期待できるかは未決定。
それもなく古典取得差もないとGPTが判断すれば、このrouteはpilotなしで閉じる。
独立差のないmethod pilotは勧めない。研究B全体の終了や新RQの採択はCodexが行わない。

次reviewは本書→[ordinary有限RTEの形式的baseline](bs05_ordinary_finite_rte_baseline_v1.md)
→[pilot案v2 amendment](block_synthesis_pilot_amendment_v2.md)の三文書。
GPTには「applicationとしての価値／route closure」「baseline実装の未固定事項」「後段の必要性・範囲」を判断してもらう。
実装・semantic tests・source freeze・別authorization・scienceへ自動進行しない。

## 同一性監査：弱いdense baselineを作らない

M=ΣP mP P、D={Vj}を固定し、real係数aとphase-bearing atomで
`||Σj aj Vj − M|| ≤ δ`を満たす問題を考える。目的・q・precision・budgetも同じものを与える。
complex係数は固定phase atomへのreal展開等を共通にし、candidateだけ便利な辞書を増やさない。

| 項目 | 現candidate | strong generic sparse/operator LCU | 独立差 |
|---|---|---|---|
| 情報 | Pauli係数・signed time・K/r・固定templates（I0）、sparse積和（I1） | 同じI0、同じsparse積和I1を使用可能 | なし |
| finite M取得 | P3(−ix Rhat)、境界は別P3の積 | 同じ有限多項式を同じ順序で評価 | なし |
| Pauli積 | phaseとsupportを保持し同一wordを集約 | 同じbit/phase表現・集約を使用可能 | なし |
| 辞書生成 | D0/D1の固定template、phase付きexact重複だけ統合 | 同じtemplateと統合。lazy列生成も許す | なし |
| 係数問題 | 同じM/D/δ、同objective・予算 | 同じ問題 | なし |
| residual取得 | sparse bound、必要なら小型dense I2評価 | 同じbound/I2 accessと費用 | なし |
| 再利用 | Rhat²/Rhat³、辞書列、共通境界積のcache | 同じcache/reuseを許す | なし |
| 古典time/memory | sparse polynomialとLCU処理の費用 | 同じ手順が使用可能 | 未計測だが差を生む別手順は未定義 |

Pauli word積には符号だけでなく±iを戻す。s個の入力word、Sk=Rhat^kの非zero word数とすると、
naive sparse積和は各次数でO(n·s·S_(k−1))のword処理、格納はO(n·Σk Sk)程度（bit演算モデル・係数bit精度は別会計）。
K=2では次数3まで、Sk≤min(4^n,s^k)。境界積にも同じsparse乗算を使える。
この上界は両者で共通で、constant-size DF fragmentや大系での改善保証ではない。
係数bit長、lazy dictionary、solver、独立residual検証、sampling処理まで含めないCPU比較を優位と呼ばない。
||ΣP eP P||≤ΣP |eP|も両者に与えられる保守的bound。dense評価回避だけで新証明とはしない。

既存案に異なるobjective・辞書・情報access・acquisition algorithmを追加して差を作ることは今回行わない。
将来method候補を戻すには、strong genericが同条件で利用できない具体処理と証明義務を、GPT reviewで数値前に定義する必要がある。

## targetはO/Cの二層

ordinary分布のUωはsystem上の**phase込み**unitary、補正b>0は各固定target内で一定。
OのtargetはM=bΣω pω Uω、全ρに対するcoherent信号Tr(ρM)。
Cのtargetはjoint-spaceのΦ=Σω pω Ad(Ctrl(Uω))、測定ではbΦを使う。
boundaryのbは二つのB2の積。finite MがnonunitaryでもΦはTPである。

joint密度行列の01 blockに対し、Φはρ01をΣω pω ρ01 Uω†へ写す。
そのcoherenceはM/bで決まるが、11 blockはΣω pω Uω ρ11 Uω†でありMだけでは決まらない。
Oは前者のoperator条件、Cは全blockに対する条件を要求する。
system channelのglobal phaseを捨ててからcontrolする操作はこの関係を保証しない。

O：Σj aj Vj≈M、Σj aj=1やbというTP由来の総和条件は課さない。
C：Σj gj Ad(Vj_joint)≈Φ。exact TPならΣj gj=1、補正後aj=b gjの総和はb。
positive Cはさらにgj≥0。単一unitaryや無補正convex mixtureでnonunitary Mを直接再現したとはしない。
同じfirst-moment誤差に戻すときも外側b・confidence・phase・workspaceを全て数える。

O/Cを同じwinner表へ入れない。cross-layerは**より強いfull-channel制約を要求した資源差**という診断。
観測された差が負にもなり得る別algorithm比較であり、非負の必要最小追加costの証明ではない。
辞書・solver・予算・bias条件も違うなら、差全体を制約だけへ帰属しない。

## prior-artの追加とscope

既存の[claim表v1](block_synthesis_claim_and_baseline_matrix_v1.md)は固定履歴として保持する。
次は今回の差分。文献の全引用網・全証明を再検証したという主張ではない。

| primary / 確認箇所 | 既知 | BS-0.5への含意・未閉条件 |
|---|---|---|
| **Wada, Harada, Suzuki, Tokunaga, Yamamoto, Endo**, [Tradeoffs between quantum and classical resources in linear combination of unitaries, arXiv:2512.06260v1](https://arxiv.org/pdf/2512.06260v1), 2025-12-06。§II Eq.(1)–(17)、§III.A Thm.1–2 / Eq.(19),(22),(26),(32)–(38)、§III.B Thm.3 / Eq.(44) | coherent/randomized LCU間のgrouped手法、回路とsamplingの交換、partition refinementに対するreduction factorの単調性 | block/group化はmethod deltaではない。強いbaseline候補として必須review。原taskはKρK†のobservable（normalized ratioまたはnumerator）で、BSのlinear Tr(ρM)・固定Φとは別。ancilla/group circuitとconfidenceのadapterが未定義のため同target armへ自動追加しない |
| [Sparse PS v2](https://arxiv.org/html/2402.15550v2), §II.1–II.3 / III.1,III.4（v1監査を継承） | channel辞書のsigned分解、l1/residualとの交換 | C側に置く。強いoperator LCUへも同じsparse入力と辞書を許す |
| [Positive synthesis v1](https://arxiv.org/html/2510.05816v1), Problem1.1 / §4（v1監査を継承） | 正のchannel混合、単一qubitのT評価 | C側のみ。2-system＋ancillaへのlifting/実装costは別証明義務。single-qubit scalingを全wrapper期待costに流用しない |
| [TE-PAI v2](https://arxiv.org/html/2410.16850v2), Appendix A（v1監査を継承） | randomized evolutionでのprobabilistic synthesis、matrix/channelと測定taskの区別 | randomized simulationへの積層という上位構想を新規性にしない |
| [Resource-optimal IS v1](https://arxiv.org/html/2603.13495v1), §II / Thm.1（v1監査を継承） | costとsecond momentの共同設計 | 一つのcost×momentを新methodとしない。BS案のprimaryは多資源Pareto |

Wada論文名・arXiv IDは利用者の追加回答で確定。[確認receipt](inputs/bs05_grouped_lcu_reference_confirmation_20261006.md)。
HTMLは§II冒頭までの不完全変換だったので、指定v1 PDFの上記本文を確認した。論文bytesのimmutable hash監査や全文のrepository複製は行わない。
Wadaのgroup-sizeだけの単調性を任意の同サイズpartition間の全T-cost単調性へ広げない。
additional workspace=0の旧案では非singleton coherent groupのancillaを無視できない。
grouped armを実装する場合はlinear-task adapter・group集合・extra workspace・PREPARE/SELECT cost・phase・confidenceを先にGPT reviewし、条件が違う別scopeとして保存する。

## 固定履歴・成果物・停止

baseは`2c8c022db39c3582d6175fcf25a6752037727a21`、独立branchは`track-b-bs05-design-audit-20261006`。
v1三文書と旧design JSONは変更せず、本書とv2 amendmentを現在の入口とする。
[監査manifest](../../../artifacts/track_b_bs05_design_audit/2026-10-06/audit_manifest_v1.json)に入力identity・変更path・保護対象を記録する。
SP-1 result `9d2bb1fa439748b02084bd9fbc9b10a705328f8a`、raw result/marker/source/authorization、SP-0.5、BF/BM closure、Track A証拠は維持。
M1/M2をBのnew held-outへ転用しない。新条件の取得・開封はない。

**実装・tests・matrix・solver・library生成・synthesis・scienceは全て0。RUN_READY=false。**
数式定義とsource text照合をsynthetic semantic test済みやfinite-time certificateと呼ばない。
必要文書のみcommit/pushしてGitHub固定commitからGPT reviewへ戻し、mandatory STOPする。
