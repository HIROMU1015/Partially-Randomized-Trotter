# A2 SPEC-04〜07：SOURCE実装・人工テスト技術報告

2026-10-18 JST。**技術分類 `SPEC04_07_IMPLEMENTED_SYNTHETIC_PASS`。** 指定された固定設計に従い、単一determinantのU/W、K0 raw BCH行和のK_SOS、grouped相殺を保持するCbar2、RF0/RF1/EW2と強化交換子対照の十分step比較を実装した。**人工SOURCE検査の完了**であり、実分子certificate、実入力での実用性、相対形式の優位、新数学保証、A2採択ではない。

[API・数学対象・独立性](source_api_and_scope.md)／[GPT引継ぎ](decision_handoff.md)／[機械可読テスト結果](synthetic_test_results.json)／[全test log](synthetic_test_log.txt)／[静的検査](static_inspection.json)／[人工証拠例](synthetic_evidence_example.json)／[証拠索引](evidence_index.json)／[公開receipt](publication_receipt.json)。

## 1. 固定資料と実施scope

- 基点・SPEC-01〜03：[e921d178291ff3cfc3bde5934f99bcee94d3e082](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e921d178291ff3cfc3bde5934f99bcee94d3e082)。旧SOURCE・旧test・旧報告を変更せず再利用した。
- 固定設計：[7af83266f96b714bb3fa2f1ef5e4bc89faf37db3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_a2_certificate_connection_design_20261018)。[接続仕様 §§5〜10](../qc_a2_certificate_connection_design_20261018/certificate_connection_spec.md)、[比較仕様 §§1〜7](../qc_a2_certificate_connection_design_20261018/comparison_and_cost_design.md)を実際に確認した。
- K0 raw辞書の固定参照：[58d8672702284d73321cc67a26c4640adaddde5f §4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)。RF/EW2・centered対照の式は[db1819587df36ee8833b37227ca0083faa0408ec](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/db1819587df36ee8833b37227ca0083faa0408ec/docs/research/qc_relative_form_ew2_step_audit_20261017/mathematical_audit.md)に接続した設計を使う。新たな定理・科学レビューは開始していない。
- 最新共通運用指示2026-10-17 §8.4と、今回のユーザーによるSOURCE・人工test限定指示を適用。科学条件、試行、authorization/grant/markerは発行しない。

専用branch `qc-a2-spec04-07-synthetic-20261018`、独立作業領域 `/tmp/prt-qc-a2-spec04-07-synthetic-20261018`。実装は新namespace [src/trottercert/a2_spec04_07/](../../../src/trottercert/a2_spec04_07/)、新test [tests/test_a2_spec04_07_synthetic.py](../../../tests/test_a2_spec04_07_synthetic.py)。旧namespaceはそのまま共有前提検査として使う。親worktree・他研究のSOURCEや仕事は変更しない。

## 2. SPEC-04：UとW

固定mode順・mode0 LSB・creation昇順の単一determinantを明示入力とする。空・全占有も許し、重複・範囲外・個数N不一致は拒否する。試行状態のnorm²はexactに1。generatorは一体作用のbit prefix符号と疎振幅表から

\[
U=c+\langle D|Q|D\rangle+\sum_\ell\lambda_\ell\|A_\ell D\|^2,
\quad W=U-L
\]

をexact Ratで生成する。負λを絶対値へ置換しない。checkerはoccupied listへの挿入・除去のparityから別にoccupation行列を作り、`(A*A)[D,D]`を計算する。全遷移表、期待値内訳、signed係数、単位、trial、親identityを再照合する。U<Lを拒否し、Wを表示floatや有利なclampから作らない。determinantは変分上界の証拠用であり、step保証のexact-ground accessを供給する状態ではない。

## 3. SPEC-05：K0辞書とK_SOS

PF表示順を1-basedの位置へ明示対応させる。固定raw辞書の `κ_jjk += 1/24` と `κ_klj += −1/12` をすべて残す。可換性や相殺を利用してraw辞書を縮めない。

generatorは各tupleの四つのB寄与をstreaming行和へ足す。添字の一致時も寄与を落とさない。checkerは全candidate tupleから別の条件列挙で辞書を生成し、明示的な対称B行列を構築して行和・最大値を再検査する。

平方根はcallerが明示した `sqrt_bits=p>0` のdyadic gridにおける最小上側点uを整数isqrtから生成する。checkerはisqrtを使わず、`u²≥z`、grid一致、z>0で`(u−2^-p)²<z`、z=0でu=0をexact検査する。精度を結果に応じて増やす機構はない。K_SOS_upperはirrationalな真の行和値の**上界**であり、真の値と等しいとはしない。根のwitness・辞書・全行和・単位Ha²・digestを保存する。

## 4. SPEC-06：強いCbar2

全fragmentのbase operatorは元のQまたはsigned λA²とする。外側PSD scalar shiftだけは交換子内で落ちる。A²のJW展開で生じるidentity項を任意に消す設計ではなく、元H/controlled絶対位相のscalar台帳も旧SPEC-02で維持する。

各jでまず `R_j=Σ_{k>j} F_k` を集約してから、**group全体**の `[F_j,[F_j,R_j]]` と `[R_j,[R_j,F_j]]` を展開する。generatorは正確なJW Pauli積、checkerは独立occupation行列の積・差を使い、`2^-n Tr(PX)`で全Pauli係数へ戻す。全支持集合・符号・実虚部・exact zeroを照合する。Hermitian double commutatorのPauli係数はexact realを要求する。

centerしたcaps `a=f_j/2, b=(Σtail f)/2` に対する標準norm積は第一groupで `4a²b`、第二groupで `4ab²`。各groupについて構造を保持した係数L1とcentered capの**両方を保存**し、小さい値を採用する。Cbar2は第一groupを1/24、第二groupを1/12で加算する。粗いuncentered対照へ戻す選択肢は設けない。二つの指定手順を実装したが、best-known norm envelope全体を尽くしたとはしない。

手計算できる二mode例では第一L1=4、第二L1=6、Cbar2=2/3を確認した。別の人工例では非零capにもかかわらずtail内のpositive/negative平方の相殺でgroup全体がexact zeroとなり、Cbar2=0を両経路が検証した。これらは人工algebra検査であって、分子での改善例ではない。

## 5. SPEC-07：適用域とstep比較

固定設計のη0、η1、η=min、B_R、centered b_C/B_Cをそのまま使う。RF0/RF1/EW2は `h>0, hΛ<1/4, η<1`、強化対照は `h>0, hΛ/2<1`。対照をRFの半径へ切り詰めない。logの解析branch、centered branchとscalar復元、exact-ground支持という理論条件は記録上区別し、実QPEのalias/readout/実装が検証されたとしない。

generatorはexact有理式と確率比較、checkerは独立に**正の分母を払った多項式不等式**で同じ判定を再計算する。rational bisectionの全probe・status・境界等号を保存する。境界をfloat sqrtやleading近似から認定しない。

| 状態 | 処理 |
|---|---|
| policy未指定 | `NOT_EVALUATED_PENDING_POLICY`。科学条件を補わない |
| W=0またはΛ=0 | `TRIVIAL_KNOWN_ENERGY`。0除算・優位例に使わない |
| Cbar2=0 | 有限のcentered log domainのopen supと、別の`COMMUTATOR_ZERO_EXACT_S2`全実h経路を分ける。relative優位を認定しない |
| K_SOS=0 | RF1の四次剰余を残す |
| bracket下端0 | `ZERO_ANCHOR`。受理された実行stepとして返さない |
| strict domain edge | 分母を評価せず`DOMAIN_EDGE_OPEN`。実行点へ丸めない |
| 反復予算で目標幅未達 | `UNRESOLVED_ARITHMETIC`。自動retry/precision増加なし |
| intervalが重なる | `ORDER_UNRESOLVED`。等値とはしない |
| 同じexact rational rootを両側で確認 | `EQUAL_SUFFICIENT_STEP` |
| 互いに分離したbracket | 宣言した対照に対する十分stepの大小だけを分類 |
| model/sector/PF/raw/期待policy不一致 | `IDENTITY_MISMATCH_INCOMPARABLE` |
| d≤0、α外域、不正trial/certificate等 | 拒否record。修復・tol緩和なし |

同じ情報poolと同じd/αで比較し、情報の取得費が無料とはしない。費用は未測定欄をnull・理由付きで保持する。最大**十分**stepの比較であり、真のPF限界step、最適QPE step、総資源改善、科学的GOを出力しない。

## 6. 検査結果

**最終98 test methods PASS、subtest 2,629 PASS、failure/error/skipped 0。** 新規SPEC-04〜07は60 methods・1,140 subtests、旧SPEC-01〜03の無変更回帰は38 methods・1,489 subtests。subtestは同一unit-test内のパラメータ例であり、独立した科学実験の件数ではない。

テストには正常系、負λ・zero caps、実/複素Hermitian、空/全占有・0 modes、CAR符号、全basis列、人工乱数の1〜3mode bundle、BCH重複添字、dyadic sqrt境界、grouped相殺、厳密な確率等号、RF/domainと対照/domainの違い、K=0四次項、open endpoint、予算未達、等値/両方向の比較、policy未指定/不一致、再hashした証拠改変、SOURCE/hash/単位/親context改変、I/O失敗のpartial保存を含む。generator側の関数を例外へ差し替えてもcheckerは動作する。

開発中の最初の55-method試験で、人工testの期待値に1件の誤りがあった。K=0、h=1/8のRF1確率は1/15なので、α=1/10では受理が正しい。四次剰余を検出する拒否testの人工αを1/100へ訂正した。科学条件や式の緩和ではない。その後58-method PASS、境界/失敗保存を加えた最終60-methodと38-method回帰を記録した。未解決のtest failureはない。

最終記録harnessの実測はwall約3.84秒、CPU/peak RSSは[JSON](synthetic_test_results.json)に記載する。32 CPU/affinityと利用可能memoryをread-only確認した。短時間のexact integer workloadであり、monkeypatchのisolationを保つ一つのunittest runnerを用いた。CPU/RAM/worker/wallの人工capや共有設定変更はしていない。GPU/BLASを起動せず、速度向上率を主張しない。SOURCE AST・禁止backend import・checkerからproducerへのimport非依存・SOURCEのfloat/complex literal不使用は静的PASS。Python -Bによる直接unittestで、pytest conftestやscience runnerを読み込まない。

## 7. 残る限界と保全

checkerは完全独立開発・形式証明ではない。Python int/Fraction、canonical JSON・hash、表現helper、SPEC-01〜03 checkerとその既知保証は共有する。新checkerの係数入力は共有前提checkerが受理したexact modelである。bit復元を三つ目の独立実装で再検証したわけではない。詳細は[API資料](source_api_and_scope.md)。

特に独立occupation行列と全Pauli traceは**指数的**である。small artificial referenceとして完成しているが、実分子規模での安価な取得可能性・runtimeは未検証。巨大入力のOOM/OS強制終了を自動復旧・自動永続保存するscientific runnerはない。APIはrecordを返すだけで、将来callerに保存責任がある。SOURCE I/O失敗は理由とpartial proposalsを返す。synthetic tagは自己宣言で、実データの偽装を防ぐsecurity機構ではない。

旧SPEC-01〜03のSOURCE/test、科学SOURCE、raw/科学契約/authorization/grant/ledger/markerは無変更。新規実装のdiscoverabilityのためPROJECT_MAP、各README、概要・validation statusに追記し、manifestの既存result_setsを変えず人工実装recordを1件追加した。旧scientific validation・既知FAIL・停止状態を更新していない。保全監査と公開同一性は[receipt](publication_receipt.json)を参照。

**実入力certificate 0、新規科学計算・保存科学数値再解析0、simulation/native compile 0、SPEC-08 0、科学authorization/grant/marker 0、新科学レビュー0。** A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認を維持。公開と保全確認後mandatory STOP。次の判断・承認事項は[handoff](decision_handoff.md)にまとめる。
