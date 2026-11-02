# Track B G6：return generator監査結果・GPT引継ぎ

2026-10-10 JST。

**数理候補のideal恒等式とbounded finite-bit局所構成を確認した。独立新規性とnative費用込みの資源価値は未確定。mandatory STOP。**

これは[GPT G5研究方針レビュー](../../research/track_b_G5_research_direction_review_20261010.md) §19を利用者が採用したことに基づく、
新しい探索的数理監査である。旧G5 toyの資源再探索や科学実行ではない。
技術状態は`G6_IDEAL_IDENTITIES_VERIFIED_NATIVE_AND_METHOD_VALUE_UNRESOLVED`。
GPTによる新method採択・次性能pilotのGOではない。

## 作業場所と入力identity

- execution branch：`track-b-g6-return-generator-audit-20261010`。
- 独立worktree：`.worktrees/track-b-g6-return-generator-audit-20261010`。
- base G5 publication：`23adf61f9342abff20f059be69089e89f66f7797`。
- rootの未commit資料から選んだのは利用者指定G5レビュー**一文書だけ**。
  snapshot SHA256 `e2c0a442cd27b73016a0c0eea2439c416684f4985f8c3c592de79bd915c6a900`。
  [origin/byte identity](../../../artifacts/track_b_g6_return_generator_audit/2026-10-10/review_input_identity.json)を保存。
- GPT側の`formal_return_selfcheck.py/json`は提供されていない。記載hashを独立結果とは数えない。
  G6は自身のraw語oracle・系列engineを新規実装した。
- 前提：有限正rational p、Hermitian involution Q_i²=I、odd m、0<x<=1、sigma±1、finite P_m。
  分子/model geometry/basis/DF rank/L_D/PF delta windowは該当しない。形式testは性能条件の採択、held-out開封ではない。

## 独立再導出の判定

| 命題 | 判定・実用上の限界 |
|---|---|
| finite Taylorの全return集約恒等式 | PASS、位相・word順序を保持 |
| P_(n+2)<=(n+1)chi P_n | PASS、挿入coverの上界であって等式ではない |
| short-step集約非負性 | PASS、0<x<=1に限定 |
| F_i/Gの形式母関数 | PASS、**既知free-product Green式のZ2特殊化** |
| even-suffix parent pairing | PASS、回転generatorはQ_i、odd語全体ではない |
| ordinary envelope・local accept | PASS、global B_new/tableなしに生成可能 |
| zero-fill mean/moment/range | PASS、全試行数で平均。未知Zをbudgetへ無料使用しない |
| finite-bit proposal・rational weight | 近似mean＋明示biasで構成。finite-bit exact元meanとは言わない |
| controlled native circuit・cost/error取得 | **未実装・未検証** |
| 既知法に対する独立新規性・総資源優位 | **未確定** |

一般証明は[独立数学監査](g6_independent_mathematical_audit_20261010.md)。
補足として16 focused testsが通過した。有限fixtureだけを一般命題の証明とは扱わない。

## 文献監査で変わった点

[prior-art表](g6_prior_art_and_method_delta_20261010.md)ではAomoto–Kato本文、CTS補足、PR v2関連節まで取得して比較した。
G5レビュー時に未取得だった本文を今回確認した事実を区別する。
free-product母関数、低次数吸収、Euler pairing、zero-fill、cost-aware ISは既知components。
CTSにも層別/Markov非列挙samplingがあり、全展開evaluatorをCTSの唯一の対照とはできない。

確認範囲で全結合samplerの同一記載までは特定しなかったが、priorityや非自明性の証拠ではない。
残る候補は「全returnを保つ局所生成が、parent依存角度/control/classical queryの費用を払っても価値を持つか」に限定される。

## Finite-bit・prototypeの実装範囲

[実装/access監査](g6_finite_bit_and_access_audit_20261010.md)に、bit長、state、support、range、coefficient biasを明記。
系列precomputeはO(Lm²) rational operations、local parent/全child queryはO(m³+Lm²)、保持系列O(Lm)。
これは入力p table/bit算術を数えるが、native oracle/rotation取得は含まない。
新しい[prototype](../../../src/trottertracks/algorithm_codesign/return_aggregation.py)は全word宇宙・全cost表・B_newを読まない。
dyadic law、有限rational補正weight、symbolic tangent、fixed-bit index mappingまでを実装した。
random draws、量子測定、angle synthesisは0。
局所word依存angle数にO(m)保証はなく、online synthesis費用も未測定。

## 検証・資源・provenance

[focused tests](../../../artifacts/track_b_g6_return_generator_audit/2026-10-10/focused_tests.json)は16件。
independently chosen p=(3/7,4/7), x=2/5, m=1/3/7;
p=(1/5,3/10,1/2), x=5/7,m=5及びx=1,m=7; p=(1),x=4/5,m=9。
ancillary checksは同off-domain pのraw degrees2/4、x0、invalid domain、semantic mutation。
G5登録toy p=(3/4,1/4),x=1/8/1/4やGPT checkerのfixtureを開封・再評価していない。

- raw形式語3,965、GF係数3,794、positive語503、insertion178、parent envelope170。
- signed formal mean12、digital event weight63、global digital L1 budget1。
- technical bundle1、science runs0、旧one-shot retry0、full suite0。
- wall0.219209s、CPU0.218931s、peak RSS39,743,488 bytes。
  wall90s/CPU60s/address-space512MiB/formal words50,000/output16MiB内。
- LP/synthesis/Hamiltonian/matrix/circuit/trajectory/DF/molecule/NPZ/GPU/quantum calls=0。
- 新technical marker SHA256：`3c6955ea58a8351f374998297d22b34e0598814021cc5102045d54820e6a773d`。
  旧science marker・authorizationとは別であり、これを新science authorizationにしない。
- [873保護path](../../../artifacts/track_b_g6_return_generator_audit/2026-10-10/prior_protected_hashes.json)の
  old source/contract/result/authorization/marker/STOP・index既存prefixを照合。
  shared API実装、Track A、root dirty worktreeは変更・stageしていない。
- test/source/runtime/入力reviewのhashとscopeを結果前保存した。
  source-bound local checksであり、external replication・immutable CIではない。

## GPT/利用者へ戻す判断

本回で新しい科学条件・baseline・閾値・合成precisionを採択していない。
R0/G1/G4-Aの限定成果、G4-B/G5の優越・固定toy閉鎖、過去technical failureを保持する。

利用者が次の重要review開始を承認した場合、GPTが次を判断する：

1. 全結合generatorに独立した方法上の差が残るか、既知componentsの組合せnoteへ縮小するか。
2. controlと可変angleを安価に取得できる具体access契約があるか。
3. ある場合だけ、その一点を判別する最小の費用実証を設計する価値があるか。

G6は科学的GOを出さず、**mandatory STOP**。
新angle synthesis、旧toy再探索、全面v4、DF/分子、PR/QPE総費用、次性能pilotへ自動進行しない。
GitHub公開は独立branchの必要資料のみ。公開full commit SHA・固定review URLはpush後に利用者へ報告する。
