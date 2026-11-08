# 新A S1 coverage訂正：独立one-shot再評価

2026-11-08 JST、SERVER 2。Parent **5fbdf0b9d7046f8fce04405757c1ee075e42fb4b**、旧SOURCE **c96e0a3a31fb0db5f1ae9a8903567012e21b78f9**、科学入力base **800554fbb64c21e584bb29a1a8f52490ab4163e0**。Branch `codex/pr-reference-noncentral-s1-coverage-fix-20261108`。

**Coverage問題を訂正し、SOURCE RTE限定版の4行を欠損補完なしで取得した。全4行で保存V1より非悪化だが、保存joint予算への残余は負。全catalog版の4行は未資格のまま。S2は再実行せず、独立S1 one-shot完了でmandatory STOP。**

## 1. 最小訂正・旧SOURCEの再利用

[旧報告](../pr_reference_noncentral_s1s2_numeric_20261108/REPORT.md)の問題は、`weighted_parts`が全mapping内のNoneを調べ、指定alphabetとarmに関係なく両armを未資格にすることだった。SOURCE component集合に含まれない2 basisの欠損が、RTE限定版にも伝播していた。

新しい[独立S1 runner](../../../scripts/diagnose_pr_reference_noncentral_s1_coverage_fix.py)は、**variant・armの式が参照するtemplateだけをcoverage判定し、そのarmだけを評価する**。必須template欠損はnullのまま残す。Unused templateを0、便宜値、別recordで補うことはしない。

旧`blob/read/archive/template_weights/rational_json`と既存helperの`weighted_template_max`は関数bodyを変更せずAST再利用した。旧`weighted_parts`の`pair_max`、`lane`、q/r assertion、B′／Referenceそれぞれの算術式もASTをそのまま使用した。変更は全域None判定と両armの同時returnを、依存template判定＋arm別returnへ分離したことだけである。CSWAPがないB′の0も旧固定式のまま。

旧runnerの`main`、`pareto_ids`、offer作成関数はロードしない。新SOURCEは[実行前PREFLIGHT](../../../artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/PREFLIGHT.json)でSHA256 **a7f2ed1d651a986d0909f0c8b070b972e0c6cf2cbd7669d7b07e41d66b978796**に固定し、実行後も同一だった。旧runのretry/resumeではなく、新namespace `science/run01`の独立one-shotである。

## 2. 固定条件・人工テスト・preflight

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A／Qiskit1.3.0。既存B′4-CX／Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event相対位相・X/Yを不変更とした。

M17 q131072／r32／bits88、B′626／Reference630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを維持。算術は旧integer／Fraction、保存floatは同じbinary64 hex keyへ復元し、保存guard・現word・保存Q＋G・loweringだけを使った。

[人工test](../../../tests/test_pr_reference_noncentral_s1_coverage_fix.py)は最終7件PASS（必須templateごとのsubtestを含む）。未使用basis欠損、variant間／arm間の独立、必須basis／D／products／CSWAP欠損、0補完の禁止、旧式とのFraction完全一致、hex key／context-max、S2除外、出力namespace分離を確認した。実行前のAST testでdocstringの認識を修正した後にSOURCEを固定しており、科学実行開始後のSOURCE修正・retryは0。

Preflightで7保存入力のbytes／hash、固定metadata SHA256 `396f11326d2a5bc86316d0997c744a315b60a856ce5912a1fab064676f285027`、旧SOURCE／helper、既存S2 artifactのGit blob SHA、新output不在、Python binary64前提と実環境を確認した。承認証明書・grant等は作成していない。

## 3. Coverageと8行の資格

SOURCE alphabetは保存STAGE1_STRUCTUREの`source_basis_probabilities`のkeyだけ（7 ID）。確率値、新alphabet、観測subsetを導入していない。

| Variant | B′必須template | Reference必須template | X/Yを含む資格 |
|---|---|---|---|
| 全catalog | b mapping 32、うち4欠損 | ref 33＋actual inverse 5、ref側4欠損 | 4行UNQUALIFIED、上界／残余／非悪化flagはnull |
| SOURCE RTE限定 | b 20、欠損0 | ref 21＋actual inverse 5、欠損0 | 4行の非中央上界を評価済み |

全catalogの欠損は旧runと同じ2 basisのforward／inverse。各template・literal・保存contextは[COVERAGE.json](../../../artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/COVERAGE.json)へ保存した。SOURCE限定版はこれらを参照しないので、欠損を伝播させていない。逆に、必須templateの欠損はそのarmの結果をnullにする。

全catalogとSOURCE限定版の直接大小比較は、前者がnullなので**未判定**。数学的な集合包含を、今回の二つの数値上界を比較した証拠にはしていない。

## 4. SOURCE限定版の非中央上界・V1比較・残余

以下はD／frames／products／CSWAPのlogical errorを保守的にjoint spectralへ加算した、既存compiler条件CとSOURCE閉包付きの**非中央上界**。Raw disk、必要shot、実誤差の下界ではない。概数表示であり、正本は[RESULTS.json](../../../artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/RESULTS.json)のexact Fraction。

| Arm／axis | 非中央上界 | 保存V1非中央load | 保存joint予算 | 残余（予算−上界） | 非悪化flag |
|---|---:|---:|---:|---:|---|
| B′ X | 0.003597257009846 | 0.06974207906143 | 9.504670287679e-6 | -0.003587752339558 | True |
| B′ Y | 0.003597257009846 | 0.06974207906143 | 9.504670287679e-6 | -0.003587752339558 | True |
| Reference X | 0.003595837287171 | 0.06973975573584 | 3.517584922187e-5 | -0.003560661437949 | True |
| Reference Y | 0.003595837287171 | 0.06973975573584 | 3.517584922187e-5 | -0.003560661437949 | True |

| 内訳（X/Y共通） | B′ | Reference |
|---|---:|---:|
| D | 9.123060223391e-5 | 8.981222210257e-5 |
| frames | 0.003506023501946 | 0.003506023501946 |
| products | 2.905666360394e-9 | 1.452833180197e-9 |
| CSWAP | SOURCE不存在／0 | 1.102897993010e-10 |

非中央上界は保存V1より小さくなったが、全4行で保存joint予算を超える。**現在のwordと今回の加法十分式では、他項を追加する前に予算が閉じない。** 上界の不足であり、真の誤差・RPE失敗・アルゴリズム不可能性の下界ではない。X/Yで同値なのは今回の非中央4groupの式が同じためで、wrapper／scalar等を同一視・消去した結果ではない。

## 5. 検算・保全・未実行事項

[CHECKS](../../../artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/CHECKS.json)に8行構成、4数値行、定義された場合だけのvariant比較、非負・有限Fraction、S2ロード／callが0であることを保存した。[POSTRUN_CHECKS](../../../artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/POSTRUN_CHECKS.json)ではparts和＝上界、予算−上界＝残余、V1比較flag、SOURCE／input hash、旧S2 artifactのGit blob SHAが不変であることを確認した。科学runnerは再実行していない。

[SOURCE_INPUTS](../../../artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/SOURCE_INPUTS.json)に旧関数と算術ASTの来歴・7固定入力・新SOURCE／preflight hash、[ENVIRONMENT](../../../artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/ENVIRONMENT.json)に実環境を保存した。GPUA100-gleap02、Python3.12.3／stdlib、64 CPU affinity、実行前MemAvailable約468 GB。軽量な有理演算を単一processで約1.62秒で実行。CPU／RAMの独自capや共有設定変更はない。

旧SOURCE／runner／raw／RESULTS／CHECKS／STOP／grant／ledger／markerは不変更。S2の375 frontier再計算・offer再構築・43候補採用は0。旧S2結果は旧commitに保持し、新しいS1結果へ混ぜていない。新word／precision／policy選択、guard・合成、回路build/compile、provider／consumer、simulation／sampling、中央探索、RPE式・shot数・総T／T-depthの計算も0。

中央、scalar、event、emitter、shell、B′ scalar域外、Reference vacuumのdomain／到達域、event到達性、terminal資格を未確定・非zeroのまま残す。M17成功・全18round総T削減・強B′への優越は認定しない。正式共通指示2026-10-25、適用AGENTSと既存科学条件を維持した。著者Codexの限定診断であり、GPTの重要科学レビュー実施記録ではない。

## 6. Mandatory STOPと次の科学判断

**[Mandatory STOP](../../../artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/STOP.json)。** 独立S1 one-shotは完了し、SOURCEを固定した後の変更・retry/resumeは行っていない。

次担当は**GPT／ユーザー**。非中央frame／Dの構造保証をさらに鋭くするか、保存S2候補の有限policy検証を行うか、又は保留するかを判断する。今回の負の残余を根拠に新規合成を必須とせず、S2／S3・新guard・domain・actual-adjoint実装・RPE・全18round総Tへ自動進行しない。
