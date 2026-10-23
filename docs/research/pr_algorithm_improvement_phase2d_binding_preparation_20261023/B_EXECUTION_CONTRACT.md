# B基本契約：11 logical cases、科学実行は別承認

文書の状態は `FINAL_SCOPE_DRAFT_ENVIRONMENT_UNRESOLVED`。科学scope／SOURCE／case／精度規則を固定した案であり、authorizationではない。専用environment lockが未成立なので実行承認可能な完成packetではない。

## 1. 対象と固定順序

P0 exact fixtureを採用する契約案。物理model、JW/register、2粒子X0X1、外部scalar、R内I、LD1、二fragment、S/P collectを[B_INPUT_SPECIFICATION.md](B_INPUT_SPECIFICATION.md)に固定する。新しい係数／fixture探索／r/K最適化を加えない。

| 順 | ID | 依存 | 一回の生成対象 |
|---|---|---|---|
| 1 | B.INPUT.physical | なし | canonical仕様、exact rational orbital V、二fragmentのS/P alternatives、typed metadata |
| 2 | B.INPUT.catalog | physical | 別4label、scalar、frame参照、catalog exact lambda |
| 3–6 | B.ARMS.SS / SP / PS / PP | physical | 各canonical atom map、L、exact lambda、I/u/s/v coverage |
| 7–10 | B.CONTEXTS.SS / SP / PS / PP | 同arm | 各F1 contextのcanonical finite-RTE law metadata |
| 11 | B.CONTEXTS.catalog | catalog | T=1/20,q=r=1,K2のcatalog law metadata |

各IDを一回だけattemptする。基本scopeは `NO_BUILD_NO_EVENTS_F1_LAWS_ONLY_PRICE_UNBOUND`。F2は数値lawを作らず、固定仕様とL由来件数だけを引き渡す。5 CONTEXTSにF2の4件を隠して追加しない。

## 2. 実際の関数単位

機械台帳は[B_CALL_INVENTORY.json](B_CALL_INVENTORY.json)。logical 11件だけで計算回数を表現しない。成功時の主要呼出しは、materialize_alternatives1、frame_coefficients2、pauli_coefficients2、rational_frame4（Pauli代数内の再構成2を含む）、make_fragment_alternatives1、catalog_atoms1、canonicalize_arm4、coverage4、exact lambda sum10。

law_record／tail bridge／make_rte_configは各5回。旧SOURCEのdataclass検査を含め、finite_rte_distribution10、paired-order weight40、Taylor residual-bound25、paper normalization20回。RTEComponent構築は `sum L_a+4`。これはscalar metadataの既存API内部検査で、F1の独立mean/operator検証ではない。疎map内の積・加算回数は生成後のmapサイズに依存し、固定scientific caseの内部workとして記録する。新しいpilot、sampling、event列挙は0件。

frozen rte.pyの一回の科学的importと、そのNumPy importは最初のCONTEXTS case内に含める。今回は実行していない。B実行でAPIが失敗したらそのcaseを失敗としてSTOPし、独立probeやretryを追加しない。

## 3. Precisionとtolerance

係数、rotation、orbital matrixは既約有理数のexact代数、exact zeroだけ除外。S originを保ち、選んだP fragment間の同一wordを加算後abs/sign/probabilityを決定する。Bはfull Fock Hamiltonian/Gamma(V)数値matrixや状態vectorを生成しない。

lawはIと同じbinary64変換と既存finite-RTEを使う。nonfinite／underflow、p mass差>1e-12、SOURCE内のparameter rel1e-14/abs1e-15、probability atol1e-12等の整合条件で停止する。これらは既存APIのmetadata資格で、独立RTE正しさ証明やphysical signalのbias認証ではない。

固定K2に対し、既存 `step_taylor_truncation_residual_bound(tau,2)` の有限非負値bを取得し、API toleranceを `nextafter(b,+inf)` とする。Kは自動選択しない。test定義の暫定1.0は使わない。このtoleranceはAPI constructor整合のためだけで、bias allowance .025を満たす証拠ではない。F2 K0のtoleranceはBで数値化せず、同じAPI用方針の採否を別F2契約へ残す。

## 4. 保存と引継ぎ

全生成物をcanonical JSONでimmutable保存し、bytes/hash/producer caseをmanifestへ登録する。生成・成功statusを別ファイルへ書き、attemptや部分結果を上書きしない。最後のIO処理でsource/env公開用summary、F1 input-only bundle、L由来count/range descriptorsを作る。これは追加のscience caseではなく、既存recordのコピー・hash・整数件数計算である。

B出力のF1 bundleは価格とF1 environment未binding。F1実行契約やgrantへ自動変換しない。価格はPRICE_PROVIDER_UNBOUND、intervalはUNPRICED_FALLBACK、shotはSHOT_COUNT_UNRESOLVED。

## 5. Preflight・実資源・STOP

B科学実行の明示承認、固定SOURCE commit／10file closure、数学spec文書／call台帳／schema、専用env、11case DAG、未消費authorization、fresh approved run ID/outputをstdlibだけで照合後にscience importへ進む。今回packet／grant／run IDを発行しない。

4mode・二fragmentの小さいexact代数と依存DAG、最初の失敗でSTOPする条件に合わせてsourceは登録順処理とする。CPU/RAM/GPU/worker/wall-timeの人工capは設けない。実行前に空きCPU/RAM/GPU/diskと非干渉を確認し、もし実workが重いと判明すれば科学条件を保った新SOURCE版・構成を固定する。共有env/cache/OS/scheduler/quotaや他jobを変更しない。

source/spec/env不一致、sign/zero/scalar/convention不整合、coverage不足、precision/API資格失敗、未登録call、priceを0とする処理、build/compileへの逸脱、原本保全違反、output重複は即STOP。同runの自動retry/resumeなし。11case成功でもmandatory STOP。F1/F2、旧C1、RQ-04、追加samplingを開始しない。
