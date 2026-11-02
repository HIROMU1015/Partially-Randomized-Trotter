# A2 SPEC-01〜03：限定SOURCE実装・人工テスト報告

2026-10-18 JST。**技術分類：`SPEC01_03_IMPLEMENTED_SYNTHETIC_PASS`。** 固定DFモデルのexact intake、identity、固定N・S2/scalar台帳、K0の既知signed PSD化とL/f_j/Λのgenerator・独立checkerを実装した。**人工データの技術合格のみ**であり、実分子certificate・全A2基盤・認証step優位・科学的GOを認定していない。

[SOURCE/API・形式](source_api_and_scope.md)／[判断用ハンドオフ](decision_handoff.md)／[人工テスト結果](synthetic_test_results.json)／[テストlog](synthetic_test_log.txt)／[人工証拠例](synthetic_evidence_examples.json)／[静的検査](static_inspection.json)／[固定証拠索引](evidence_index.json)／[公開receipt](publication_receipt.json)。

## 1. 基点と実装範囲

設計正本は[固定commit `7af83266…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_a2_certificate_connection_design_20261018)、特に接続仕様§§2〜4とSOURCE表SPEC-01〜03。必要な定義・API境界を確認し、既存RF/EW2/K0/K0.5の証明・先行研究レビューを再実施していない。

新しい実装は[独立namespace `src/trottercert/a2_spec01_03/`](../../../src/trottercert/a2_spec01_03/)に置いた。既存`src/trotterlib`・runner・科学入力loader・既存tests/索引・結果・契約には変更を加えない。新APIに実ファイルpath/npz/DFHamiltonian objectを受け取るadapterやCLIは設けていない。in-memoryの明示raw byte envelopeを使い、`input_class=SYNTHETIC_TECHNICAL`を要求する。

このtagは自己宣言のscope gateであり、実分子データを偽装して入力できないことを保証する認証機構ではない。今回の実行ではテストSOURCE内の人工係数だけを使用し、保存科学配列・実分子入力を一切読まなかった。

## 2. 実装したこと

### SPEC-01：exact intake・identity

generatorのbinary64復元は整数sign/exponent/mantissaと2の冪によるFraction、checkerの復元はbit文字列のsign/exponentと有理significandによる別実装とした。normal/subnormal、正負、±0、複素実虚部、little/big endianを扱う。復元にfloat計算、`as_integer_ratio`、近似分数化、eigensolverを使わない。NaN/Inf、不正dtype/shape/endian、非有限値、不正hex、非Hermitianを理由付きで拒否する。`complex128`は宣言endianのbinary64実部・虚部の連続格納と定義した。

モデルはc/q/λ/gのfull値、mode/block順、CAR/JW/bit規約、単位・意味論versionからidentityを作る。raw envelope・array bytesのidentityは別に保持する。endianやmetadataが違っても数学的係数が同じならmodel_idは同じになり得る。raw_input_id・MatrixRefは異なる。signed zeroもrawでは区別し、正確係数では0へ正規化する。

Int/Nat/Rat/ComplexRat、canonical JSON、Digest/ObjectRef、MatrixRefを実装した。科学実数をJSON floatへ変換しない。重複key、非canonical整数/分数、JSON数値、循環object等を拒否する。整数の十進encode/decodeはPythonのプロセス全体のdigit-limit設定を変更せず、chunk処理で行う。固定分母の最大bit長や科学精度の新しい閾値を設定していない。

原入力のHermiticityはexactに検査し、対称化や`real_if_close`で修復しない。full/correction-only・scalar/correction・truncation roleをtagで区別する。`df_truncation_value`はmetadataのままで、cへ加算しない。単位は設計どおりc/q/λがHa、gが無次元。元分子への厳密対応は常に未認証とする。

### SPEC-02：固定N・PF・scalar

0≤N≤mode数、mode順・occupation-bit・CARの固定規約を検査する。入力文法はidentity、number-conserving一体項、その平方のみであるため、固定N保存の構造証拠を作る。古典indicesや基底外skipを利用していない。spin/S²への拡張、量子sector encodingは実装しない。

全fragment（零λ・零capも含む）の順序を宣言し、palindromic S2の表示積・逆順の回路時系列を保存する。各fragmentの総時間係数1、半stageと中央stage、全inventory、零fragment保持をcheckerが位置ごとに再検査する。順序を変えるとpartition_id・pf_idが変わり、checkerの期待順序も明示的に変えなければ受理しない。S2はpalindromeのため表示列と時系列の値が同じでも、規約fieldを省略しない。

各F_j=B_j+s_jIと、`L+Σs_j=c`を由来付き台帳で保存する。元のsigned Bとfull cを一度使うframe、controlled scalarを相対位相または明示frameで復元する意味論を記録するだけであり、実回路生成・controlled操作・compileは実施しない。

### SPEC-03：PSD shiftとL/f_j/Λ

generatorは行列の全entryに対してΣ(|Re|+|Im|)をexactに加算する。checkerは、exact Hermiticityを別検査した後、対角項＋上三角pairの2倍という別経路でcapを再構成する。

設計/K0の既知構成をそのまま使う。q_-=-s_q、q_+=s_q、u_l=s_g_l、一体はQ−q_-I、正符号平方はλA²、負符号平方はμ(u²I−A²)。f_0=2s_q、平方capは|λ|u²。Lはc+q_-−Σ_{λ<0}|λ|u²で、Λはshift後fragment capの和。数値対角化やcap最適化を行わない。

checkerは元係数とcapから各base weight・PSD rule・identity shift・fを再検査し、**全scalar shiftの和をcから引く別経路**でLを再構成する。generatorのLを再利用していない。positive/negative/zeroの符号、全fragmentと係数digest、Lの由来、cap/単位、親context、witness/hash・SOURCE identityを検査する。`LowerCertificate`と`FragmentCapCertificate`だけを生成し、未生成のU/W/K_SOS/Cbar2等をplaceholderでPASSにしない。

## 3. Generator/checkerの独立性

| 対象 | generator | checker | 共通依存・限界 |
|---|---|---|---|
| binary64復元 | int word・mantissa・bit shift | bit文字列・有理significand・scale | raw descriptorのschema/bytes幅、Python整数/Fractionは共有 |
| Hermiticity | 全i,jのconjugate equality | 対角虚部と下三角pair | 規約は同じ。暗黙修復なし |
| cap | 全entry絶対値和 | diagonal＋2×upper triangle | exact Hermitian係数を前提とする既知boundは共有 |
| L/scalar | q下限と負λ項から生成 | c−全検証済みfragment shift | scalar/base/PSD ruleをcheckerで別々に確認 |
| PF | forward/center/reverseの列を生成 | 各位置・期待fragment/時間・総和を検査 | 固定S2の数理定義は共有 |
| schema/identity | 共通canonical/型/raw schema | 同じrepresentationを検査 | hashing・Rat・source hash・MatrixRefは共有。完全な独立開発・形式証明ではない |

checker moduleはgeneratorをimport/callしない。package API初期化は両moduleをexportするが、checkerの計算はgenerator関数に依存しない。generator関数を例外に差し替えたテストでもcheckerが正常受理し、generator capを故意に0へ変更したテストでは拒否した。SOURCE identity検査はファイルhashを共有するprovenance検査であり、数理検算の代用ではない。

外部の信頼できるmodel_id、期待N、期待PF順序は呼出側の責任である。raw全体と証拠を首尾一貫して別モデルへ作り直した場合、それは別の正しい数学対象になり得る。元から欠けたblockや既に二重加算されたcを、来歴なしで復元することはできない。`expected_model_id`を渡すことで意図した固定対象との不一致を拒否できる。provenanceの自己宣言を元分子との同値証明とはしない。

## 4. 人工テストと実行境界

**最終テスト：38件PASS、subtest 1489件PASS、failure/error 0件。** subtestはunit-test内のパラメータ例であり、独立した科学実験1489件という意味ではない。[JSON](synthetic_test_results.json)に全test IDと各subtest件数、SOURCE/test hash、Python版、実測時間・process RSSの観測範囲を保存した。

| coverage | 主な検査 |
|---|---|
| bit復元 | 手計算の8種wordと両endian、seeded有限512 word×両endianをtest-only stdlib float/Fraction oracleへ照合。normal/subnormal/最大有限/±0/NaN/Inf |
| positive | 実・複素Hermitian一体/平方、positive/negative/zero λ、zero cap、N端点、M=1、空modeの形式端点、identity再現性、JSON roundtrip、合法順序 |
| negative | dtype/shape/endian/単位、非Hermitian/対角虚部、mode/block inventory、correction-only/二重scalar role、N、非対称PF、過小cap/Λ、過大L、親context、partial/追加未承認field |
| mutation | raw λ/g/cとexact係数の不一致、mode/PF順、符号、scalar、witness/hash/SOURCEの改変。hashを付け直した証拠、scalar式と親hashを整合させた不正shiftも拒否 |
| 独立人工oracle | 40組の人工対角model×4粒子数の全occupationでPSD・cap・H=L+ΣFをexact確認。非対角複素modelのoccupation-bit行列作用でoperator/scalar一致、test-only有理複素Schur検査でF≥0とfI−F≥0 |
| 失敗保存 | 不正入力のdigest、理由・段階・partial証拠、人工I/O failure。generator success flagはcheckerへ渡しても受理根拠にならない |

small matrixは**人工test内のexact algebra**であり、新しい科学simulation・実分子certificate・benchmarkではない。generator/checker本体へCAR/JW行列生成やnorm最適化を追加していない。試験用floatは人工byte構成・独立bit oracleだけにあり、認証SOURCEの係数復元へ流入しない。

最初の31件PASS後、複素g/PSD oracleと入力拒否、hash付直し攻撃等を追加し34件・37件へ拡張した。失敗入力のdigest保存とcheckerの入力snapshot処理を追加して最終38件を実行した。いずれも人工scopeの開発検査で、科学one-shot/retryではない。記録されたテスト失敗はなかった。実装時には巨大整数の十進変換でglobal設定を変更しない処理、証拠の共有containerによる検証中変更を避けるdetach、source I/O失敗のpartial返却を整えた。

テストはPython3.11.1のstdlib unittestをdirectに使い、repositoryのpytest conftest/既存integration testsを読込まない。SCF/FCI/DF生成・保存配列・NumPy/SciPy/OpenFermion/Qiskit等を起動していない。32 CPU/32 affinity、空きRAM等をread-only確認したが、約1秒規模の小さな整数testであり、patchによる依存性検査のisolationを保つ一つのunittest runnerを選んだ。CPU/RAM/worker/wallに人工capを設定せず、BLAS/GPUや共有環境を変更していない。速度向上を測定したとはしない。

## 5. fail-closedと限界

intake受理、generatorの未検証出力、checkerのSPEC-01〜03限定受理を別verdictにする。scopeは全出力で `SPEC01_03_SYNTHETIC_TECHNICAL_ONLY`。失敗時はreason code/path/stage、可能なら提出bytesのdigestと途中objectを返し、verified_quantitiesを空にする。未知量は0で埋めない。

返却recordはcanonical JSONで保存でき、[人工例](synthetic_evidence_examples.json)に受理/改変拒否/途中失敗を保存した。本APIは自動的にdiskへ書かないため、将来のcallerが結果を永続保存する必要がある。OSによる強制終了・電源喪失等でpartialの自動永続化まで保証していない。SOURCE読込のOSErrorはtechnical failureへ返すが、全システム故障の回復runnerは今回実装しない。

このSOURCEの数学的正しさは形式的に証明されていない。checkerも共通schema/hash・Python runtime・同じ既知公式を前提とする。capは保守的full-Fock三角上界で、sectorでtightとは限らない。良いLや小さいΛ、実入力でのruntimeやstep改善は未検証。原分子への厳密保証、量子state access、回路・native精度、platform certificateは提供しない。

SPEC-04〜08のU/W、K_SOS、grouped Cbar2、RF/EW2 step、QPE/native/state/costは**未実装**。科学的GO/NO-GO・A2採択の自動判断、科学実行契約・authorization/grant/markerは作成していない。

## 6. 保全と終了

基点 `7af83266f96b714bb3fa2f1ef5e4bc89faf37db3`、専用branch `qc-a2-spec01-03-synthetic-20261018`、独立作業領域 `/tmp/prt-qc-a2-spec01-03-synthetic-20261018`。変更は新namespace・新test・この専用技術資料だけ。既存SOURCE・raw・契約・markerと親worktreeのHEAD/branches/registry/index/exclude/bytesを照合する。

最新版共通運用指示2026-10-17 §8.4と固定正式状態を維持。**実分子科学計算、実入力certificate、保存科学数値再解析、SPEC-04〜08、新科学レビュー、科学authorization/grantは各0件。** 人工generator/testは今回の明示承認内で実行した。公開と保全確認後mandatory STOP。次の担当と必要な別途承認は[handoff](decision_handoff.md)に記す。
