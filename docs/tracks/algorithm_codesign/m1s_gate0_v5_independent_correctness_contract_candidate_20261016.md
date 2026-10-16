# M1-S Gate 0 v5：事後独立correctness検証契約候補（2026-10-16）

**CANDIDATE_NOT_ADOPTED / IMPLEMENTED_UNEXECUTED。新規科学計算0件、mandatory STOP。**
今回は独立検証SOURCE・契約候補・静的検査の準備と公開のみ。既存v5を再実行していない。
将来の実数値検証には、この候補のユーザー採択・採択記録固定と、さらに別途の明示的な実施承認・新しい専用authorizationが必要。

## 既存作業の確認・引継ぎ

別Codexの公開branch track-b-m1s-gate0-v5-independent-correctness-preparation-20261016、commit d6d35cfb79b97f9cb218c9169b427aacf6a4a334に、未完成・未テストの4ファイル草稿を発見した。
元worktree/branchへ書き込まず、固定v5結果baseへそのcommitだけを別worktreeでcherry-pickした。
[発見・再利用台帳](../../../artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1/existing_work_discovery.json)に原本hashと差分の役割を保存する。
decoder、dyadic、one-body、matrix square、CAR contraction、scale/combine、Hermiticity、ladder correspondence、Pauli traceの11 helperは元草稿とAST同一。
state/metric層とHD照合・provenance・guard/save adapterを補完した。旧v5の計算helperはimport／呼出し／流用していない。

## 固定対象

|対象|識別子|
|---|---|
|v5結果・base|cdcb7f6f151df217960611f502eb1bac8be63f58|
|v5 SOURCE|69de022dc8142f703423b1ce0bd42f05e123954e|
|v5正式契約|50b3669b9253eb889fa4daa87e793646df2d5138|
|v5証明|b5a185cc0c3aae0e477bfc52b93061e86feb7dec|
|独立検証SOURCE S|4787cf90b053d251bf6392988af35516bdbbe663|
|v5 raw result SHA256|8e32925e99b64ce6bc2c939e45ba51485d18b1eb4b62952e22a0debe1d9d4ec5|
|[machine契約候補](../../../artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1/independent_validation_contract_candidate.json) SHA256|0635dc992eeda64d4d19b2c61569391c6179ad7d74547262da50875731c1896f|

SOURCE7ファイル、raw NPZ、比較結果、v5契約・SOURCE・proof、Python実行ファイルと選択したstdlib5ファイルをhashで固定する。
raw NPZは8652 bytes／SHA256 3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a。
[SOURCE・比較対象identity](../../../artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1/source_identity.json)を参照。
科学scopeはlinear H4 1.00 Å、STO-3G、8 spin orbitals、保存DF rank12、保存順HD prefix G3/G6/G9（0–2/0–5/0–8）、HR=残り、delta window/finite-step評価なし、保存数値sector oracle。
結果公開後の**事後的な独立再現性確認**であり、新規held-out validationではない。

## 独立referenceの対象と経路

入力は保存binary64 bit値が表す正確な数。有理数targetに対して、H=constant I+sum h_pq a†p a_q+sum_l lambda_l A_l²を構成する。
occupationのソート済みwedgeへcreation/annihilationを挿入・除去し、挿入位置の符号から全256状態の行列を作る。
各A_l²を疎行列積とCAR normal contractionで別々に組み立てて厳密一致を要求する。local tensor ladderとの対応も厳密照合する。

全65536個のPauli語についてc_P=Tr(PH)/256を、XOR対角と整数FWHTで計算する。
full-H 1回、HR before identity 3回、HD after identity 3回、計7回を全語走査し、exact zeroを含むsupport bitmapとdigestを保存する。
Pauliのlocal I/X/Y/Z行列から別にtensor行列を組み立て、traceからの再構成をfull/HD/HRで厳密照合する。
HRのexact identityを除きHDへ移し、直接prefixから組んだHD+HR=fullとfull-map minus HRのHD係数の一致も要求する。

lambda_R=sum|c_P|と理想sampling lawはexact rational。state norm²とsector embedding/basis orderingを独立に確認し、保存vectorを上書き・再正規化せず正確な元norm²で期待値を割る。
WはHRのinteger matvecのnorm²、vはsigned Pauli期待値のexact first/second moments、rho=(W+v)/(2W)で求める。
W/v/meanのmatrix/law整合、0<=v<=W<=1をexact arithmeticで照合する。W=0ならrhoはUNDETERMINED。
残差はsqrtを丸めずexact residual² [Ha²]として保存し、元solver絶対品質(1e-8 Ha)²と比較する。これはexact固有状態距離・full-Fock ground・非縮退の認定ではない。

Python [Fractionのfloat入力はその値を正確に表す](https://docs.python.org/3.11/library/fractions.html)。本SOURCEは整数/有理数で比較を行い、Decimal32桁は表示専用とする。
reference丸めallowanceは**0（SOURCEとCPython整数/有理数実装が正しいという前提）**。正式証明・hardware/stdlib certificateではない。
任意係数cutoffや近似有理数化、Decimal80桁に対する未証明2^-80 allowanceは用いない。

## 比較規則・保証範囲

既存係数・W/v/rhoをreferenceの入力にしない。全3分割の独立referenceを完成しcanonical snapshot hashを固定した後でのみ、保存結果JSONをparseする。
保存HR tableからの行列組立は最後の**比較側**に限定する。比較後もreference snapshot hashが変わっていないことを確認する。

|項目|固定比較規則・単位|
|---|---|
|exact algebra、basis embedding|差0|
|state norm² quality|元の2^-40、dimensionless|
|Pauli係数|exact/computed support unionでL1差[Ha]<=採択済みalpha_R=gamma_(713+r)S_R。入力尺度から別実装で算定|
|HR行列|Frobenius差²[Ha²]<=256 alpha_R²。PauliのFノルム16とL1三角不等式に基づく|
|支持・符号差|L1上界超過ならFAIL。上界内の支持/符号差は厳密support照合をUNDETERMINEDとして調査待ち|
|lambda_R|保存係数のexact abs sumと保存lambdaの差を既採択gamma_(65568) Lで囲い、exact targetとの差をalpha_R追加で照合|
|sampling probability|abs(saved coefficient)/saved lambdaのnearest-even binary64 cellを整数bitと有理数midpointで確認。保存probability massは元2^-46|
|identity|exact moved identityと保存移動量の差をalpha_Rで照合。HD scalar beforeは元global係数上界で照合し、after加算は既採択fsumのgamma_34 absolute envelopeで照合し、nearest-evenは診断のみ。phase符号/quarter-step記録も確認|
|W/v/rho|保存binary64値をFraction化しexact referenceとの差を元2^-24/2^-24/2^-12以下と比較。reference丸め誤差の追加は0|

gamma許容式は既採択の数学モデルを別実装で再記述するもので、そのモデル自体を今回独立に証明したわけではない。
共有CPython/Fraction/物理hardwareという依存も残る。[独立性・coverage台帳](../../../artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1/independence_and_coverage.json)を参照。
一致を観測して許容値を選ぶ設計ではなく、実入力/模擬入力へ本SOURCEを代入した回数は0である。

**保存比較データの限界**：v5 rawにはfull-H行列とHD非identity係数/行列が保存されていない。
それらの独立targetは再構成・exact内部照合できるが、旧v5の未保存行列そのものとの一致を確認できない。
この部分をUNDETERMINED_NOT_SAVED_BY_V5とする。HR/law/state/metricsのavailable_comparator_statusがPASSでも、全要求scopeの最終statusはUNDETERMINED。どこかにFAILがあればFAILを優先する。
このデータ欠落は追加の技術実装だけでは解消しない。旧v5再実行や新比較データ生成は今回認可していない。

## Resource・保存・実行guard

[Resource見積り](../../../artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1/resource_estimate.json)：CPU1 worker、各thread1、GPUなし、2 GiB address space、result/log各128 MiB、context4 MiB。
500000000 primitive-event cap、integer numerator/denominator bit cap65536、matrix/coefficient cell上限65536、CAR transition cache4352、Pauli tensor cache512。
7×65536 trace cellsと7×256×128×8 FWHT butterfliesは静的構造からのcount。primitive-event capはroundoff gammaの演算数ではない。
疎行列/transition/DF blockを再利用し、Pauli expectationを3splitで共有し、有理数matvecをinteger momentへ置換した。性能実測・speedup主張はない。
scientific wall/CPU-time capなし。aggregate memoryは一つのworkerのAS制限で管理。caps到達・実行不整合はUNDETERMINEDとし、完了ledgerを保存してSTOPする。
全係数を参照・比較するが、出力は完全support bitmap/digestと各種最大64件の差分witnessへ抑える。これは計算上のcutoffではない。

実行にはresult base < 独立SOURCE S < ユーザー採択検証契約 C < 新しい別実施authorization Aを要求する。
旧v5の実施許可は受け付けず、独立検証専用schema/namespace、clean HEAD=A、全commit/hash/runtime/threadを確認してからreferenceをimportする。
outputは未存在directoryを排他的に予約し、その時点でone-shotを消費。context/log/result/checksumはxb保存、上書き・implicit retryなし。
全independent referenceを先に固定した上で、最初にFAIL/UNDETERMINEDとなった比較行までを保存し、残る比較行を未評価としてSTOP。
未保存full/HDによるcoverage留保だけではavailable scopeの3行照合を省略しない。
reference途中の例外、resource cap、interruptもprogressと全split statusを可能な範囲で保存する。OSの非捕捉killやI/O failureを数値PASSとはしない。

## 準備状態・未決事項

[SOURCE](../../../src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1)／[runner](../../../scripts/tracks/algorithm_codesign/run_m1s_gate0_independent_v1.py)／
[static checker](../../../scripts/tracks/algorithm_codesign/check_m1s_gate0_independent_v1_static.py)／[準備artifact](../../../artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1)。
構文・schema・旧helper非利用・情報流・guard拒否・保存仕様は静的検査の対象とする。reference kernelやNPZ配列decodeは呼び出していない。
数値correctness・runtime/メモリ実測は未検証。人工入力の数値テストも0である。
[未決事項](../../../artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1/unresolved_items.json)：検証契約のユーザー採択、別途実行承認、full/HD比較データ欠落、SOURCE/CPythonの未実行検証。
guarded adapterは実装済みであり、設計上は実施承認後に別adapterを新規開発する必要はない。検査PASSは実数値成功の保証ではない。
adopted=false、新科学実施authorization/grant/marker0、旧v4 STOP/null・v5 one-shot/raw/消費済marker・既存SOURCE/契約・元Codex/親worktreeを保全し、**mandatory STOP**。
