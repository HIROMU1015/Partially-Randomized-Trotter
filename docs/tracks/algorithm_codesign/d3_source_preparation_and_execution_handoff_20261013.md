# D3専用source・実行guard準備と引き継ぎ

2026-10-13 JST。**READY_FOR_SEPARATE_D3_ONE_SHOT_AUTHORIZATION**。
利用者の「続きを行って」を、直前に指定されたsource実装・guard・登録外tests・公開の許可として扱った。
科学実行の承認としては扱っていない。
採択済み[比較計画](d3_guarantee_price_preregistration_v1_20261013.md)から独立branch
track-b-d3-source-preparation-20261013を作成。基点3e1011e3cb7e306a1cc992644aa70af4bf6ff514。

## 1. 実装したものと固定条件

[B固有namespace](../../../src/trottertracks/algorithm_codesign/rte_robustness/README.md)に専用実装を追加した。
旧D1/D2 module、source、契約、authorizations、結果、markers、STOPは変更しない。
共通trotterlib APIを変更していない。

- G0/P0、G1/P0、G0/P1、G1/P1の4 ledgers、各12行、計48行。
- 同じ3-qubit explicit synthetic provider、p=(1/5,3/10,1/2)、tau=5/7、sigma+、状態000、P5、r=1,2。
- ordinary、partial-return、closed P3、local full-return、closed P5、matched CTSへ同じ取得機会。
- 六方式の保存law/IR、production moment/range、finite-bit precisionと同じconfidenceを継承。
- G0は3x^6/720、G1はHermitian contractionのx^6/720。反復・実装誤差・axis factor2を分離保存。
- **全ケースのnative誤差会計は旧1e-6**。新sequence取得1e-7の余裕をshotへ返さない。
- 新しい36 keysを各一回、pygridsynth2.0.0、旧tool/runtime、seed0、100dps、strict phase。
- signed Rは同じsequenceのactual adjointを用いる既存native templateの意味論。独立負角合成・projective guardは使わない。
- CD、DF/分子、L_D、決定論PF window、SP、別provider、新r、angle/seed gridは対象外。

新契約は[execution_contract_v1.json](../../../artifacts/track_b_d3_source/2026-10-13/v1/execution_contract_v1.json)。
旧design JSON原本は保持し、そのSHAと科学条件・capsの全field一致を機械的に要求する。
execution contractへの変更はkind/status、今回の準備許可を表すmetadataと実行path blockだけ。
別途作成するauthorizationがlive実行可否を決め、契約内の原design由来のfalseは準備時のmetadataである。

## 2. 会計・取得の流れ

起動認可後に保存inventory/lawを読み、全36新keysのguard/coverageが完了してから登録会計を行う。
同じrowのP0/P1費用を一つのsaved-event passで計算。保存1890 eventの読み取りだけで、新IR・回路・provider matrixを生成しない。
G0/G1ごとのbudgetを二つのprice条件へ共有し、同じG内のN・moment/range・acceptance一致を検査する。

G0/P0では全budget field、native切片、slope、旧conservative tail/hardをexactに再現することを要求する。
実際の登録anchor比較はone-shotの一部で、今回未実行。
r2は静的R非相殺certificateをold event SHAへ結び付け、保存D2のliteral per-trial savingsを再使用する。
219,303 pairは再実行せず、depth別完全帰属の欠測も残す。
new N/priceへold総費用・tail totalsをそのまま流用しない。

全h>=0のenvelope、primary partial/P5の双方r選択、fixed-r対照、quantum Pareto、保証/価格/交差項を保存する。
48行のexact affine入力を残し、重複する巨大pair-difference表は比較reportへ再複製しない。
最小領域とtieをcompactに保存し、元入力から全relationを復元可能にする。
512 KiB超のJSONはUTF-8安全なlossless chunksとbytes/SHA manifestを付ける。output capを緩めない。

各12-row ledgerは同じfamilywise0.05のcounterfactual resource forecast。
96-axisの同時量子実験について0.05を主張せず、量子shotを実行しない。
prep/readout hは記号変数、少なくともh_1Q>=h_Tを区別する。古典online費用は未知でありゼロにしない。
最良r/provider/compilerの一般性、新規性、algorithm採択をCodexが判定しない。

## 3. 実行保護と外側process

準備中のauthorizationはpending、source_commit=null、runs0、science_execution_authorized=false。
未承認起動はruntime・登録入力・Git source queryより前に拒否される。
新source Sのauthorization-only直接子Aを要求する。Aの変更pathはauthorization JSONと任意receiptだけ。
remote SHA=A、clean execution worktree、source/contract/manifest/protected hashes、runtime/tool identityを確認する。

外側ownerが新namespaceをexclusive createし、marker、job-start receipt、ledger jobを保存して一回だけ起動する。
ledgerとkey workerはparent PID、source/HEAD、contract、authority、inventory、tool、markerのidentityを照合する。
ledger/key started receiptはexclusiveで、同じjobの再起動を拒否する。
backend entry直前に各keyのexclusive tokenを記録。token消費は完了した合成件数とは区別する。

内部periodic Guardと、外側process-tree監視を併用。終了直後にもCPU、peak RSS、output/logを確認する。
高速な終了やsignal終了を内側COMPLETEだけで正常完了としない。
inner_terminalは原記録として保持し、outer receipt、最終terminal、STOPを別保存する。
全48行・36 keys・source identity・exit0が整合した場合のみcomplete。
cap/guard/partial/output failureではtechnical inconclusive、prefixを科学的勝敗に使わずretry0でSTOP。

resource mapのmatrix telemetryは「provider/signal matrix0」と「new keyのstrict 2x2 guard36」を分ける。
Clifford+T backend内部の2x2演算を、全matrix処理0と呼ばない。

## 4. 資源と加速

workers2、numeric threads各1、per-key CPU20秒、total CPU1800秒。
RSSはparent/worker512 MiB、aggregate1536 MiB、process address1536 MiB。
output16 MiB、log64 KiB/key、cache4 MiB/36件、sequence20,000 chars、review chunks512 KiB。
shot forecast cap1e8/axis。total/phase/per-key wall capsはnull。
独立keysを並行取得し、同じkeyとbudgetを共有する。matrix validationやpair列挙は追加しない。
速度倍率を主張しない。登録fixtureでのruntime・outputサイズは未測定で、cap不適格ならSTOPする。

## 5. 技術検証

[focused tests](../../../tests/tracks/algorithm_codesign/test_d3_robustness_source.py)：
最終63件PASS。51→56→60→63はsource/test追加に応じた準備中のcheckpointsで、raw logsを保持する。
全て人工scalar/word、temp namespace、mock worker・48-row orchestrator・外側processに限定。
実際の合成器APIは呼ばず、登録budget/price/pairを計算していない。

主なcoverage：

- G0の旧会計とのoff-domain exact一致、G1のfactor、非unitary積、native account固定、budget exhaustion/shot cap。
- all36 inventory/precision/seed/hash、missing key、signed価格、worker一回・failure・再起動拒否。
- source/A/direct parent/dirty/remote、marker/parent/job ownership、実行中manifest変更の検出。
- strict phaseを捨てない2x2人工identity/W control。ここでの2x2 checksは2回、登録provider matrix0。
- zero-fill、r2保存literal savings、4 ledgers、mock48行、factorial効果、compact envelopeのtie。
- CPU/RSS/output/log/worker上限、alarm競合/復元、fast childのfinal log cap、partial terminal不使用。
- UTF-8 lossless chunk復元、改ざん・output failureのpartial保存、禁止APIへの到達がない静的検査。

固定runtimeのPython3.10.12、全packages、pygridsynth/mpmath source treesとexecutable identityはread-onlyで一致を確認。
これらはlocal technical evidenceで、immutable CI・外部再現・科学的優位ではない。
登録source全体を実データでend-to-end起動したことにはしない。

## 6. 次の一回実行に必要なもの

技術準備による追加GPT詳細レビューを必須とはしない。新科学問題は見つからなかった。
次は利用者の**別途明示one-shot実行承認**。
承認後に固定source Sから専用branch/worktreeと直接子Aを作成・pushし、launch gateを確認する。
今回A、実行marker、登録結果namespaceは作成していない。

将来の入口は、固定runtimeで

    python scripts/tracks/algorithm_codesign/run_d3_one_shot.py --source <full source S>

を一回のみ。内部ledger/key CLIsを独立起動しない。
結果はD3_GUARANTEE_PRICE_MAP_COMPLETE_AWAITING_GPT_REVIEWまたはD3_TECHNICAL_INCONCLUSIVE。
どちらでもmandatory STOP。成功/失敗を理由にprecision、seed、r、capを自動変更・再実行しない。
重要な科学的解釈・研究継続は結果取得後のGPTへ戻す。

**今回：source準備のみ完了。登録再会計、新合成、旧run再開は0。mandatory STOP。**
