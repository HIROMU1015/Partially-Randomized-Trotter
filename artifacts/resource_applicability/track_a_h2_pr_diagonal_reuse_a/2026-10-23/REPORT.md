# H2 PR diagonal角度：既存実装・独立receipt再調査とA専用準備

**ケースA2。検証済みの既存primitive実装はそのまま再利用できるが、限定調査した保存資料に同一H2条件の独立diagonal receiptは見つからなかった。既存証拠のみではA残条件を解決できず、134/134軸ともUNRESOLVEDを維持する。独立witnessを追加するなら3 piece時間×2 block＝6 primitive評価が必要な案を準備した。今回は実評価0、代替契約未採択・COMPARE BLOCKEDのままmandatory STOP。**

基準 `65085ff7f818fd92af3424b1980f6bd9dbf1c742`、A専用branch `codex/h2-pr-diagonal-reuse-a-20261023`。新技術SOURCEは[SOURCE.json](SOURCE.json)。B作業のworktree・出力領域は使用していない。旧strict SOURCE `46a3b33e3717c4fadf11846e922caf17d10156d9`、旧契約digest `a65f3e90b7ed70701f57bcf1519e1d74acdfd1c45c320a5fe91d00d4d44d9ef2`、未採択の代替契約、原科学SOURCE/入力/結果/grant/ledger/marker/STOPを変更していない。

科学条件はlinear H2/1.0 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4。PR S2 `H2O32_c7ec590a32a859bc`/q4とPR Yoshida4 `H2O32_8a2ecdd0294ffb74`/q2、prefix1/K4/outer32、元RTE law/usual signed bias/symmetric-directional/native DF/scalar/full Hadamard wrapper/compilerを保持する。Full固定2候補とallowance/reference/shot政策も不変で、今回評価していない。

## 先に確認した既存SOURCEと証拠

実装前に[既存SOURCE一覧](existing_SOURCE_inventory.json)と[限定receipt探索](existing_receipt_search.json)を保存した。O00002は指定原本を直接読取り、期待SHA `f011e12b5df6c534fd610899abd2f0f8a17dd05e1cfefaf418a7cb09c0c9a8c7` と一致した。公開成果物へprivate原本や内部絶対pathはコピーしていない。

| 実装・証拠 | 確認した契約と再利用範囲 |
|---|---|
| `h46_new2_native.prepare` | 実SOURCEと原封印SHA一致。共通QR Gaussianをprocess内だけで選択し、既存`_prepare`へ委譲して必ず復元する。科学処理の書直し不要 |
| `pr2_matched_accuracy_m1_execution._prepare` → `prepare_df_partial_s2` | generation順prefix1、extract_identity_phase、coefficient_atol0、explicit_ordered_partition。元H2のone-body＋fragment0を準備する |
| `df_trotter.ops.one_body_diagonal_primitives` / `df_squared_diagonal_primitives` | 原封印と同一。返り値はglobal_phase、ordered RZ、ordered RZZ。gate/circuit appenderを呼ばずprimitive値だけを返す。新wrapperに式・角度算法をコピーしない |
| `h46_m5_native.compose` → native v5 | 実SOURCEでpieceのsigned_time、q1、delta/2の境界を確認。元SOURCEは変更しない |
| H4/H6 native検証SOURCEと保存報告 | 元QR・producerの使用履歴/負時間・control・単一状態検証の根拠。ただしH4/H6のrank/prefix/H/stateが異なり、今回H2の独立角度receiptではない |
| native Qiskit接続42-test保存receipt | 実producerの人工n2、time ±.3/0でのgate/IR/matrix検証。コード再利用の根拠であり、H2 n4の6 primitive値へ置換しない。今回は再実行なし |
| 既知H2の旧D/control/resultとinput/snapshot、現DIAGNOSTICS bundle | input/binding・正しさ・SOURCE・Gaussian・vector scopeはあるが独立6件のprimitive parameter receiptは確認できない |
| 元H2探索の既知run・task保存形式 | 指定run直下と525 task directoryのファイル名patternだけを限定確認。primitive/angle/preparationの専用保存fileなし。全server・他利用者・agent履歴の探索なし |

SOURCE inventoryの8つの科学実装fileは全て歴史的seal SHAと一致した。関数名だけの推定ではなく、入出力・call境界・関数本文のSHAも記録している。

原QPYの角度、compiled raw、cost/fingerprint cache、そこから作られたmetadataコピー、同じQPY内のmirror関係は独立証拠にならない。single-state error receipt・保存vectorやtail eventも、共通H/preparationから独立生成されたdeterministic primitive receiptの代わりにはならない。同じ値を別fileにコピーして受け入れる実装は拒否した。

この結論は上記の限定scopeでの未発見であり、未指定の保存媒体を含む絶対的な不存在を主張しない。同条件の独立receiptが別途提供されれば、SOURCE/input/時間/生成順序/依存関係/保存SHAを検査して新評価を省略できる。

## 時間定義の訂正と二重適用防止

[ユーザー確認](time_clarification.json)と[実SOURCE call監査](time_and_call_audit.json)。初期指示と前段DRAFTの「signed half-time」という名称は誤りだった。ユーザーは元SOURCEを優先し、次の未実行契約案への訂正を明示した。科学実行の承認は含まない。

| piece識別子・全時間 | primitive実引数 |
|---|---|
| `-0x1.5ca78b1289ec1p-1` | `-0x1.5ca78b1289ec1p-2` |
| `0x1.14ba2befab5c7p-1` | `0x1.14ba2befab5c7p-2` |
| `0x1.999999999999ap-3` | `0x1.999999999999ap-4` |

元経路は`compose(T=piece,q=1)` → `build_partial_native: delta=T/q` → `_append_block(time=delta/2)` → `ops` primitiveへそのままtimeを渡す。primitive関数はさらにtimeを1/2にしない。primitive内部の係数・global phaseの`/2`はHamiltonianの既存代数であり、入力時間の再半減ではない。

新wrapperは上表のprimitive実引数を直接元関数へ渡し、`_append_block`やnative lowererを呼ばない。piece値を直接渡す誤りと、引数をさらに半分にする誤りの両方を人工testで拒否。契約/各receipt rowにはpiece時間・primitive引数を別fieldとして記録し、SOURCEは別manifestへ結合する。

## 134軸の静的照合で解決できた範囲

[全134軸](axis_parameter_scope_134.json)で既存QPY cursorを再利用し、全pieceのdeterministic diagonalとdirectional側の保存phaseを読み取った。[6保存parameter群](six_saved_parameter_groups.json)は同じpiece/blockごとに全軸の保存角度/wire/phase/orderのdigestが一致。S2 33 trajectory/66軸、Y4 34 trajectory/68軸、原eventとcohortを保持した。

この6群は保存QPY由来であり、**独立inputには使わない**。6群の一致で解決できたのは、対象の抽出範囲、時間対応、同じ対象の保存field identityと比較入口である。`absolute_deterministic_diagonal_vs_common_preparation`は0/134軸、未確認134/134軸。新しいprimitive数値評価・保存値からのHamiltonian逆算・cost/shot/順位の再解析は0。

## 最小wrapperと未実行SOURCE

`scripts/h2_pr_diagonal_a.py`はstdlibのreceipt schema、入力由来検査、call制御、固定時間対応、保存QPYとの比較を提供する。`h2_pr_diagonal_a_backend.py`は既存実装へ直接委譲するdormant adapter。bootstrapのscience import、固定snapshotの読込、元prepare、元primitive呼出は今回一度も実行していない。

入力処理は固定snapshot/receiptの全array hashとHのbindingを確認する最小loaderで、旧production `setup`内のOccupationOracle・tail spectrum・reference等の不要な科学処理を呼ばない。prepareは既存関数をそのまま使い、元preparation SHAと共通Gaussian operation SHAを確認する。6回のprimitive呼出はone-body3、fragment3。prepare内のGaussian6群は既存prepare1の内部処理で、新しいevolution回路やnative lowererを追加しない。

科学的独立性の入口を分けた。独立stageはsnapshot/receipt・固定SOURCE/runtime・共通preparationのbindingだけを受け、QPY/event/cost/expected angleを拒否する。6件のreceiptを全て生成・serializeしてbytesのSHAを確定した後にだけ、compare stageへraw receiptとjobを渡せる。保存QPYはcompare stageのみで読める。新規数式やprimitive算法は実装していない。

比較は元QPY full-stream validatorを先に通し、その後でindependent receiptのordered RZ/RZZ/support/sign/hexとblock phaseを一致させる。fixed Gitの候補別signed scheduleとpiece数・順序を照合し、既知の別timeの混入も拒否する。source/input/prepのhash、独立input manifest、生成経路、scopeが揃わないreceiptは、値が一致してもPASSにしない。

現SOURCEのCLI `--execute`は入力を読む前に拒否し、real backendのactivationも必ず拒否する。人工domainのinject callbackだけがcall順・予算を検証できる。これはSOURCE/人工検証を準備した段階で、実科学を起動できるgrantやauthorization plumbingを発行した状態ではない。[有効化前の条件](execution_activation_requirements.json)。次の科学実行時には別途明示承認と、固定SOURCE/input/runtime/unused grant/rootを検査する独立activationが必要。追加するのは認可・封印・出力配線だけであり、科学処理を再実装しない。

## exact契約と受入条件

[未実行契約案](execution_contract_DRAFT.json)の予算は元案のまま、bootstrap1/setup1/prepare1/diagonal_primitive_evaluation6/archived_axis_parameter_compare134。sampling/request/event/build/lowering/compile/transpile/simulation/u/N/G/allowance/rankingは0。予算の増加0、科学実消費0。内部call順も人工ledgerで確認した。

[metadata-only preflight](metadata_preflight.json)と[CLI一致](metadata_preflight_CLI.json)はbinding/固定scopeを確認し、`BLOCKED_PENDING_SEPARATE_SCIENTIFIC_AUTHORIZATION`。独立receiptがなく実評価未承認なのでexecution_ready=false。旧strict契約や未採択代替契約のfieldを編集して通すことはしない。

将来の受入は、元H/input/preparation/Gaussian/SOURCEとsingle-half時間が一致し、6独立receiptの全ordered angle/support/phaseが134保存軸とbinary64 hexで一致すること。欠測・不一致・別candidate/cohort・二重半減・原QPY由来inputの混入でSTOPし、追加callや再試行は行わない。成功してもBのcross-runtime compiler費用同等性、過去worker実load、全状態operator保証、代替契約採用、COMPAREの実行承認を与えない。

## テスト・保全・終了

[人工284件PASS](test_results.json)、[ログ](technical_tests.txt)。新36＋既存248。原SHA/SOURCE/input/prep、QPY入力混入、6件のtime/block順序、角度/符号/phase/wire、cohort/軸/元signed schedule、契約変更、独立性不足、未承認実科学入口、予算超過、全6receipt確定→134比較の順序を検査した。既存primitiveへの実引数をmockで記録し、追加の`/2`がないことも確認した。初回の静的PASSを[履歴](attempts/schedule_binding_before_final/NOTE.json)へ保全し、元scheduleとの照合を強化した最終SOURCEと新36で確認した。旧PASSと新PASSを二重加算していない。Codex主担当の技術検査であり、独立科学レビューではない。

[保全監査](preservation_audit.json)は16,804保護pathの前後変化0。元SOURCE/input/runtime/結果/旧grant/ledger/marker/STOPは不変。[終了照合](final_verification.json)と[技術inventory](file_inventory.json)で旧manifest/既存成果物と新SOURCE・report・metadataを確認する。private原本・seed・raw/event/回路payload・内部絶対pathの新公開なし。共有venv/OS/scheduler/他job/他worktreeに書き込まず、利用可能CPU64と空きRAMを読取確認し、恣意的な低いworker/thread上限は追加していない。歴史的固定compiler設定は変更しない。

担当ユーザー：今回のA2と修正済み時間定義・exact6件の案を確認し、必要なら限定科学実行を別途明示承認する。担当Codex：別作業で既存関数を変えないactivation/封印・unused grant/rootの具体化を行う。新しい同条件の独立保存receiptが提供されれば、その独立性とSHAを確認して実評価を省略する。B作業・代替契約の正式採用・COMPARE本実行は別の判断/承認であり、自動進行しない。

[mandatory STOP](STOP.json)。A残条件は既存証拠だけでは未解決、同条件の別receiptがない限り独立primitive評価の追加witnessが必要という結論で終了する。実H2 primitive評価/新科学計算/回路生成/compile/simulation/COMPARE/会計/順位/grant/詳細科学レビュー/最終evidence freeze/pushは0。専用branchへのローカルcommitのみ。
