# M1-S shot・cost：SOURCE対応、resource根拠、技術引継ぎ

2026-10-17。非数値technical design。新実行SOURCEの開発・科学条件採択・科学実行の記録ではない。

## 既存SOURCEをどう利用できるか

以下はSOURCEを静的に読んだ対応であり、import／runner実行／数値テストはしていない。

|SOURCE|利用候補|流用してはいけない仮定／不足|
|---|---|---|
|`src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v2/analysis.py`|保存binary64をexact rationalとして受けるadapter、全grid、A/bのenclosure、fresh-IID/equal-axis policyの参照|旧実行を再起動しない。連続margin/境界であり絶対整数shot engineではない。旧α/p/capsを新defaultにしない|
|同`bounds.py`等のinterval層|将来のsin/log/enclosure設計の候補|新γ/profileとinteger ceilingについて独立した検証が必要。再利用しても全体の独立数値検証とは呼ばない|
|`src/trotterlib/rpe_resource_accounting.py`|round/axis/budgetの分離、Hoeffding式、fresh-IID状態の扱い|既存のunit-radius/eigenstate/alias-free条件をM1-Sへ転用しない。既存float ceilだけで整数境界を認証しない|
|`src/trotterlib/rpe_hadamard_interrogation.py`|X/Y測定wrapperのinterface・bit→±1・metadata要求|対象は別のcontrolled repeated DF partial-S2。M1-S wrapper未対応。相対time、符号、branch、identity phaseのadapterと意味論検証が必要|
|`src/trotterlib/df_rpe_hadamard_compiled_cost.py`|軸別全wrapper会計、compiler/cache/seed trace、古典sampleと量子shotの区別|state preparationは意図的に除外されている。既存costをM1-SのCやPにコピーしない。fresh-IID物理実験の証拠でもない|
|`src/trottertracks/algorithm_codesign/m1s_finite_signal_v2/pilot.py`|保存raw-H総bias、数値半径とmodel-difference ledger、元norm²規約|dense演算の診断SOURCE。HDのnative実装や実機costを実装済みとはしない|

旧libraryの`certified`等のscope付き状態を、M1-Sのplatform certificateへ拡張しない。今回は既存SOURCEの修正0、実行0。

## 最小SOURCE構成案

将来stage Sを選ぶ場合：

1. `spec/control`：固定契約・SOURCE・入力SHAと新しい別明示承認を照合。adoptionだけで実行しない。
2. `saved_adapter`：凍結raw信号または必要十分な保存intervalを読み、参照model・元norm²・全cell対応を照合。方式の値をreferenceから逆算しない。
3. `shot_bounds`：採択済みγ/profile・confidence・算術policyからS1の区間とceil端点を計算。整数をfloatへ丸めて比較しない。
4. `compare_and_ledger`：48対照を落とさず旧区分と新分類を別列で保管。G9絶対差の未解像性は維持。未評価・不成立・算術未解像を分ける。
5. `save/stop`：未評価テンプレート、失敗prefix、resource/入力/SOURCE/provenanceを保存。one-shot・retry禁止・mandatory STOPを契約どおりに実装。

今回これらの数値moduleは未実装。`m1s_shot_cost_design_v1.py`は式IDと未決fieldを持つ非数値specで、`request_execution`は無条件に拒否する。checkerはそのspec、machine候補、schema、metadata、ファイルidentityを確認する。数値engineがあるように見せるrunnerは作らない。

将来stage Cを選ぶ場合は別SOURCEが必要：M1-S control/HD/tail branch builder、符号・位相・積順序を確かめるsemantic reference、固定精度のHD/synthesis実装、native compiler adapter、X/Y/P/F component accounting、paired compiler policyとcall ledger。既存wrapperを二重にcontrolしない。H/Sdg/readoutを含めるscope、state preparation除外箇所、identity相対位相を明記する。native synthesisの前に新規数値correctness testも別承認されたscopeへ含める。

## 性能・resourceの候補を置く根拠

stage Sは保存scalar/intervalに限り、Hamiltonianを再生成しない構成。保存rawのbytes identityは`input_identity.json`参照。これらのinputは提案input16MiB内だが、将来の算術処理・peak RSSを保証するものではない。CPU1/GPUなし・threads1で十分かは新SOURCEの演算経路と採択γ評価数を固定してから確認する。

- memory2GiB、input/result/log各16MiBは**未採択技術候補**。無制限のfraction展開・出力や保存入力由来のpath traversalを許可しない。
- p=128 bits固定interval、integer bit cap65536を技術候補とする。旧SOURCE資産の再利用と有界整数処理の出発点で、今回数値correctnessを確認していない。整数境界を解像できなければ未解像としてSTOPし、結果を見て精度を上げない。γ点数JはD1のprofileに依存する。非zero方式行72なのでmargin評価上限は72J（全48対照で再利用）と置ける。sin/logの内部反復・integer演算call capは新SOURCEの静的演算経路から有限に導出する作業であり、科学判断の未決とは分ける。有限capsをSOURCEへ固定するまではRUN_READYにしない。
- CPU/wall time capは新契約既定なし。旧V1のCPU160/wall320を保全する。ツールのpoll timeoutと科学実行制限を混同しない。
- label/timeを跨ぐ重い新演算は予定しない。同じbias/trig/logのcache、固定inputの一回parse、独立cellのbatchは実装候補。速度向上率は測定していない。
- native stageは支配的workloadが別である。profile/回路数/HD実装が決まるまでworkers・aggregate memory・compile calls等は見積不能。将来承認実行前にCPU/GPU/空きmemoryを確認し、compilerとBLAS/OpenMP等のthreadsを調整する。

## 解決した技術論点

- γ*<π/3と整数shotの差は別条件。S2/S3で切り上げ・同数・算術未解像を区別できる。
- raw直接biasのmodel差再加算を禁止し、物理実装とtarget-energy橋渡しは別項にできる。
- 同じstate準備費でもshot数によって総額が変わる。Pを相殺せずC1へ入れる。
- formal branch演算数からnative価格を推測せず、既存DF/RTE cost SOURCEのscope不一致を特定した。
- 新scopeに旧α/p/time capを自動転用しない。新契約の未採択fieldをnullに固定し、guardを無条件closedとする。

## 未決事項と次の担当

未決はD1–D7の科学条件・実施範囲。input SHAやSOURCEの場所など技術的に調べられるものを科学判断として残していない。一方、実行SOURCEと有限call/bit capsは条件採択後の実装作業であり、今回の非数値Decision Packetを実行準備完了と呼ばない。

次はユーザーが共通γ条件とconfidence・Task A範囲・次期stageなどを判断する。採択された範囲でCodexが実行SOURCEと固定契約候補を完成させ、その正式採択と別の科学実施承認を受ける。native検証・多round RPEへの自動移行なし。今回新しいGPT詳細レビューは開始しない。

## 準備checkerの技術修正

初回metadata検査では、Gitの日本語pathのquoted表示をraw pathとして比較したこと、親snapshotのtracked-only statusとuntrackedを含むstatusを比較したこと、および文書の未採択表記検査で3件のFAILとなった。旧ファイルのbytes差は0だった。Git pathをNUL区切りで扱いstatusの取得条件を揃え、文書の未採択表記を明示して修正した。初回記録も別artifactに保存する。科学入力・結果・数値SOURCEの修正や実行は含まない。

## 監査の読み方

`static_validation.json`はローカルmetadata/AST/schema/guard/identity検査。`preservation_audit.json`は基点の旧非index Git blobs、親checkout追跡内容・index・HEAD、既存worktree HEAD、レビュー原本の保全。GitHub照合は別publication receiptに記録する。数値correctness、runtime、physical IID、platform仮定、科学的GOは検証していない。

最終metadata/静的・契約検査は116件PASS、FAIL0。旧非index7770 Git blobs、親追跡1284件・index・HEAD・tracked status、既存212 worktreeのHEAD、レビュー原本の保全はPASS。これは数値correctnessやCI実行の結果ではない。公開同一性はpublication receiptで別に確認する。
