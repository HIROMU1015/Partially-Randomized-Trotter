# 最小実装・全入力correctness

## 実装と既存SOURCE

基点62c7b578の既存科学codeは無変更。新規は `track_d_head_tail_artificial.py` と専用runnerだけ。StrongBuilder.transition/actual basis/vacuum_phase、C3 mixed_head/physical_segment/prepare_steps、既存full-frame Provider/control_one/reference_gadget、C7-2 context/event/exterior lift、既存cut人工raw_checks/post_checks、既存P0/MPFR160を呼ぶ。

C7-2のmake_casesは54回路を無条件に構築するため呼ばず、既存contextとeventからq2_mixedだけを作る薄い入口を追加した。独立targetは既存exterior lift/occupation diagonalとC7-2のordered積規約に従う。Gaussian合成器・Hamiltonian builder・sampler・simulator・checkerの別実装はない。

head出口G_Hとtail入口G_T†は、既存emitterの実native命令・rotation・wire順・global phaseを照合した意味上のspanで特定する。IDや近似一致でsearch/deleteしない。異frameは既存transitionによるG_T†G_H、同frameはactual同回路とinverseをidentityへ置換する。全体のdeclared phaseから取り除くG_H/G_T†の寄与を差し引き、新しい橋のphaseを一回composeする。逆向き・別wire・未知gauge・間のsystem gateは拒否する。

Strongは両stepのhead→tailに同じ規則を適用。Referenceは既存physical_segmentで作られたB/Aへ同じ規則を適用し、**融合後A.inverse()**を使ってgadgetを再接続。tail→head/内部tailの異frame/step追加融合/global cut/CSWAP/readoutは不変。q2のStrong既存step融合1、Reference内部0・cut損失1も保持する。

## 検査と結果

| 範囲 | 結果 |
|---|---|
| 局所の異frame・同frameの全Fock unitary/exterior/vacuum、π/7 witness | positive7 PASS、最大成分差1.704e−16 |
| N1：G_H†G_Tを使う逆向き橋 | 不一致1.407を検出 |
| N2：analytic gauge欠落、局所π/7 witness | 不一致0.4450を検出。whole-pathへwitnessを混入しない |
| N3：ordinary tail→directional | branch不一致0.3922を検出 |
| N4：同IDのrotation/wire/gauge不一致 | 3種類とも拒否 |
| N5：Gaussian間の非可換X | 隣接span照合で拒否 |
| 既存contextのbasis/exterior確認 | 3件PASS |
| baseline/fused whole-path raw | 41比較PASS、最大成分差2.665e−15、最大Frobenius4.797e−15 |
| 全8 wrapperのcompiled対raw full-unitary phase | 8 PASS、最大成分差2.891e−14 |
| post X/Y/coherence/reference leakage | 16比較PASS、最大Frobenius7.179e−15 |

全4 system入力、全reference出力成分を小規模Operatorで確認した。AB=V、actual A†、A/B vacuum ray、scalar/extracted identity/analytic χ、Strong両branch/coherence、X/Y誘導observable、reference leakage、全unitary位相を保持した。component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなし。不一致後の閾値緩和・fixture/cut/mode変更は0。

S/Rの宣言global_phase欄はP0のgate分解に応じて変わり得るが、全unitaryをphase込みで照合している。ヘッダーの角度だけの等値を要求したり、phase fitで差を隠したりしていない。normalization/attenuationをunitaryへ乗算せず、元H groundや状態準備を計算していない。

これは有限人工条件のbinary64検証で、独立レビュー/immutable CI/roundoff enclosure/全H4・全law保証ではない。

## 技術修正・実行履歴

初回SOURCE `da975efa9c11fcc41602d6acb56e8fd85f2ff514` のrun01はlocal positive7/negative7 PASS後、保存時にSimpleNamespaceのeventへto_dictを呼んだAttributeErrorで停止した。旧SOURCE/partial/FAILURE/STOPは無変更で保存。raw checks/compile/postは0。

ユーザーの同scope技術修正許可に基づき、保存表現を実際のfieldへ修正したSOURCE `af6fc07f35dce88d1833163f01781478e5919c7e` で未完了処理を継続。局所PASSはhash・SOURCE/input・失敗型を照合して再利用し、local検査を再実行していない。最初のprocessのraw objectは保存前に失われたため、同じ4構成×X/Yのrawを再構成した。contextに付属するbasis検査も再構成に伴う。同一caseの構成履歴は2回、unique whole pathは1であり、別入力の追加や有利な結果を探すretryではない。

run02ではtail emission2、whole-path raw check2bundle、8 wrapper結果、新P0 4・旧P0再利用4、post8、cache hit0。初回と合算しても新compile4。保全前後照合PASS、旧baseline原本/索引/STOP・今回SOURCE/inputは一致。q64や他worktree/共有環境操作0。

詳細：[LOCAL_CHECKS](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/LOCAL_CHECKS.json)／[RAW_CHECKS](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/RAW_CHECKS.json)／[POST_CHECKS](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/POST_CHECKS.json)／[CONTINUATION](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/CONTINUATION.json)／[INTERFACES](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/INTERFACES.json)。
