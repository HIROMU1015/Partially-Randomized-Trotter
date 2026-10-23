# H4 C run01引継ぎ：reference欠陥のため科学受理不可

branch `codex/non-df-h4-c-evidence-run01-20261023`、worktree `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-production-20261023`。実行SOURCE49c18ccc35f14041df45bfeff455ba5e67543182、基点c4497d8b49a3d796513df439dd025c72fb964a21。結果commitはこのbranchのgit logで確認する。push未実施。

## 今回の終了状態

固定40軸の候補数値証拠生成を、新grant8af94df114e6485c7a5f10d12e77352347aa21b2e9525647a8ad8ebc988f4554で一回実行した。grantは消費済み・予約返却なし・再使用禁止。Pauli PF/PR各10軸rawは保存したが、reference実装欠陥のため不受理。DFは同じ欠陥によりwitness経由で停止し、最終tierなし。run01のSTOPはFAILED_NO_RETRY。

両bridge exit0、Pauli worker exit0、DF worker exit−15、supervisor exit1、全native wait/reap証拠を保存した。DFの停止は実装欠陥によるintegrity STOP。計算時間や共有CPU/PSIを理由に停止していない。元H4のnative欠落は未回収、原因UNKNOWNを維持する。

## 根拠・修正案

reference=signal(state,expm(H))が右側stateを適用していない。helperも行列outを受け入れ、first-columnの成分をreferenceとしていた。人工|1⟩・Iで期待1に対して0となる1 FAILを保存した。前回の最初のbasis vectorに偏った人工fixtureでは見逃す。

最小patchはdiagnosis01/proposed_reference_fix.patch、未適用。source_diagnosis.jsonに必要な非真空/複素重ね合わせ/shape拒否/Pauli・DF経路の追加回帰を列挙した。実行中/実行後の封印SOURCE・runtime・grant・結果・ledger・STOPは書き換えない。

## 次の担当・作業（未実行）

**ユーザー**：修正適用・追加人工回帰・新SOURCE/runtime seal、および必要なら別grantでの固定40軸再実行を承認する。修正前SOURCEでの実H4適用は再開しない。N/G再会計・正式certificate受理・公開pushは別承認。

**Codex（承認後）**：別の作業来歴でpatchを適用し、一般状態のreferenceとshape検査を人工例で検証する。新SOURCE・runtime/input manifest・契約を固定し、実H4再実行も承認された場合だけ未使用namespaceと新grantを結合して実行する。今回の予約・失敗・欠陥履歴は保持する。

既存のDF exact event範囲、literal compiled wrapperへの転送、IID仮定、正式certificate受理、旧native未証明は別問題であり、reference修正だけで解決済みにしない。

## 現在のSTOP

修正未適用、科学retry0、正式certificate受理0、新shot/G会計0、push0。旧20行N/G=null/40軸UNDETERMINEDを維持。保存証拠の監査・報告書・local commit後mandatory STOP。自動的な修正・再計算へ進まない。
