# A2現行経路：終了・将来の再開条件

入口は [ユーザー採択済み方針](decision_record.md)、[machine状態](decision.json)、[成果・限界](achievements_and_limits.md)、[固定証拠索引](evidence_index.json)、[レビュー原本](inputs/qc_a2_phase_b_results_scientific_review_20261018.md)。

## 現在地

**現行の実用化・十分step改善経路はPhase Bで区切り、追加投資・探索を行わない。** A2全体は `NOT_ADOPTED` のまま、A1 `CLOSED`、K1-F未承認・未実行。数学保証の撤回・A2一般のNO-GO・研究全体の全面終了ではない。①②③や他の④研究の状態は変えない。

Phase Bの初回one-shotは完了しており、同じSOURCE/契約/inputの再実行は不要。canonical checkout `/tmp/prt-qc-a2-pilot-phase-a-20261018` の `artifacts/qc_a2_pilot_execution/c3b13d5007636f83def99bb04bbb00d31dbbdeaff9a6474a4a4319f9ea403c35/` は**消費済みclaimを含む保全対象**。この場所の原本13filesとGit上の公開copyを保持する。claim削除・解除、別run_id/別checkout/契約再serializationによる迂回をしない。

## 自動実施しないこと

- 別分子・geometry・rank・PF順・trial、d/α/算術精度・反復数の走査。
- 保存値からの寄与診断・追加ランキング、再計算による成功例探し。
- 改善証拠・新数学保証・構造付き有限剰余の探索、SOURCE修正・再実装。
- SPEC-08、QPE/native、state preparation、費用目的変更、論文化・追加priority調査。
- 新しい重要科学レビュー、authorization/grant/marker/claim発行。

直前に候補として挙がったcap・trial・K_SOS改善や限定診断を、今回の終了記録から実施許可へ読み替えない。元レビューの「採択待ち」、実行報告の「科学レビュー開始待ち」は各時点の履歴であり、最新方針は本記録を参照する。同じ重要レビューを再実施する必要はない。

## 再検討に必要な新しい根拠

再検討の材料になり得るのは、合法に鋭いΛを認証する具体構成、相対係数と適用域を改善する根拠、構造・相殺を保持する有限剰余、または有用な入力classと証拠取得費用を説明できる具体的成果である。これらは**必要になり得る根拠の例であり、探索課題の発注・採択ではない**。現時点でいずれも得られたとはしない。

同じ情報を強い既知対照にも公開し、証拠生成・独立検証・state/readout/native・償却などの不足費用を0にしない。「算術精度を増やす」「例を追加する」「新規性がまだ否定されない」だけでは自動再開しない。cheap certificateを新目的にする場合も、cheapな既知対照と共通費用軸を含む別方針を先に定める。

再開する場合の担当と承認は次の順序とする。

1. **ユーザー**：新しい具体的根拠を踏まえ、対象・目的・範囲を指定して再開方針を別途承認する。
2. 必要な重要科学判断がある場合、**ユーザーの開始承認後にGPT**が新証拠との差分だけをレビューする。
3. 承認された範囲でのみ、**Codex**が必要な新version・契約・identityと旧成果保全を準備する。旧SOURCE/C/claimは上書きしない。
4. **ユーザー**が科学実行を別途明示承認した場合だけ、**Codex**が新しい固定契約に従って実行する。

## 完了状態

今回の作業は決定記録・原本保存・索引・保全・公開のみ。新科学計算、保存科学値再解析、証明・詳細科学レビュー、SOURCE変更、authorization/grant/marker/claim作成・消費は全て0件。**公開・保全確認後mandatory STOP。現時点の追加作業は不要であり、再開への自動移行を行わない。**
