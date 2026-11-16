# C・B2・統合担当：Full PF先行完了結果

- 固定optimizer：`53bb0a1d7b1da70644da1c22552224291205af78`。入力固定commit：`055e58da456a30350edef51173f3c4808b3e96cd`。
- Full PF原結果snapshot：`4b28d35da197c7a8dcde0a8d60a6a0f9ed917d75`。
- Full PF最安：M5・16h、480,258,064.00 RZ、original configuration `70f7b31744ecd87fe277511bad7d1fc7dace9b1a097c315729c566f706c75edd`。
- X/Y各軸：`[20, 25, 26, 24, 16, 20, 17, 15, 12, 11, 8, 7, 2]`。
- MSE区間：[2.5395541121403294e-06, 2.5395541123957259e-06] Ha²。exact τとの比較はraw/summaryの有理数値が正本。
- 244新規Q3、保存11件再利用。停止NO_IMPROVING_PROPOSAL_INPUTS_MISSING。入力ready3/不足17。STEP_LIMIT/UNKNOWN停止ではない。

C向け固定locatorはこのdirectoryの`C_RESULT_JOIN.json`。各recordに元/total configuration・binding・allocation・target/law・原API/RESULT・26軸の元physical ID/単価SOURCE/N×C/合計を保存した。元price receiptのphysical IDをtotal IDで書き換えていない。新配分自身のNを使い、normalization・scalarを再乗算していない。Cでcatalog入力登録を行い、価格資格・十分N/global/Gのnullは保持する。

B2の再利用は `4b28d35da197c7a8dcde0a8d60a6a0f9ed917d75:docs/research/h4_pf_optimizer_search_20261116/run01/FULL_PF/ROUND_RESULT.json` を`--resume-from`へ渡す（本runを再起動しない）。244件のaccepted resume_recordsが含まれる。

PR比較はC417のLD5固定点とA b8bcdf97bの全26軸MODEL参照価格。最適化済みPR全体との比較ではない。現Full PFはその固定点より条件付き費用が11.34961389%低い。精度は保存signal・proxy・理想shotモデルに条件付く。物理的ground/実sampler移送・数値certificate・十分N・正式Gは認定しない。

**B0I p7は継続中**。その終了結果が揃ってから二方式の最終表を追補する。今回のFull PF終了後にそのmethodの追加探索は開始しない。ロードマップ差分：従来S2点からM5点への探索内最安更新、17構成の入力不足は残る。ロードマップ本体は統合担当が更新する。
