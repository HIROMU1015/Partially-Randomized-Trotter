# 4方式通し受入：現行版を先行公開

現行Skill19252baf3→実行版a2e2ef50a→固定catalog417a73e73の保存比較は26/26 PASS。初回/同じrun再開とも4方式の構成・X/Y配分・MSE判定・exact条件付き費用が、その版のcatalog最安点と一致した。実job/科学callは0。

[REPORT](REPORT.md)、[方式別表](current_a2e2ef50a/RESULT_TABLE.md)、[受入機械記録](current_a2e2ef50a/ACCEPTANCE_RESULT.json)、[期待値正本](current_a2e2ef50a/EXPECTED_FROM_ACTUAL_CATALOG.json)。入力は方式4件・各予算0・initial=all。最安構成やshot seed列を入力に埋め込んでいない。

入口は公開Skillの手順を読んだCLI。Skill専用起動toolを使ったとは報告しない。追加runner `scripts/acceptance_four_method_skill.py` はA87d00ccbaのload_skill/Programを再利用し、4方式の同一round判定だけを追加した。人工6検査のSOURCEは`tests/check_acceptance_four_method_skill.py`。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <Skill記載python> -B scripts/run_rpe_method_optimizer.py \
 --problem configs/rpe_method_optimizer/h4_df10_problem.json \
 --round <本資料/current_a2e2ef50a/ROUND_INPUT.json> \
 --run-dir <本資料/current_a2e2ef50a/run>
```

実際に通った絶対command/cwd/runtime・終了codeはACCEPTANCE_RESULT.commands、初回原結果はfirst_snapshot、再開後はrun、累積はINVOCATIONSを参照。再現は実行版a2e2ef50aの専用checkoutを使う。

C項目も通常周期で完了。公開8b22ca234のCATALOG.candidates[57]は新Full S2/2h、13信号・26価格、Q3待ち、達成費用null。p7二評価行は不変・重複なし。C_PERIODIC_SYNC_ACCEPTANCE/LOCATORSが固定selectorと実周期の証拠である。Dの自前reconcile testだけを周期反映の証拠としていない。

B2の統合更新版はこの先行公開時点で未公開。local ba35de77を代用しない。公開されたら別program checkout/別runで同じ検収を行う。新κ規則やD2準備callbackの動作は既存の人工fixtureを使い、実信号・sampling・compile・fit・Q3は0を守る。新旧catalogが違えば期待値を各版から再取得して差を記す。既存runを更新版の再開先へ流用しない。

A/B2/D/Cの作業branch、科学raw、稼働job、STOPに変更なし。ロードマップ本体は統合担当に任せ、現行受入と更新版未検収を分離して報告する。

## 更新版公開後の同じ検収

公開確認と別checkoutの作成後、今回のhelperをそのまま使う。`--program-source`はSkill19252が指すa2e2の書換えではなく、利用者が求めた更新版検収の別SOURCEである。

```bash
<Skill記載python> -B scripts/acceptance_four_method_skill.py \
 --program-root <新統合公開SHAの専用checkout> --program-source <新統合公開fullSHA> \
 --python <Skill記載python> --phase updated \
 --output docs/research/four_method_skill_acceptance_20261116/updated_<固定SHA>
```

各版の固定catalogから期待値を取り直す。新κ規則のround overrideが必要なら、新公開HANDOFFにある値だけを`--search-overrides <固定JSON>`へ渡して別記する。実D2補完を有効化して予算0を迂回しない。saved-only入力を新入口が拒否する場合もFAIL/UNKNOWNとして原入力と到達工程を保存し、実科学承認へ勝手に変更しない。D2への不足引渡しは、新版の保存/人工fixtureで呼出し引数と保存候補再利用を確認する。

更新版未公開は`UPDATED_VERSION_STATUS.json`の観測時点の状態であり、現行PASSを更新版PASSと読み替えない。追加の科学実行は今回の全段階で許可されていない。
