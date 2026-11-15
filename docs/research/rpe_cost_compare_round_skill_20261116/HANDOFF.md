# A・B2・統合担当：rpe-cost-compare探索round版の受入用HANDOFF

## 使用したcommit

| 対象 | 固定commit |
|---|---|
| B2公開成果/HANDOFF | `a4b4d0e1f2110e63ae9a1dc6a89e1699f95dc0eb` |
| B2実行SOURCE | `a2e2ef50a108d6f34089a99bd00f11a7085dd9c9` |
| 今回起動/再開を確認したSkill手順 | `3262c0b4a2610a60c6b0bfb2c4ca010beda2d271` |
| 更新前Skill | `eb14ca70e4de61e0fe1a8556f75571ee5e56fa18` |

今回のSkillと検証記録は既存専用branch `codex/rpe-cost-compare-skill-20261116`に公開する。3262c0b4aには新しいSkill・B2実物の入口説明・予算0例があり、公開branchの後続commitはこのREPORT/証拠の追補。Aは公開branchのSkillをそのまま読み、実行SOURCE側の共通programを起動できる。Skillを読み込む場所とprogram checkoutは別でもよい。

## Aが実際にSkillを呼ぶ方法

Skill側checkoutの `.agents/skills/rpe-cost-compare/SKILL.md` が見える状態で、例えば次を指示する（pathはA自身のdirectoryにする）。

> $rpe-cost-compare 保存H4/1.00 Å/STO-3G/DF10、sector(2,2)、要求energy RMSE 0.00159360010199040 Haを比較して。B2実行SOURCE a2e2ef50aの独立checkoutを使い、各方式の新規Q3予算0の新roundを作成して起動する。同じroundを一度再開し、最安点・次候補・停止理由を報告して。新規科学計算は0件。

利用者へ個別のLD・shot列を質問する工程はない。採択済みproblemとroundの方式/予算/範囲を使い、programに候補・配分・次候補を選ばせる。

## 実物のCLI

B2実行SOURCEをcheckoutした自分のworktreeで実行する。公開a4b4d0e1fは文書/round例を含み、実行codeはa2e2ef50aと同じ。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python \
-B scripts/run_rpe_method_optimizer.py \
--problem configs/rpe_method_optimizer/h4_df10_problem.json \
--round /absolute/path/to/round.json --run-dir /absolute/path/to/run
```

**再開も完全に同じコマンド・同じround file/round_id・同じrun directory。** 新しいdirectoryでの起動を「再開」と呼ばない。明示許可のない予算増額やScience workerのretryはしない。許可された増額は方式別の総上限として同じround fileに反映し、既起動分はprogramが消費済みとして数える。

Skill内の`references/h4-round-budget0.json`、またはこのディレクトリの`ROUND_BUDGET0.json`をcopyすれば予算0で確認できる。B2実行SOURCE a2e2には公開文書commit a4b4側のround例がまだ無いので、Skill同梱例を使えるようにした。`approved_by`は任意の記録欄であって承認証明ではなく、例には付けていない。

## 検証済み範囲と出力

- 新規Q3予算0で、Skill手順/直接CLI/同一round再開の全4方式のbest・pending_request・stop_reasonがexact一致。
- p7はYoshida4・32h/5、372,035,410 RZ。次候補は同構成のm12各6→4、329,694,342 RZ。
- 全方式停止理由はNEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET。新規Q3・signal・sampling・compile・fit各0、jobのLAUNCHなし。再開後INVOCATIONSは2行。
- `ROUND_RESULT.json.methods`が主結果。`best`/`pending_request`/`budget`/`held`/`resume_records`を読む。`NEEDS.json`、`PER_METHOD.csv`、`INVOCATIONS.jsonl`を併読する。
- 別agentがSkill手順から予算0の新round/同じround再開を実行し、全方式のmethodsは直接CLIと一致した。科学的レビューやAの受入ではない。詳細はverification/forward/OBSERVATIONS.json。
- 正予算の実科学Q3や実行中/死んだ実jobの再開は今回Bが実行していない。B2公開人工受入を参照し、Aが必要な範囲で残る受入を行う。Aの受入をBの自検で完了扱いにしない。

[VERIFICATION.json](VERIFICATION.json)にこの検査で実行した絶対path付きコマンドと両SOURCE、両結果を保存。旧Skillの初版検証と旧科学結果は保全した。新しい科学計算へは自動進行しない。
