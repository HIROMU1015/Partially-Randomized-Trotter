# リポジトリ専用Skill：rpe-cost-compare

**Skill作成済み・保存H4の4方式比較への接続確認済み。** 専用branchは`codex/rpe-cost-compare-skill-20261116`。Skill手順経由と直接CLIで、方式別の構成ID・全round X/Y整数配分・MSE上下界・exact費用・停止理由を含むRESULT.json全体が一致した。新規科学Q3、signal、sampling、compile、fitは各0件。

## 作成物と責任範囲

- [.agents/skills/rpe-cost-compare/SKILL.md](../../../.agents/skills/rpe-cost-compare/SKILL.md)：Hamiltonian/要求精度の受け取り、採択条件・既存実行予算の確認、共通programへの入力と結果の提示、利用例。
- [program-entry.md](../../../.agents/skills/rpe-cost-compare/references/program-entry.md)：公開B2実物の起動コマンド/引数、入力fieldと結果field、現在の機能・未対応の対応表。
- `agents/openai.yaml`：repository skillの表示・既定prompt。暗黙発見は既定のまま、global skill領域や他担当branchは変更していない。

LDや個別roundのshot数を利用者に順番に尋ねない。既存設定と許可で足りる事項はそのまま使い、不足する重要な科学条件/予算だけをまとめて確認する。候補生成、shot配分探索、価格算術、Q3応答、選択、budget/再開状態はB2へ委ね、Skill内に別実装や承認証明機構を作らなかった。

## B2の実物への接続

作成開始時はB2の実装/入口が未公開だったため説明・入力/出力手順を先に作成。その後B2の固定SOURCE **08b75526b561cbcf97ae58c7a75fefeb226911dd**が公開され、実物の`--help`とschemaに合わせた。

入口は`scripts/run_rpe_method_optimizer.py`、引数は`--problem`、`--output`、任意の`--ingest`。`--resume`等の架空の引数は記載していない。正確なコマンドとPython/cwdは[VERIFICATION.json](VERIFICATION.json)に保存した。公開H4入力は`configs/rpe_method_optimizer/h4_df10_problem.json`、search.max_new_q3_per_method=0。Skill側の手順SOURCEは`fbcf7f1d3`。

B2の作業worktreeには書き込まず、公開SOURCEの自分の独立検証worktreeを使用した。最初の直接実行はその新worktreeのsparse checkoutにconfigsが未展開でFileNotFoundErrorとなり、数値処理前に終了した。[元log](verification/direct/CONSOLE.log)を保全し、自分の検証worktreeだけに同じ公開SOURCEのconfigを展開して`direct02`で実行した。Skillの手順にもこの確認を反映した。B2 code/設定/入力データは変更していない。

## H4の接続検証

H4/1.00 Å/STO-3G/DF10、保存sector(2,2)、元H/state/snapshot、保存DF10 solver energy proxy、Z1+T1 total、MODEL_MSE_FILTER_ONLY、εE=0.00159360010199040 Ha、P=0。B2の公開問題入力を両経路で同じbytesのまま使った。

| 方式 | programが返した確認済み最安構成 | 条件付き費用 RZ | Q3状態 | 停止理由 |
|---|---|---:|---|---|
| FULL_PF | FULL_PF:formula=2nd,delta_over_h=16/7:11cf8ca6 | 559,457,386.00 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET |
| TRUNCATED_PF | TRUNCATED_PF:formula=2nd,delta_over_h=16/7:a6e55777 | 395,962,678.00 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET |
| PR_PAULI | PR_PAULI:L_D=5:4d6e2c9c | 541,743,905.57 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET |
| PR_DF | PR_DF:L_D=7:ce9c7f08 | 1,235,340,731.91 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET |

これらの値は公開共通programの保存結果出力であり、本Skillで新しい科学計算や最適性の認定を行ったものではない。安い次候補はpendingであり、有限の既評価範囲を超えた最適性は主張しない。価格もMODEL参照・条件付き閉包等の資格を引き継ぎ、正式E[C]・G・十分N・物理的RMSE保証に昇格しない。

- [直接実行のRESULT](verification/direct02/RESULT.json)と[Skill手順経由のRESULT](verification/skill/RESULT.json)：JSON全体一致。
- PER_METHOD.csvはbytes一致、NEEDS/REPLAY/OWN_PRICEの保存結果JSONも一致。
- 両主比較のmethods.*.new_callsは全4方式で0。traceのanswerはすべてSAVED。
- CLIには既存の人工FIXTURE/INGEST_DEMOも含まれる。各fixtureのcallbackは12回でARTIFICIALと明示され、科学Q3ではない。主結果へ混ぜていない。
- Skill構造検査`quick_validate.py` PASS。使用者がSkillを読む経路、実際のCLI/入力、sourceと出力を確認した作成者自身の技術検証であり、別AIや独立レビューによる確認ではない。UIの自動発見/再起動は試験していない。

## 残るB2側の機能・引渡し

この公開初版は保存比較・不足Q3要求の生成に対応する。新signal/価格取得/科学Q3 executorはまだなく、完全なjob checkpoint再開も未実装。`--ingest`は結果取り込みと次要求更新であり、workerの再開と同一視しない。方式menuもFULL_PF/TRUNCATED_PF/PR_PAULI/PR_DFの4方式で、Full-H pure RTE込みの5方式比較完了ではない。

Skillはこれらを未対応として返し、旧個別runnerへ迂回したり別ロジックを増やしたりしない。B2の新しい公開版で機能が揃ったら、Skillのprogram-entryをその実物へ更新し、保存H4で接続検査を繰り返せる。新しい科学実行は利用者が許可した条件・有限予算内に限る。

ロードマップへの影響：repository専用Skillと保存比較の入口が利用可能になった。B2側の新規科学処理/再開/方式coverageの不足は解消していない。ロードマップ本体は編集せず、統合担当へこの差分を渡す。既存科学SOURCE/raw/結果/旧STOPを保全した。
