# Full PF/S2/2h Q3一件：通常入口の通し検証

対象9e96ed75…、公開D352539経路、科学実行SOURCE90a52a211f6acc4d7b01e1017572c15224ac25de。実Q3一件が元モデルの閾値へ適合した。MSE上界約6.563770861285207e−7 Ha²、元τ約2.539561285063813e−6、条件付き費用956,592,146RZ。新信号/sampling/build/compile/fitは0。

通常探索器がB_PRICEから選んだX/Y各列 `[59,55,51,47,43,39,35,31,27,23,19,15,11]` を使い、研究者のshot/seed入力を追加していない。対象はこのconfigurationだけ。job`22bbe4a8d5aee8bfa9b8`、allocation`RPE_OPT_22bbe4a8d5aee8bfa9b8`。

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
 /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B scripts/run_rpe_method_optimizer.py \
 --problem configs/rpe_method_optimizer/h4_df10_problem.json \
 --round docs/research/d_full_pf_s2_2h_q3_e2e_20261116/ROUND.json \
 --run-dir docs/research/d_full_pf_s2_2h_q3_e2e_20261116/run01
```

この同じCLIを同じ総予算1で再開した。追加起動0、累積1、same best/停止理由、原jobfile不変をRESUME_CHECKへ記録した。結果はQ3_BUDGET_EXHAUSTEDで、未実行次要求に進んでいない。B2統合版検収は別作業のまま。

C向け正本はC_RESULT_JOIN.json/ SUMMARY.json、参照入力はC_INPUTS.json、26項価格はPRICED_AXES.json。原CASE_INPUTS・RESULT/API_RETURNは55c174a3f04f560a57bd30abd19b1197b2edea67のrun01/jobs/FULL_PF/22bbe…以下。generic PF consumerを再利用し、C変更や手動同期を要しない。source-boundなledger→input viewをmetadata commitで固定しただけで、科学callを追加していない。

Cの実定期反映はC_PERIODIC_SYNC_CHECK.jsonを参照する。自前C_CONSUMER_CHECKと区別し、configuration・配分・MSE判定・条件付き価格の結合を確認する。FAILの人工fixtureはINFEASIBLEかつ最安/達成費用へ不採用となることを確認済み。本実測はPASS。

価格はdirect10＋条件付きモデル16。未知長q誤差/CIは未取得で、G/winner/十分N/物理的global RMSEを認定しない。ロードマップには「新Full S2/2hのQ3待ちを一配分のモデル適合・条件付き費用付きへ更新、通常入口と予算1再開を確認」と伝える。本体編集は統合担当に任せる。旧raw/SOURCE/STOPを保全し、追加配分/候補へ自動進行しない。

実定期同期PASS：C公開5fbe4782725ae3ba36e2b431d199e39ae1a29b7b、CATALOG.candidates[57]。同じ配分・exact MSE・条件付き956,592,146RZ、FEASIBLE。14:10:01〜14:10:25 UTCの通常周期で反映し、手動再同期はしていない。これでQ3一件→通常探索へ取込→予算1再開追加0→C通常同期まで完了。
