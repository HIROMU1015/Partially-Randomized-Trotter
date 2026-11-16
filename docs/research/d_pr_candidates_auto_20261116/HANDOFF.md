# B2/C：両PR方式の自動準備と探索復帰

PR-Pauli LD4とPR-DF LD6について、研究者が候補を手で準備せず、通常B2探索→不足取得→intake→候補固有価格→同じ探索のshot/Q3→条件付き費用→Cの通常定期同期まで実データで通った。今回は各方式の最初のREADY候補1件・Q3各1件の有限検収。最安点更新、全探索完了、正式E[C]/CI/Gを意味しない。

## 固定APIと使い方

現API SOURCE：`98b40e4390ec23f581eea10bdbb33f2caa2870dd`。公開branch：`codex/d2-pr-df-pauli-autoprepare-20261116`。通常入口は `scripts/run_rpe_method_optimizer.py --problem … --round … --run-dir …`、組込APIは `trottertracks.pr_candidate_completion.Completion.complete` / `install`。新たな候補選択器・配分選択器を作っていない。[API.md](API.md)に入力/出力、SOURCE依存view、直接呼出例を記した。

B2は採択済みのH/state/精度contract、L_D、公式、要求Δ/実schedule、κ絶対値を渡す。MODEL/DIRECT/NEEDSと外挿欄を使い、未取得価格を0や不適合へ変えない。物理keyは方式固有のhead/tail/scalar/law/pieces/actual scheduleを含む。κ宣言aliasは元B2登録を使用し、Q3は同じExecutor/job identity/累積予算で再利用する。

固定検収command（完了後は再利用し新規科学callを出さない）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-candidate-autoprepare-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B \
  scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round docs/research/d_pr_candidates_auto_20261116/ROUND_RUN02.json \
  --run-dir docs/research/d_pr_candidates_auto_20261116/run02
```

`ROUND_RUN02.json` は実取得SOURCE1691…と元A checkoutを保存した再開用正本。今後の探索には `ROUND_NEXT_EXAMPLE.json` を基に別round/runを作り、SOURCE・探索範囲・予算を固定する。これは例であり、D2はその後続runを起動していない。B2の稼働中run/worktree/checkpointへ変更を入れる必要はない。

## Cが読む固定結果

- 機械可読まとめ：[RESULT.json](RESULT.json)（configuration・schedule・26軸価格・配分・exact MSE・条件付き費用・元receiptへの入口）
- runtime/cohort/seed・各工程：[run02/completion/STATE.json](run02/completion/STATE.json)。DF取得の元directoryはrun01、実行SOURCE194e…を保持。
- C用join列挙：[run02/completion/C_EXPORTS.json](run02/completion/C_EXPORTS.json)
- Pauli join：`f6477f5009d52c50836036f4eb6c5835e1fa02ef:docs/research/d_pr_candidates_auto_20261116/run02/completion/comparisons/PR_PAULI/7a94374fe4d2d0aaee49/C_RESULT_JOIN.json`
- DF join：`1e107cf4de2186ee7df324412d18be9644a87d91:docs/research/d_pr_candidates_auto_20261116/run02/completion/comparisons/PR_DF/3b9c916b4f882df82b2d/C_RESULT_JOIN.json`
- C通常同期：`1d44d07397031547e402171faae268a640a2807b:docs/research/h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json`。両join INTEGRATED、26軸/配分/MSE/費用一致。[C_DAEMON_ACCEPTANCE.json](C_DAEMON_ACCEPTANCE.json)にactual consumer SOURCE76474c3d…と公開SHAを保存。

|実候補|exact条件付きモデル費用RZ|Q3|備考|
|---|---:|---|---|
|PR-Pauli LD4|5081576176/7|適合|A実step較正＋元audit、q/R/event-step外挿を保持|
|PR-DF LD6|20341620657/14|適合|候補自身のα+βq、m3独立記述holdout、m7以後外挿|

DFの最初のOUTCOMEはprocess receipt形式の接続不備でUNKNOWNだった。旧raw/OUTCOMEは固定保存し、元supervisorの実exitcode0・identity・closure・AST guardをcollectのみで再照合して、別COLLECTED_OUTCOMEで採用した。再Q3なし。元P2で未実施のfull-wrapper statevector検証は新たに保証しない。

## 再利用・不足・停止

開始前にB2 d7a976a62、A 02e7cc6a1、Bの通常intake、D2 326feadd0 / PriceRules c1f95e080、元DF LD7取得・価格と現在の関連jobを照合した。PauliはB HANDOFF13326577…/SOURCE33e324b…へ統一し、A entry6e4b6bd6aのplan/execute/assemble全てにtail descriptorを渡した。LD6を手動で起動していない。

同じ分割のDF descriptor、保存signal round、同じcohort/seedの個別axis receipt、元Q3を再利用する。DF partial価格では、4 indexの検証済み保存cohortをINPUTS固定前にmetadata順で採用し、X済なら同pathのYだけ取得する。旧/新SOURCEはPATH_INDEXの個別locatorへ保持し、両軸のevent/seed対応を照合する。mean-onlyや独立性不明の保存物は保全して影響軸をHOLD。保存holdoutを新較正へ移さない。

未対応公式/pieces/K、複数actual-stepの価格登録、rank不足、不明なSOURCE/receiptは具体的NEEDS/HELDとし、他の候補を進める。新registry証拠による価格HOLDの解消はmetadata再照会だけで行う。既失敗・曖昧な科学leafの無断retry、seed差替え、guard緩和は行わない。

技術STOP履歴、検収範囲訂正前の不要な既存ready Q3 3件（全FAIL）も[REPORT](REPORT.md)とRESULTに保全した。総Q3は5件で、新候補の検収2件と区別する。新規signal26 round・normal12 path・wrapper24 compile・DF fit2。完了runの再開で新規science0は[REPLAY_CHECK.json](REPLAY_CHECK.json)。人工52件と後続登録metadata検査はTESTS/NO_DROP_REGISTRY_CHECKに記録する。

ロードマップ差分：方式別の人手による候補準備・HANDOFF転送を、既存の科学契約と価格policyの範囲で同じ探索へ戻す実経路が揃った。候補/κ/shot選択と探索高速化はB2の担当のまま。候補固有モデルの外挿と理想law/実samplerの区別は残り、厳密な母期待費用・CI・全L_D最適性の証拠にはしない。ロードマップ本体は統合担当が更新する。

## 追記：LD7/8/9の登録済み新P0 identity

Aの最新HANDOFF `d3a1a3d359014e7d0bd95e648b4174cfe1598c9e`、準備API/登録表 `9f00ddd86a44b5d1443f29b5f02dbf1db4ef9ab8`、builder `25a08afafa32af2ca52bca4f63233c38b3af0e6e` を反映した。今後の `settings.pauli_workspace` は新登録表を含む所有checkout `d2-pauli-auto-nodrop-evidence-20261116` を使う。LD7/8/9は通常入口へ `--p0-builder no-drop` を渡し、`REUSED_CONFIGURATION` と `P0_NO_INTERMEDIATE_DROP_V1` の保存辞書・基準構成を使う。LD8を含めtail descriptorをplan/execute/assembleへ渡す。

旧辞書のSTOPは旧SOURCE/identityの履歴として残す。新辞書のidentity・λ_P・lawを別物として登録し、旧辞書の価格係数を黙って移送しない。新P0を再生成せず、分割は同じまま公式/幅/κを変える際も辞書を共有する。固定参照とmetadata検収は[NO_DROP_REGISTRY_CHECK.json](NO_DROP_REGISTRY_CHECK.json)。3分割の信号はここでは取得せず、実候補選択と不足round取得は通常B2探索に委ねる。完了済みRUN02のA checkoutとSOURCEは変更していない。
