# B2/C向け：B0I pの自動選択と既存D補完への接続

通常入口から未評価pを選び、探索器が `pf_completion.selected` を埋める接続を実装・検証した。保存H4の初回選択は **p9／標準Yoshida4／要求幅32h/5**。これは取得順の提案であり、科学的な最適候補認定ではない。[REPORT](REPORT.md)、[保存計画要約](SAVED_PLAN_SUMMARY.json)、[人工の方式別受入](artificial_comparison/CHECKS.json)を正本とする。公開SHAはこのHANDOFFを含むcommitを用いる。

## 通常入口と入力

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B scripts/run_rpe_method_optimizer.py  --problem docs/research/b0i_prefix_auto_selection_20261117/PROBLEM.json  --round docs/research/b0i_prefix_auto_selection_20261117/SAVED_PLAN_ROUND.json  --run-dir docs/research/b0i_prefix_auto_selection_20261117/saved_plan/run01
```

同runの再開も同じコマンド。独立再現では新しいrun-dirとprovider output_root/registry_fileを使う。保存済みSELECTION/REQUEST/RESPONSEは同じrunで変更しない。SOURCE viewの絶対pathは実行環境に合わせるが、元Dのcommitを維持する。

利用者が指定するのはmethod、p/formula/幅の**範囲**、精度条件、科学予算と取得providerの固定環境である。次の候補やshot列は書かない。新default problemのB0I `p_values=0..9`、Full `p_values=[10]` は探索範囲であり、次のpの選択結果ではない。今回の保存計画は元problemを残し、roundのmethod_overridesで同じscopeを明示した。

- 候補metadata：`pf_candidate_space.configure(root,method,p,formula,delta_over_h)`。既存PF mapperからcanonical bodyを生成するだけで、H4のarray・preparation・回路は生成しない。
- 自動選択：通常CLI → `pf_candidate_selection.prepare` → `select`。入力にselectedがなければ、未準備候補から有限の1候補を選ぶ。
- D受渡し：`pf_optimizer_connection.complete_selected`。探索器が選択後に `selected=[{configuration_body:…}]` を作り、D36adの `scripts/prepare_pf_candidate.py --request … --output … --mode plan` へ渡す。独立provider viewは `36ad751d4ebf0ec160532615da6425b268a0c18a`。実装を複製して科学callerを増やしていない。
- 復帰：既存registryをO.prepareへ結合し、同じcanonical ID/p/formula/幅でreadyになった候補を元のshot探索に戻す。予算、caps、generator、Q3 routeは元B2のものを使用する。
- `pf_completion.mode=plan` は科学0件。entry未設定なら `PLANNED_D_ENTRY_REQUIRED` と不足を返す。`complete` は元の科学実行条件・有限SOURCE/input/範囲が実際に許可された場合だけ使用する。今回の公開は実H4取得の許可ではない。

## 固定の追跡先

|対象|本公開commit内のlocator|
|---|---|
|利用者入力|`PROBLEM.json`, `SAVED_PLAN_ROUND.json`（selected/shot列なし）|
|探索器が選んだ根拠・全候補scope|`saved_plan/run01/pf_auto/SELECTION.json`|
|探索器が埋めたD要求|`saved_plan/run01/pf_auto/pf_completion/REQUEST.json`|
|D responseと不足計画|`saved_plan/run01/pf_auto/pf_completion/RESPONSE.json`, `saved_plan/run01/pf_auto/D_PLAN.json`|
|同run再開の不変性|`SAVED_PLAN_RESUME_CHECK.json`, `saved_plan/run01/INVOCATIONS.jsonl`|
|人工応答→自動shot→人工Q3→方式別結果|`artificial_comparison/CHECKS.json`, `FIRST_RESULT.json`, `RESUMED_RESULT.json`及び同directoryのrun/jobs|
|公開後続・既知B所有jobの照合|`PREEXISTING_WORK.json`|
|テスト・再現SOURCE|`VALIDATION.json`, `SOURCE.json`|

この表のpathは `docs/research/b0i_prefix_auto_selection_20261117/` からの相対path。人工結果は本番catalogへ登録しない。Cの科学consumerはartificial registry/ROUND_RESULTを拒否する。人工単価・人工MSEを新しいH4価格やQ3として採用しない。

## D/C/B2に残る範囲

保存H4のp9選択は既存D planで保存6 mean・新規候補7 mean（m3,m6〜m11）、同prefixの価格不足を返す。価格の元有限案は最大16 wrapper、early直接値・較正m5/m6・holdout m8を含むが、今回は取得も価格規則の採択も行っていない。モデル誤差上界・許容差は未確定のまま。新計算をする場合は選択された物理round/軸について、旧ID・元receipt・compiler/lowering・保存/進行中の同等jobを§10.5で差分確認する。調査未了を「未実施」に置換せず、未解決分は保留する。

p7の保存Q3適合点と価格は保持。保存集合内最安372,035,410 RZはp7固定集合の記録であり、未評価pを含む全域最適とはしていない。評価済み集合にp9が新たに加わったわけではない。p10はFull PFの同canonical IDでありB0I側で二重評価しない。各pは自身のsignal/price・元Full DF-H targetでQ3を評価する。T1/T2は取得優先順位だけに使う。

既知B p7 runの未完了1件は元所有者へ残し、LAUNCHの同一keyを別runで投入しない。既存Dの比較可能な完了結果と、IN_PROGRESS/STALEの保留を維持した。関連branchへのmerge・main統合、他者rawのcollect/retryは行っていない。

高速化e839のconcurrent queue、HYBRID、DP prefix-cache、worker profileを維持した。今回の主変更は科学処理の高速化ではなく、不足候補を選んで既存取得へ渡すmetadata接続である。実H4の速度・資源は未測定。使用したcontrollerはPython3.12、元D plan providerはPython3.11.0rc1、prefix-cache回帰は元専用worker環境。実際の版はSOURCE/VALIDATION参照。

## ロードマップとSTOP

統合担当へは「B0I pを人手指定する接続欠落を解消、未評価pの補完→shot探索への復帰は人工確認済み」と反映する。p7以外の実信号・価格・Q3・費用比較は未検証のまま。B2のPR-Pauli専用driverをB0I入口に流用せず、通常方式共通CLIへ取り込める差分として公開した。

**mandatory STOP。新H4信号・sampling・build/compile・価格fit・実Q3は0件。次の科学実行は対象・SOURCE・有限範囲の別途明示承認後とする。**

実装固定SOURCE：`d6f383715e05cab3018e8b4212816db06b74fb21`。ローカル技術検査時のbase HEAD表示と最終実装SOURCEの対応は [SOURCE.json](SOURCE.json) を参照。独立CI・外部再現の結果とはしていない。
