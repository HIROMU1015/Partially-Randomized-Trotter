# B2/C向け：選択された決定論PF入力の補完

B2は同じ通常入口 `scripts/run_rpe_method_optimizer.py` を使用する。Dは `scripts/prepare_pf_candidate.py`、`pf_evidence_preparation.py`、`pf_candidate_space.py`、`pf_optimizer_connection.py` を提供する。探索/shot生成/Q3の数値実装は既存B2/Bのものを再利用し、Dで複製しない。

正本は[C_RESULT_JOIN.json](C_RESULT_JOIN.json)、[元17構成の差分](COVERAGE_DELTA.csv)、[round/axis別の固定locator](ORIGINAL17_ASSETS.json)。新Full/S2/2hは13信号、直接10＋条件付き16価格で準備完了、Q3は未取得。既存p7/Y4/32h/5は完成済み保存入力だけを再登録した。

## 通常入口の例

固定SOURCEをcheckoutし、専用Python3.11.0rc1環境で次を使う。`ROUND_RETURN.json`はQ3予算0の実行済み受入確認で、同じrun-dirなら再取得しない。

```bash
PYTHONPATH=src /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B scripts/run_rpe_method_optimizer.py --problem configs/rpe_method_optimizer/h4_df10_problem.json --round docs/research/d_pf_on_demand_evidence_20261116/ROUND_RETURN.json --run-dir docs/research/d_pf_on_demand_evidence_20261116/optimizer_round02
```

新しくB2が選んだ候補はroundの`pf_completion.selected`へconfiguration Git locator、canonical body、又はmethod/p/formula/delta_over_hを渡す。入力例は[ROUND.json](ROUND.json)。`science_approved:true`は実際のユーザー許可がある場合だけ設定し、APIが承認を作る機構にはしない。取得前にSOURCEと有限PLANをcommitし、§10.5を差分照合する。未選択の17構成や全pを自動計算しない。

取得済みregistryのみを使う場合はroundの`pf_evidence_registry`を渡す。元の`seed_rules`からB_PRICE/A_SHAPE等を自動生成するので、人手seed列は不要。Q3 template中の元allocation又はunused1列はworker契約の容器で、探索のseedではなく実requestで置換される。Dの受入確認は新Q3予算0とし、実shot探索の継続はB2へ戻す。

registry `D_PF_EVIDENCE_REGISTRY_V1` はconfiguration_id別にmethod/p/formula/幅、configuration、13信号packet、26軸価格、Q3_inputs、total_configuration_id、binding_id、price_scopeを持つ。価格行は(m,axis,physical_id,unit_price_RZ,kind,price_evidence)で一意。sourceはcommit/path/selector。Sはcompiled価格へ昇格しない。direct/conditionalを保持し、エラー上界不明を0にしない。

## prefixと物理契約

B0I p0は一体項のみ、p1〜9は一体項＋fragment0..p-1、p10はFull10のcanonical alias。元H/state/snapshot、signed formula、exact T/q/actual幅、scalar、P0、control、measurement、compiler/loweringが一致しないものは共有しない。旧IDの同等性は元receipt/sidecarとphysical_contextで確認する。

B0I scalarは保存T1 prefixごとのtail identityと、保存Hから得るc_Hを元順序で加える。Fullはc_Hのみ。runtimeのbinary64完全一致を維持し、新しいtoleranceを入れない。新prefixのpreparation hashは別prefixのhashを借りず自身の値を記録する。人工検証済みと実preparation成功は区別する。

T1全p・T2 p7/8/9の42cellは候補のpriority用。truncation shiftだけでQ3不適合とはしない。p7以外のQ3比較は未実行で、次に選ばれたp自身の信号・価格で公式/要求幅/shotをB2が選び直す。

## B所有jobの引継ぎ

[B_OWNED_Q3.json](B_OWNED_Q3.json)の既存runを`executor.prior_run_dirs`へ渡す。完成312件は同じroute・入力・配分・評価契約で既存readerが受理する範囲だけ再利用する。未完了 `f999a51b2e1fd9e317e8` は今回STALEとして引き継いだ（[実確認](B_INHERITANCE_CHECK.json)）。別runのLAUNCHも所有権として扱い、IN_PROGRESS/STALEを新規投入に変換しない。DからBのrawへcollect・retry・書込みを行わない。

## Cの通常自動同期

最新C `1d44d07397031547e402171faae268a640a2807b` を基に、同じ`sync_saved_rpe_publications.py`/`Catalog.add`/`reconcile`へ新registry schemaを追加した。C_RESULT_JOINは既存のD-p7 branch discovery対象。prepared入力はQ3_WAITING、N/配分/費用/Q3判定=nullで受理され、正式Gへ昇格しない。

Cへの最小差分は `saved_rpe_cost_catalog.py` のschema dispatchと`pf_evidence_registry`、`sync_saved_rpe_publications.py` のschema許可（及びQ3新規0・bestなしのmetadata-only ROUND_RESULTをREFERENCE_ONLYとする条件）、`pf_optimizer_connection.py` のmetadata/source joinである。必要なbaseline physical-ID依存を保持する。C branchをDから書き換えてはいないため、現行Cジョブが既にこの差分を採用済みとは報告しない。本SOURCEの通常reconcileによる実保存入力確認は[C_NORMAL_SYNC_FINAL.json](C_NORMAL_SYNC_FINAL.json)に記録する。新SOURCEが未採用ならC側は旧schemaをPENDINGとするので、数値producerの再実行ではなくconsumer差分を採用する。

## 科学SOURCEと状態

- 新8 mean：03f619e4f以降の固定入力、公開73ee402cd。保存5を混合せず出所付きbinding、合計13。
- 係数/予測凍結：e093fe760、`selected/9e96.../FROZEN_MODEL.json`。m2/q2・m3/q4、軸別307+7783q。
- 正しいFull wrapper取得SOURCE：0b3d104e2、出力`holdout_remaining/run01`。raw公開2ab10ef1e、C guard公開a654e21b1。m5/q16 X/Y各124,835RZ。
- 旧pre-compile STOP：8e9878866と`selected/9e96.../STOP.json`。旧holdoutは空価格のまま保全。完了信号を再取得せず、未実行のwrapperだけ別SOURCEで完了した。
- 26軸価格：9c02f4728、`selected/9e96.../AXIS_PRICES.json`。holdout完全一致はq16の有限事実。モデル16軸の誤差上界はUNKNOWN。

各省略SHAは同branch上の固定commitであり、機械可読locatorはfull SHAを保持する。新prefix、別幅、未確認長qへ科学資格を移さない。今回の範囲外のQ3・N/G・winnerは未実行。

## 残る取得とロードマップ差分

元17構成は1構成の信号/条件付き価格入力を補完し、残り16構成はB2選択待ち。各候補の実compile件数は保存axisのavailabilityを調べてから決める。同実幅較正2点＋later holdoutがない幅、片軸だけの未解決証拠、SOURCE/scalar不一致は該当分を保留する。p0〜6/8/9はQ3費用比較未評価。

p7固定の保存最安条件付き点はC固定catalog引用372,035,410RZ。p探索で評価済みのprefix集合はまだ{7}で、その集合内の保存最安も同点。未評価prefixを含めた最適性はUNKNOWN。Fullの追加候補もQ3待ちである。ロードマップ本体の更新は統合担当へ渡す。追加候補・Q3は自動開始せず、本有限作業の公開でSTOPする。
