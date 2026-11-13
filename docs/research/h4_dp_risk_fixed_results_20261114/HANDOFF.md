# 固定H4 DP：SOURCE・結果の読み方と次の境界

2026-11-14 JST。元kernel `06db9aefec13bb5c5be6d7c0f81f29b3a10b929c`、範囲`4139bed3c3364f4cf2afa3451d4a956961319060`、実行caller/入力`ce0517de6e7b58cf4bf4c69f99f998d86099c406`。**2件とも完了・検算PASS**。数値・完了状態の正本は[REPORT](REPORT.md)、[COMPLETION](run01/COMPLETION.json)と各RESULTとする。

採用モデルMSE上界はLD1約9.039527981195e−4、LD2約9.032441317634e−4 Ha²。旧上界から約99.62%縮小したが、両上界の約99.9477%は既知未解決loss寄与である。追跡外質量は約7.24e−27/5.64e−27。上界は実riskの点推定や物理的精度保証ではなく、現在のNの十分性を新たに認定しない。

## 再利用と固定条件

新callerは旧`run_pr_ld1_ld2_deterministic_risk.evaluate_one`を再利用し、既存D.evaluateを各LD1回だけ呼ぶ。K.accept_saved/K.prepare、元count変換/R3/二項確率/whole-vector/DPを変更していない。

INPUTSは43898ada1の科学条件をコピーし、任意のstate_refinementだけを追加した。raw=A91349、solver条件付きproxy、元loss family、ideal finite-K IID、全13roundのX/Y N=`64,60,56,52,47,43,38,34,30,25,21,17,12`、ρ_m=1、EXACT_AFFINE_OF_BINARY64_RAWを保持。保持8192/round、遷移141559036/構成、元count/検証範囲も保持した。元target/bundleのproxy=false、mean UNQUALIFIED、canonical/physical/law/SOURCEを変更しない。

## 正本

- `run01/RESULT_LD1.json`、`RESULT_LD2.json`：各1回の科学結果。exact分子・分母を保持する。
- `record.diagnostic.deterministic_risk.coarse_MSE_upper`は同じ新評価で計算した旧上界。`state_refinement.MSE_upper`はDP上界、トップ`MSE_upper`は有効な両者のmin。
- `DECOMPOSITION.json`：保存結果からexact算術で作ったη、u L_U、r L_bad、改善率。risk APIの再呼出しではない。
- `VERIFY_RESULTS.py`：原mean/target/SOURCE/条件、保存旧上界との一致、DP質量・loss恒等式、scope・nullを検算する。科学評価を再実行しない。
- `run01/RESOURCE_MONITOR.jsonl`は自分のworker PIDだけのCPU/RSS観測。workerの最終CPU/wall/peak RSSは各RESULT、全体時間はCOMPLETIONを参照する。

Pythonは既存専用環境3.11.0rc1/NumPy1.26.4。巨大整数はfloat化しない。decimal桁数制限のある別PythonでのJSON読取りは局所I/O設定が必要になり得るが、今回serializer・共有環境設定は変更していない。

## lossと確率を混同しない

DP上界は`η + u L_U + r L_bad`。ηは追跡した解決終端の実energy loss寄与、uは吸収した未解決履歴の確率、rは追跡外履歴の正確な確率質量である。rのlossには従来の最大上界を使い、tailを0にしない。ηを純shot項と同定せず、rawモデルに含まれ得る近似影響へAδE/δE²を重ねて加算しない。

global zeroはcount法則からexactに計算できる。未解決/tieはr>0の場合は範囲であり、reference-oracle tieと異なる。吸収したprefixの未来zeroや残余内のzeroもあるため、u L_Uとr L_badにglobal zeroのlossを別加算しない。r=0の場合だけ実全体tie/未解決のexact確率が定まる。

既存top-levelのunsafe/reference-oracle-tie/未解決範囲は旧診断、新しい値はstate_refinement配下。unsafeは誤branch又は未解決そのものではない。新kernelの数学的条件は[元HANDOFF](../rpe_mse_state_refinement_20261114/HANDOFF.md)を参照。

## 資格と次の担当

本件は指定binary64 raw・solver proxy・IID・loss familyに条件付くモデル診断。物理target/state/actual sampler law・数値allowanceの資格、元Minkowski各項への接続は別に未取得である。十分N/global RMSE bound/正式G/winnerを生成しない。上界超過だけで方式や配分の不可能性を判定せず、改善率でLDを正式順位付けしない。

**ユーザー**が、実行結果と負荷を見て次の有限whole-shot評価の必要性・risk契約・候補集合を判断する。**Codex B**は新しい明示許可なしに、K/遷移範囲の拡大・配分追加・N探索・価格/Gへ進まない。既存cost/selectは保全し、C/Dへ未資格Nを正式費用入力として渡さない。A/C/監査のworktree/processに介入しない。

本2件の結果保存・確認・commit/push・remote照合の完了後は**mandatory STOP**。旧SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPは保全する。
