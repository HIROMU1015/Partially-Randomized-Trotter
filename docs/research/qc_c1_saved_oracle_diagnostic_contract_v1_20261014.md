# C1保存oracle・selector診断：固定実行契約 v1

2026-10-14 JST。scope `QC_C1_SAVED_ORACLE_POSTHOC_V1_20261014`。
基準commit `cc992b579290233ab5a7d8408d4a81c6d295f36d`。
今回認可された作業は実装・人工fixture検査・identity/schema-only preflight・公開まで。
**登録保存データからの新規oracle/regret評価は未認可。940-call認可は消費済みで流用不可。**
source commit、runner/checker、source/input/runtime SHAは[execution plan](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/execution_plan.json)と各manifestで固定する。
本契約はsource freezeに含め、公開追記で書き換えない。

## 入力・処理

固定comparisons.json、selector_seal.jsonをread-onlyで使用する。summary/旧STOP/元source20filesもbyte/SHAで保全。
保存eligibleをそのまま使い、branch/overlap/bias/gap/PF bound/費用/selector scoreを再計算しない。
原条件は非分子2qubit Pauli4H、24macro、6h、通常state16/bad-state16contexts、target10^-5/10^-6 model units、controlled価格3/6。
h=1/(8Λ2^s),s0…5、Λ=fragment係数L1。geometry/basis/DF rank-policy/L_D/固定粒子sectorなし。
同一contextの保存eligible・positive finite costから次を得る。

- C_M：24macroの最小費用。
- C_B：single/subdivision12＋S4三つ＋C2三つの最小費用。
- C_K：single/subdivision12＋24macroという同じC1 catalogの最小費用。
- macro_ratio=C_M/C_B。
- method_regret=C_saved_selector/C_K−1。C_saved_selectorは保存frontierで、選んだmethod内のhが既にoracleとなっている。

empirical/joint/boxを別記する。運用precision選択を含むregretではない。jointの保存precision/nullはmetadataとしてコピーし、新しいboundを評価しない。
minのexact tieは保存row順で先頭を代表にし、同費用row indexを全て保存する。未判定を削除しない。
null ineligible costは許可、eligible-null/非finite/zero/negative/bool/欠落/重複/未知method-h/条件不一致はschema failureで停止。

## 算術・分類

costの保存binary64値をFractionで正確に扱う。費用モデルや費用そのものの再生成ではない。
5% gateはstrict macro_ratio<19/20。境界判定は各costの隣接binary64値一つずつによる感度guardを固定する。
upper ratio<19/20ならADVANTAGE、lower ratio>=19/20ならNO_ADVANTAGE、跨ぐ場合はUNDETERMINED。
これは浮動小数の丸め感度で、元eig/PF/costの物理的・科学的不確かさの保証区間ではない。energy eligibilityは変更しない。
exact ratio、guard、null理由、原row pointerを出力し、見かけの小差を強い結論へしない。

分類は二軸で、相互排他としない。

1. `A_NO_ORACLE_5PCT_ADVANTAGE`：oracleにも登録5%優位のmacroなし。
2. `B_ORACLE_ADVANTAGE_SELECTOR_MISS`：oracleは優位、少なくとも一selectorが5%利益を得ない（selected eligible pointなしを含む）。selector毎の判定も保存する。
3. `KNOWN_COMPOSITION_EQUIVALENT_ORACLE_ADVANTAGE`：oracle優位があっても現known-compositionと同じfactory/catalog。第2分類と共存し得る。
4. `UNDETERMINED`：oracle gateの候補不足/丸め境界、または一部selector/regretが未判定。`unresolved_parts`でどの部分かを明示し、他の判定を上書きしない。

C1とknown controlのseal選択・保存frontierの一致はpointerで確認する。新しいselectorを生成・最適化しない。

## 実装と独立監査

core/schema/spec/executionは `src/trottertracks/representation_exploration/c1_saved_oracle_*_v1.py`。
runnerは `scripts/run_qc_c1_saved_oracle_diagnostic_v1.py`。
独立checkerは別moduleと `scripts/verify_qc_c1_saved_oracle_diagnostic_v1.py`。
checkerはevaluatorをimportせず、別のsorted-pool/cross-multiply手順で指標・tie・分類を再構成する。
共有するのは宣言scope、schema、authorization/hash、stdlibの算術だけ。外部研究者の科学再現ではない。
検査は `tests/test_qc_c1_saved_oracle_preparation_v1.py`、人工F_COST/費用表だけ。登録H/vector/比較データをテストへ読み込まない。

## 一回制限・資源・認可

future outputは `artifacts/qc_c1_saved_oracle_diagnostic_v1/2026-10-14/attempt_1/` 固定。
source/input/runtime/planと明示承認文のidentityを照合し、output mkdirを原子的に一度だけ消費してから新規集計する。
既存outputは再利用せず拒否。独立auditも専用markerを原子的に一度だけ消費し、失敗/完了後の再集計を拒否する。例外はfailure記録を残してSTOPし、retry/別namespace/再launchで回避しない。
承認typeは `C1_SAVED_ORACLE_POSTHOC_APPROVAL_V1`、status APPROVED、scope/limits/identity/ユーザー明示文が必要。
準備時のapproval_targetは認可ではなく、runnerが拒否する。今回発行しない。

最大8064rows/32contexts、1worker/1thread、address-space256MiB、総output2MiB、log1MiB、attempt1/retry0。
eigensolver/PF/compile calls0。phase/total wall capなし。
JSON/辞書とFractionのbounded table処理なのでCPU serialを使用する。GPU/Numba/行列backendを起動しない。
一回のschema走査とcontext内pool scan、checkerはcontext内sortを用いる。speedup factorは主張しない。
output上限からfailure用8KiBを予約する。CLIはbounded JSON status/errorだけを出し、長い数値tableをstdoutへ出さない。
チェック後も元科学source/入力/結果/認可/STOPは書き換えず、新結果・独立監査・GitHub保存後Mandatory STOP。

## 科学的な限界

oracleなしは現catalog/grid/modelでの限定的診断。C1一般の不能性ではない。
oracle優位はselectorの情報障害を切り分ける材料であり、運用可能な安い情報取得/分子移送/full QPE-RPE cost/独立新規性の証拠にはならない。
現constructor/known compositionの同値性は変わらない。
正式な継続/終了/再設計/C4/研究採択はGPTへ残す。新数学探索・一般文献再調査を今回の準備や後続診断へ含めない。
