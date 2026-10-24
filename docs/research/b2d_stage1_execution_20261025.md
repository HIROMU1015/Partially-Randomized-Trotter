# B2-D Stage 1：限定科学検証・技術復旧記録

2026-10-25 JST。実行条件は[固定科学契約](b2d_stage1_simple_execution_20261025.md)を継承する。
Synthetic 2-mode one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、A0〜A4、δ=±0.09、q=2,3、n_D=1,2、r_RTE=2／K=2、complex128、correctness atol=1e-12、phase無効。分子geometry／AO basis／分子DF rank探索は非該当。
比較は36行・941 DAG dispatch・448 exponential key、記述的診断のみ。科学条件・係数・scope・参照・許容差は変更しない。

## 今回の実行範囲と技術復旧の例外

ユーザーは固定Stage 1の科学実行と、予期しない技術的中断に限る原因調査・最小修正・同一条件での新attemptを明示許可した。
各attemptはfresh outputとし、旧runを上書き・resumeしない。科学的FAIL・非可換性不成立・実際の参照不一致・精度不足・改善なしを、技術エラーとして再試行しない。
旧契約・SOURCE・raw・STOPは遡及変更しない。承認証明・grant等は作成しない。Stage 2、別fixture／sector／state／精度／grid、native compile、総資源比較へ進まない。

## Attempt 1：技術的中断

固定SOURCE `54008c697e686a42ec48941d25219d523bafd454`、指定の `simple-once` へ初回実行した。
preflightはPASS。入力fixture・split・preparation・sector・full H・D block3・tailの9境界は完了したが、Input packageで `AttributeError` となった。
承認管理撤去後も `permit.contract["numerical_atol"]` と `permit.contract["source_commit"]` の2参照が残っていたことが原因。
状態 `FAILED_MANDATORY_STOP`、science DAG未開始・比較0行。STOP・events・binding・partial result・statusを保全した。
wall5.64秒、user CPU10.69秒／system CPU2.95秒、max RSS364156 KiB。Matplotlibは非書込の既定cache先から専用tmpへfallbackし、計算停止の原因ではなかった。

## 復旧差分と独立した人工確認

変更は2参照だけ：atolを科学scopeのgraph settingsから、SOURCEをrun contextのsource_commitから取得する。
承認contractを再導入せず、科学値・入力recipe・数値algorithm・許容差を維持する。
全科学import／関数をopaque stubへ置換した実loaderテストを追加し、正常なleakageと漏洩拒否の両branchを確認した。
旧SOURCEの関数を同じstubで実行すると両branchともmissing-contract AttributeError、新SOURCEでは正常完了／所定のSECTOR_BASIS_LEAKAGEとなることを独立確認した。
人工6ファイルは15＋14＋16＋15＋4＋7＝**71件PASS**。実科学量を使った原因探索・係数変更・閾値緩和は行っていない。

## 次のattemptの起動条件（この記録の固定時点は実行前）

本修正とこの記録を含むcommitを新科学SOURCEとして固定し、clean detached checkout `/tmp/prt-b2d-stage1-recovery-science-20261025` を用いる。
新出力は同checkout内 `artifacts/research_b_b2d_stage1/20261025/technical-r1`。旧出力・STOP・生成済み入力を再利用せず、同じ固定recipeを最初から実行する。
SOURCE変更理由は上記の明確な接続バグのみで、固定scope・科学条件は不変。予期しない技術中断だけがユーザーの今回の再実行許可の対象である。
32 CPU、空きRAM約51 GiB、GPU利用不可を確認した。2×2 sector・vectorized演算・448 key共有cacheを再利用し、共通入力／gate順序を守る単一processとする。CPU／thread／RAMの人工上限や共有環境変更なし。
結果・全gate・call実績を確認後、正常／科学的STOP／INCONCLUSIVE／解決不能な技術停止のいずれでもmandatory STOPする。
