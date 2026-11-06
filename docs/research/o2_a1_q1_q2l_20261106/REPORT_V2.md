# A1 v2：payload保存修正・限定再構成の結果

2026-11-06 JST、SERVER1。専用branch `codex/o2-a1-q1-q2l-v2-20261106`。
診断SOURCE：[2121a455d8d84557a26fef969aac4d92d6da8b0f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2121a455d8d84557a26fef969aac4d92d6da8b0f)。公開・照合後に使用した。

**修正した保存処理で準備計算を再実行したが、旧partition/preparation/tail identityとの対応は未閉鎖。CDF・ν・bias計算には進まずmandatory STOPした。**
期待hashの書換え、guard削除、WP04 lawへの置換、label変更はない。[v1結果](REPORT.md)と旧STOPは保持する。

## 修正と再利用

[実行仕様](EXECUTION_V2.md)どおり、hash guardより前に全partition/weight/λ、完全component、basis操作、float.hex、dtype/strideを保存するよう修正した。
同じbasis操作はhashでまとめ、runtime gate objectを除く。既存loader・split・preparation、CPU import helper、純粋な旧partition hash関数を再利用した。
共有coreや元WP11のSOURCEは変更していない。

対象は保存linear H4/1.00 Å/STO-3G/legacy DF rank12/4電子/8qubit/L_D3、m0/q1/δ=.02/r1/K2、threshold0、X/Y共通law。
固定NPZ SHA256は `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。
小さい軌道基底の再構成を含むが、physical-sector ground/state solver、RNG、測定、full interrogation build/compileは実行していない。

## 得られた証拠

初回の[diagnostic_v2](../../../artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2/)と、元runに保存されたthread設定をそのプロセスだけに適用した[diagnostic_v2_recorded_runtime](../../../artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2_recorded_runtime/)で、各一回のload/split/preparationを実行した。
後者はOMP/OPENBLAS/MKL/NUMEXPR/NUMBAの各NUM_THREADS=1を再現したもので、恣意的な一般資源上限や共有設定変更ではない。両者の保存PREPARATION payloadは完全一致した。

| 項目 | 結果 |
|---|---|
| H hash | 元記録と一致 |
| partition hash | `900342ad6262021ff9b919c36ee182a8ed83914a697524e3fae07248d0d082f8`、旧`903439…`と不一致 |
| preparation hash | `f02980a5cbec9478aa1f4e6dd010f117d152f89770c40389544cf876dd197299`、旧`50c77f…`と不一致 |
| tail hash | `890442900e605a262201338f19afdad638302d670c75a9d95371c10b83ea38d8`、旧`c51e35…`と不一致 |
| 旧/現partition hash関数 | 同じ取得payloadでは一致。hash関数の版差だけでは説明できない |
| loaded / complex C-copy / complex F-copy / real C-copy | tensor値は完全一致。いずれも旧partition hashを再現せず |
| 同じthread設定の復元 | 今回の不一致を解消せず |

現partitionの決定論fragmentは0–2、random側は3–11。preparationは324 symbolic componentを保持した。これはCDFの53-bit positive support数ではない。
元の完全partition/basis/component payloadを回収できていないため、完全比較やtarget同値の認定はしていない。係数や一部basisの一致だけをlaw全体の一致の代用にしない。
数値原因は未特定であり、丸め・BLAS・基底非一意性のいずれかに断定しない。hash差だけから物理的に有意な差とも主張しない。

[COMPARISON.json](../../../artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2/COMPARISON.json)は全共通trotterlib Python sourceのbyte比較と期待/実hashを保存する。
共通ファイルではdf_partial_s2.pyとpf_delta_validation.pyに版差があり、それ以外は元WP11と一致した。依存全体・原実行時のbinary環境が完全再現できたという意味ではない。
各診断directoryのSOURCE/PREPARATION/DIAGNOSIS、後者のRUNTIMEが取得証拠である。

## 検証・判断

[人工テスト29件PASS](../../../artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2/TESTS.json)。新規1件はpayload保存・basis重複除去・hex保持を検査した。分子のlaw/biasが検証成功した証拠ではない。
Q1/CDF/support/ν/joint、Q2-L biasの実科学値は未取得。測定可能・CV改善・独立再現/CI・最終総費用評価は認定しない。
修正済みscience runnerをguardの先まで進める条件が閉じていないため、同じ失敗を繰り返す追加runner呼出しは行わなかった。

次に必要な証拠は旧50c77f/c51e35を生成した完全preparation/catalogまたはその原数値環境である。それを回収・比較できるまでは、同じopt2 targetのQ1完了とは扱えない。
現再構成を別targetとして採用することは今回行っていない。今回の許可だけで科学条件を変更しない。

## 保全・公開・STOP

[PRESERVATION.json](../../../artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2/PRESERVATION.json)：元mainのHEAD・statusと既存dirty/untracked 2,136ファイルのbytes、v1四artifactの不変を照合した。
旧研究のSOURCE/input/raw/STOP、C1凍結、S2/R1判定、共有venv/package/cache/OS/他jobを変更していない。
新branchに結果・status/概要/ノート/manifestを追記し、公開後にremote本文・treeを照合する。
[STOP.json](../../../artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2/STOP.json)：**mandatory STOP。Q2-T、Q3–Q5、RNG/測定、full compile、variance/N/costへ進行しない。**
