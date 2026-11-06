# 新A B′限定検証：保存結果

[REPORT](../../../docs/research/pr_reference_bprime_20261107/REPORT.md)、
[SCOPE](../../../docs/research/pr_reference_bprime_20261107/SCOPE.md)、
[構造補遺](../../../docs/research/pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md)。
科学実行は `STOP.json` で完了。V3〜V6へ進まない。

- `SOURCE.json`、`EXECUTED_PRICE_SOURCE.py`、`EXECUTED_ACCOUNTING_SOURCE.py`：SOURCE hash、呼出順、入力、archive、保全状態。
- `INPUTS.json`：固定common入力の来歴。旧commitの圧縮pickleを再利用し重複公開しない。
- `ENVIRONMENT.json`、`V2_PLATFORM_DETAILS.json`：SERVER 2、runtime、compiler、wheel、CPU/libc/丸めmodeの読取観測。
- `CORRECTNESS.json`：24実角の全joint/inverse、4CX/2RZ、r2全512次元照合。旧suiteは再実行しない。
- `LOCAL_SCOPE.json`、`CATALOGUE_SUMMARY.json`、`Bprime_local_catalogue.tar.xz`：4 contextとpre-final境界。
- `BOUNDARY_CHECKS.json`：9境界例と継承lowering一致。経験的な例を全supportの相殺一致定理にしない。
- `PRICE_EXECUTION.json`、`paired_Bprime_prices.tar.xz`：保存64 path・128 whole B′ compileの実価格、seed対応、両passのCX削除node IDとwire。reference/旧baselineは保存価格を含めるだけ。
- `PAIRED_PRICES.tsv`：raw価格欄の転記。unweighted stratum平均を計算していない。
- `Bprime_native_qpy.tar.xz`：q4/j0/X、両stratumの2 native QPY。
- `PRICE_BOUND_CHECKS.json`、`REFERENCE_RZ_LOWER_CHECKS.json`：全128軸の価格包含・RZ一致。B′ CX下界は全parity CXを捨てる安全側境界。
- `V2_INPUTS_CONSTANTS.json`、`V2_BUILDER_WITNESS.json`、`V2_Q1_BUILDER_WITNESS.json`：336/49/24/20、実lambda/角度、既存SOURCEの小端点観測。
- `ERROR_RPE.json`：旧区間再利用、新kernel解析誤差0、B′全18roundのnative/raw disk・半径・phase・再計算整数shot。
- `TOTALS_18ROUND.json`、`ROUNDS.tsv`：3方式の期待総費用、finite-bit有理価格/確率、上下界、保証できる差。
- `STOP.json`：V1/V2完了、旧STOP保全、次担当GPT／ユーザー。

archiveは展開元の全ファイルとbyte一致を確認した。展開元と入力cache、旧catalogueの専用読取snapshotは `.gitignore` で重複公開を防ぐ。
結果は固定保存H4・理想gate・binary64算術モデル・compiler Aに限定する。全18round depth/size/Tは未資格／未取得。
