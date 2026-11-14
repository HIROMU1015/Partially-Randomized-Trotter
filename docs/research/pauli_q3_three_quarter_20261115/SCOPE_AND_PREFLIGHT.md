# Pauli Q3：3/4 whole-vectorの固定2評価

2026-11-15 JST。ユーザーの直接指示のB1はB担当のQ3評価であり、DのPauli B1-W/V/CAL価格検証とは別作業。価格・G・PF riskは対象外。

- 対象：保存Pauli LD1/LD2、各1配分・最大2 science call。caf1417の元control評価は再実行しない。
- 配分：両軸`[48,45,42,39,36,33,29,26,23,19,16,13,9]`。元controlの各要素をexact整数でceil(3N/4)。allocation IDはINPUTSに固定。
- 入力：caf1417のINPUTSからraw/target/law/canonical/数値追跡条件を再利用。B SOURCE9c0e12d87、total kernel710aa69b、高速backend9e294778、C c628e9e8。旧runnerのgate・保存・資源観測を再利用する。
- 採択条件：Z1＋T1、solver proxy、全事象平方loss、ideal IID exact-affine law、Q3上界/下界によるτ判定。εEの十進文字列からτ=εE²をexact有理数で固定。元canonicalのPROPOSED表記は生成当時の履歴として保全し、今回の採択は外部評価条件で明示する。
- 有限範囲：ρ1、round outcomes4225、verification pairs55043173、frontier8192、transition pairs141559036。
- countのみの保守的preflight：round最大2401、隣接組上限18263012、DP遷移組上限80154405で、指定範囲内。実rawの確率やriskをこの段階で評価していない。
- 重複照合：既存controlの2結果と旧partial DPは別保存・再利用。今回の新allocation/run出力は未取得で、同じPauli/Q3科学Python jobは確認した範囲で稼働していなかった。A/Cの科学processには介入していない。
- 高速化：既存benchmarkと完了2診断で確認した各8 worker・2構成並列を再利用。worker数は恒常的上限ではない。共有設定は変更しない。
- 失敗：UNKNOWN/gate/source/範囲/保存エラーではfailure flagを立て、未開始のscience callを禁止。開始済みの独立評価の証拠は保存し、retry・vector/ρ/backend/範囲変更はしない。
- 出力：新namespaceのrun01。元結果・SOURCE・grant/ledger/marker/STOPは不変。数値certificate、ground/実sampler移送、global RMSE、十分N、Gは未認定/null。

実行前にrunner・INPUTS・本範囲をcommitしてSOURCEを固定する。終了後はPF6候補のmetadata準備とともに公開・remote確認し、mandatory STOP。
