# C1実compiled価格の取込み（2026-11-11 JST）

[REPORT](../rpe_c1_price_results_intake_20261111/REPORT.md)／[HANDOFF](../rpe_c1_price_results_intake_20261111/HANDOFF.md)。C1公開`2e6a0cdf5`の32価格recordを、既存adapter `724d313b`の`read_published`でcopy受入した。adapterとtestは無変更で、CLIにだけ`--full-menu`を追加した。凍結commit `7eeabf0`が公開commitの祖先であること、凍結時にVALIDATIONが存在しないこと、凍結ファイルが公開時までbytes不変であること、PLANと科学SOURCEが一致すること、8セルのidentity・時刻（candidate.T＝qPF·δ。binding.Tは旧参照値）・allocation・compiler・値を確認した。

保存校正、新R校正、独立検証、model予測はscenario名と`price_kind`で分かれる。PR価格はすべて有限cohortで、modelは未採用。校正点のmodel予測は、3セル3係数の厳密fitなのでdirectと一致し、精度の証拠ではない。coverageはAの全R menu＋C1の118行で、C1の8行だけに価格がある。新R4セルはN欠測、長q・大Rは価格未取得、Gは未評価。

VERIFY PASS、既存130 test（129 PASS＋任意skip1）、新科学計算0。公開照合後mandatory STOP。
