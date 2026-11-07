# N3-B1b-1：保存成果物

固定H4/STO-3G/DF12・Partial S2/LD3/h=.02/K2・q128/r1/m7、b36、43基本key、基本strict precision10^-13／最終10^-11。43/43取得、providerと誤差予算PASS、B_dict=6072、B_cache=156、B_P=6072<173790。全support CT RPE成功／18round総Tは未資格。mandatory STOP。

- [REPORT](../../../docs/research/pr_reference_n3b1b1_dictionary_20261108/REPORT.md)／[provider仕様](../../../docs/research/pr_reference_n3b1b1_dictionary_20261108/DICTIONARY_PROVIDER_SPEC.md)
- [SCOPE](SCOPE.json)／[INPUTS](INPUTS.json)／[ENVIRONMENT](ENVIRONMENT.json)／[SOURCE](SOURCE.json)
- [43 key manifest](BASIC_WORDS_MANIFEST.json)／[取得summary](ACQUISITION_SUMMARY.json)／[実records archive](basic_records.tar.xz)
- [strict budget](STRICT_ERROR_BUDGET.json)／[C0 headroom](C0_STRICT_HEADROOM.json)／[correctness](COMPOSITE_CORRECTNESS.json)／[CHECKS](CHECKS.json)
- [T価格上界](T_PRICE_BOUND.json)／[validation inputs](VALIDATION_INPUTS.json)／[STOP](STOP.json)

basic_records.tar.xzは、manifest各rowのrecordパス（synthesis/key.json）、acquisitionパス・log、実取得SOURCE snapshotと開始markerを同じ相対パスで含む。raw directoryとlockはSERVER 2にも保持し、削除・上書きしていない。archiveのSHA256と各member hashは[RECORD_ARCHIVE.json](RECORD_ARCHIVE.json)。C0の元wordは親commit内のCT-4B archiveを参照する。

取得SOURCEは既存acquire／strict_guard／exact_word_matrixを変更せずに実行した。取得済みrecordのguardは再取得・再guardで作り直していない。backend recordとproviderのcomposite certificateを混同しない。

再現に必要なSOURCEはrepositoryのscripts/run_pr_reference_n3b1b1.py、scripts/validate_pr_reference_n3b1b1.py、src/trottertracks/pr_reference_dyadic_dictionary.py。rawを復元する場合はfresh領域へarchiveを読み、旧recordを上書きしない。現在のSTOP済み出力へacquireを再実行してはならない。追加の科学実行・再解析は別途の明示指示が必要。
