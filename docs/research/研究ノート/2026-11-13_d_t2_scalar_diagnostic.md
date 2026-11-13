# D：T2 scalar quantityの限定原因診断

2026-11-13 JST。[REPORT](../d_t2_scalar_diagnostic_20261113/REPORT.md)／[DIAGNOSTIC_RESULT](../d_t2_scalar_diagnostic_20261113/run01/DIAGNOSTIC_RESULT.json)／[NEXT_STEPS](../d_t2_scalar_diagnostic_20261113/NEXT_STEPS.md)。

ユーザーが別途明示承認したLD2/m12 preparation-only診断をSOURCE244609df5b704723cae5ccd01f78468bca9588b3で一回実行した。新9人工確認後にSOURCEをcommitで固定し、元loader/context/guardを無変更で使った。input復元1/context1、preparation cache1、sampling/event/native/full wrapper/compile/T2 retry0。

A descriptor0.2223759094107018はtail-only identityで、D抽出値とhex完全一致。D totalはham.constant2.29310124732を加えた2.515477156730702。D748専用selectorがtail-only値をtotalのexpected欄へ渡したことが停止原因。合計のbinary64丸め残差2^-54は別に存在するが、今回の大きな差を説明するものではない。

実値/decimal/hex/差/ULP/hashと時間を新run01へ保存した。旧SOURCE/guard/prepare/expected scalar/入力SHA/T2原本/STOPは保全。新値を採用せず、接続修正・T2新run・E[C]/Gは実行しない。ユーザーが修正と有限科学再実行を別途判断する。公開照合後mandatory STOP。
