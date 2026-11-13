# 残22round completionの検査

専用Python `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B` を使用。

| 実行 | 結果 | 対象 |
|---|---|---|
| tests/check_df_head_pauli_tail_completion.py | 6 PASS | 22件限定/sector36のみ、旧点非呼出し、first-FAIL/SOURCE不一致STOP、旧値/identity/law/SOURCE保持、partial/null、duplicate/範囲外拒否、人工m12 |
| tests/check_df_head_pauli_tail_p1.py | 6 PASS | 既存B辞書identity/κ/固定R・full/sector・独立raw・normalizer共有・foreign SOURCE拒否 |
| caller --preflight | PASS | canonical/旧4件の内容/SOURCEのみ。real mean/normalizer/配列loadなし |
| 新run01 | 22/22局所数値gate通過 | 新66complex/normalizer22組、旧4meanは再計算せず再利用 |

計12人工・回帰PASS。新テストはnp.loadを禁止した人工/mock fixtureで実H4を使っていない。q4096も人工小行列で確認した。既存provider/P1 adapterはbyte不変。新しい物理手法・閾値は実装していない。

SOURCE7084fbc9a固定後fresh run01を一回実行、4.0085秒。retry/precision fallback/閾値変更/再正規化0。最大closure4.3728735746241794e−10、power unitarity8.81893828278443e−12（gate1e−9）、保存reference pair差9.279812219411241e−12（gate1e−10）。normalizer22組は有限/b>1、精度未解決/overflow/underflow0。

rawをfce4e4f7eへ固定してからpacketを統合。旧4recordはP1と同内容で、bias/allowance表示も保存値から転記。これは著者による技術・数値検査で、独立レビュー・外部再現・十分N/全RMSE/価格優位の証明ではない。
