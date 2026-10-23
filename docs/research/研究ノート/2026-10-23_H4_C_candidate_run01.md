# 2026-10-23 H4 C候補run01：reference欠陥による停止

利用者の次段階承認に基づき、固定SOURCE49c18ccc3・元40軸・有限precision/予算を維持して新grantを一回実行した。B結果・旧科学SOURCE・原grant/STOPは保全した。

Pauli20軸rawのreferenceが0となることを確認し、sourceで右側state作用の欠落を特定。人工|1⟩・Iの追加回帰は1 FAIL（期待1、現call0）。前回の最初のbasis vectorに偏った人工fixtureではこの欠陥が見えなかった。過去110 PASSの履歴は保全し、今回のFAILを別記録とする。

自分のDF witnessをpidfdで同定してSIGTERMを送り、同じ欠陥を含む計算を停止。Pauli exit0/DF exit−15/両bridge exit0/supervisor exit1・全回収を保存した。Pauliのraw TARGET_NOT_METは科学的な方式不適格・予算達成不能を意味しない。DF最終tierは未取得。数値bias受理0軸、旧N/Gはnull/UNDETERMINEDを維持する。

最小patchは未適用。新SOURCE・非真空/複素重ね合わせ回帰・別grantでの再実行はユーザー承認へ戻す。元H4 native欠落/原因UNKNOWN、DF event範囲/compiled wrapper転送等は別未解決。科学retry/shot/G/公開push/最終freeze0、local commit後mandatory STOP。

[実行・診断報告](../../non_df_h4_c_candidate_run01_20261023.md)、[保存資料](../../../artifacts/non_df_h4_c_evidence/2026-10-23/README.md)。
