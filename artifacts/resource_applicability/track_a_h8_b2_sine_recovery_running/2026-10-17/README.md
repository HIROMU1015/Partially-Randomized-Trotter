# H8 B2 sine recovery: running snapshot

保存cosineは実測1,852,640 RZ/誤差0%でPASS。元runのMEMORY_PRESSURE/STOPを保持し、検証済みsine QPYから残る1軸だけを再開。新しい実測完了結果はまだない。

`python3 -B verify_saved_running.py` は保存bytes/metadataだけを確認し、live probeや新科学を実行しない。Phase C未承認、B2終了時mandatory STOP。
