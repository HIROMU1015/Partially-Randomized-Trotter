# 2026-10-23 H4 C修正版：新grant実行

利用者の「次の作業に進んで大丈夫」を、修正版SOURCEでの固定40軸実行への承認として受領。SOURCE e80d7510880f8db993b777c627bb8e95b05a070d、base450b0a787、新grant6862c863f376be24c4ab56bc2d21abd2f8c2560b80250f95d4a1e6fd8b0cb4b4。旧grantの再利用・旧runのresumeではない。

18:53 JST：source/runtime/input SHAと人工reference14件をfresh確認し、新しいgrantを一回消費。Pauli/DFを独立process、各worker・bridgeとsupervisorをnative witnessで監視。科学条件・閾値・予算は不変。

18:54 JST：Pauli20軸が64 bit TARGET_MET、worker native exit0、両bridge native exit0。DFの計算は継続。正式certificate受理・N/G会計・pushは実施しない。

[結果・監査](../../non_df_h4_c_reference_evidence_20261023.md)。以下に終了時の結果を追記する。


19:13 JST：利用者が計算待ちのため一旦chat終了を許可。科学workerを止めず、保存済みPauli結果・新grant/claim・入力model・正常native receiptとread-only観測をhandoffとしてlocal commitする。DF結果/全体STOP/最終監査は未確定。既存の実行承認は継続し、次回Codexが読取確認と終了後の保存監査を回収する。再launch・grant再発行・retry/resume・科学条件変更はしない。


19:15 JST：DFを含む固定40軸が64 bit TARGET_METを保存して終了。全worker/bridge/supervisor native exit0・回収、所要約21分24秒。19:20以降、利用者の残り時間確認を受け、既存承認の範囲で保存結果の構造・SHA・native終了・元precision tableを回収照合し、旧2,239 file不変を確認。PRの保存log_B上界は両表現で包含true。正式certificate受理0、DF event範囲/compiled wrapper転送/PCG64独立性・旧N/G未確定/元native欠落は保持。新科学計算・grant・再会計・pushなし。結果と終了監査をlocal commitしmandatory STOP。
