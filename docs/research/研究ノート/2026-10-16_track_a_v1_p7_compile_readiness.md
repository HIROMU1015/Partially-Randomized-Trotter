# 2026-10-16 V1：B0+I p7/q1 S2実測の準備

利用者の指定科学レビュー第8節の第1優先として、[p7 S2の2軸実測契約](../track_a_v1_p7_compile_readiness_20261016.md)を準備した。新compileの実行は明示的に未承認。
結果commit25bbdd33…のremote一致後、専用branch/worktreeを分離し、元V1結果branchと科学SOURCEは不変。固定予測72621/5 RZ/軸、各軸2%以上で即STOP、合計2compile、再fit/retry/拡張禁止を採用した。
scalar C・full-H reference・経験的allowanceと保存shotを再利用し、signal/shotを再計算しない。native control検査は将来の回路同一性検査として上限4回を明記する。
実測後のweighted RZとPR保存値403423200の比較は、さらに別の数値再解析承認後に行い、compile phaseから自動接続しない。
p7専用19を含むpinned90人工tests PASS、metadataでSOURCE/provider/runtime/inputと既存660ファイル不変。新科学計算・有効grant発行0。commit/push/remote確認後mandatory STOPし、実行承認を待つ。
