# 2026-10-20 PF残106軸の修正・実行承認

ユーザーが「修正を行って再計算して」と承認。D-sourceまで調べ、前回のB0I preparation仮説を訂正した。実際は回転済みprimary/未回転forwardのcontrol比較と、単一項S4の古いmerged時刻列が問題。B0 discard/native回路は変更せず参照の来歴を修正。単一項S4の20軸は保存exact_truncatedへ固定して旧signal未整合のflagを残す。新13＋既存13人工検査PASS、実分子計算未開始。旧682と旧SOURCE/STOPを保持し、106軸の新契約をsealして1回実行する。詳細は[報告](../track_a_pf_direct_compile_retry106_20261020.md)。

## 終了追記：11:02 JST

SOURCE9c65fb312891d8334037f30ef953336c287bc262、106全成功/失敗0/未実施0/unknown0。全53worker・coordinator・metadata auditor exit0/reap、39,522件最終gate PASS。wall152.063秒、worker CPU1033.940秒、最大単一RSS631103488 bytes、標本合計RSS最大21349957632 bytes。control最大7.055e-13<1e-9。旧682をfield一致で継承し計788の保存task受入が揃った。20軸の旧classical signal未整合と初回572のglobal final gate欠測は保全。新fit/費用順位更新/次stage0。科学実行終了、結果commit後mandatory STOP。新公開未承認。
