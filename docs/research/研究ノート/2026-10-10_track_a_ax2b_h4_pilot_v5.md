# 2026-10-10：AX-2B H4 pilot v5完了・保存監査STOP

利用者の「その作業に進んで」を、固定v5のH4限定pilot一回実行・保存監査・STOPへの認可として適用した。
元v5準備を保持するため、新execution worktreeへ既存2,051 filesをbyte一致でコピー。
157 source/6 validation hashes、188-entry inventory、保存input/環境を照合し、実行前167 synthetic testsがpassした。
新v5 schema認可・CPU3/thread1・exclusive outputを束縛し、capを変えず一回だけ実行した。

linear H4 1.00 Å/STO-3G/legacy rank12/generation-prefix/T=0.8、保存state/36-dimensional sector、
prefix0/6/12、q1/q4（δ0.8/0.2）、R8/r2/K2,K4,K6の登録8 correctness cellがpass。
global四次B1 prefix12/q1,q4を含む28/28 measured wrapper（state preparationを除く）を保存した。
parent wall787.984819秒、worker exit0、phase900秒/total2700秒・RLIMIT_AS8 GiB内でCOMPLETE。
recorded cumulative peak RSSは約3.290 GiB。部分停止runとの単純なmemory/runtime倍率比較はしない。

凍結verifierはH4_SAVED_PILOT_EVIDENCE_VERIFIED。8 cell/28費用/7 trajectoryをcomplete pathで監査した。
追加stdlib監査では旧v3と8 correctness bytes・24費用/identity・6 trajectory bytes、
旧v4と8 correctness bytes・16費用/identity・5 trajectory bytesが一致した。
391 diagnostics、7 group解放、compile28/trajectory4/occurrence16/control108/primitive962/reference3621を確認。
四次q4 cosineのordinaryはRZ238,601/CX138,464、directionalはRZ88,985/CX43,160。
数値allowance・shot/total cost・方式順位を結論する値ではない。

判断：H4_TECHNICAL_PILOT_COMPLETE_SAVED_AUDIT_PASSでmandatory STOP。
strong baseline四次q4の費用欠落を解消し、登録H4 scopeのbounded feasibilityは確認できた。
総数値allowance・accuracy適格性・shot/normalization・探索公平性の科学レビューが次の段階。
H6 state/DF rank/prefix政策と計算契約を先に確定し、H4完了だけで自動GOは出さない。
元v5/v4/v3/dirty worktree、旧研究結果・原稿・Track Bを保持し、commit/pushなし。

[結果・証拠・次の判断](../track_a_ax2b_h4_pilot_v5_result.md)に詳細を記録する。
旧v3/v4の失敗結果とv5準備の未認可状態は履歴として書き換えない。
