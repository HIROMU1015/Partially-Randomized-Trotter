# 2026-10-10 Track A：coverage serialization修正・準備v3

[準備・実行範囲](../track_a_ax2b_h4_coverage_preparation_v3.md)へ記録する。旧STOP/source/freezesは不変。
tuple/list表現差だけのstrict canonical比較、actual coverage/差分のbounded保存、H4-only v3 grant/専用output結合を新versionへ追加した。
49 local metadata/mock tests pass。分子load/prepare/signal/probe/sampling/build/compile0。数値toleranceを緩和しない。
今回利用者指示を、直前の新固定→別認可→同じ8 cell/capsのH4一回実行→公開後STOPへ適用する。旧grant/output retryなし。
新固定とremote確認後に別認可を作る。科学的矛盾/意味論変更が判明した時点でGPTへ戻す。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATIONを維持。
