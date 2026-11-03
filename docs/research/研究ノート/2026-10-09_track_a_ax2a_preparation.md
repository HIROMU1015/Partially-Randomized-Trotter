# 2026-10-09 Track A AX-2A準備のdecision record

利用者が共有したAX-1b後の独立科学レビューと「次の作業に進めるか」を受け、
設計・限定実装・synthetic tests・pilot準備に進んだ。
主RQ-R、補助RQ-P1、RQ-P2非必須、旧FEW凍結を追補案へ反映した。
理由は、H4 fitの追加改善や単なるサイズ増加より、強いbaseline・数値headroom・
同一DF政策・実資源の比較が研究の結論を強くするため。

既存sector/matrix-free/GPU・finite-RTE・高次PF機能を確認し、未実装と呼ばない。
不足したTrack A有限平均state-actionとglobal四次callback、対称controlled意味論IRを追加した。
57 local synthetic tests passed。native DF lowering/H4 correctness/H6 profileは未実施。
DF tolの実装はdiscarded weight和を使うが、pipeline全体の誤差保証と採用tolは未確定。

元dirty worktreeを保護するため、AX-1b結果commit `b2e1bf65e21893b6c617223b42313623d3186f12`から
branch `track-a-ax2a-preparation-20261009`を作った。旧source・契約・結果・manifestは変更しない。
準備artifactを別inventoryへ追加し、原稿v0.1/Track B/H12/過去STOPも保存した。

AX-2B草案はH4 correctness 3 bundle→構造的H6 6 cell、H8 0。
CPU1/RAM8 GiB等は新しい提案であり実割当・認可ではない。
native adapter、DF/state/reference、gate/matvec cap、source/task freeze、別の明示認可が必要。
新科学計算0、commit/pushなし、終端 `AX2A_PREPARED_AX2B_NOT_AUTHORIZED` で停止する。

正本は[比較契約追補案](../track_a_ax2a_research_amendment_v1.md)と
[技術実装・pilot準備](../track_a_ax2a_technical_preparation_v1.md)。
