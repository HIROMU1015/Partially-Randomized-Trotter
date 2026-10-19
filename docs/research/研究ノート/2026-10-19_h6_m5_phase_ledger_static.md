# 2026-10-19 H6 full m5：保存gate列・位相帳簿の静的調査

結果71481e48を受け、userが静的調査を承認。新branch track-a-h6-m5-phase-ledger-static-20261019で、a3089e7dのmetadata検査器を追加した。2QPYを2processで構造だけ読み、全回収。新statevector/compile/実位相総和0。

ancilla12のH2回・scalar P1回、binary64 parameterとgate数を確認。固定Qiskit release896d1c8bの11 source/選択Python5file byte一致を確認し、basis伝播→1Q再合成→binary64位相加算/2π wrapへ追跡。Rust Complex64はf64成分でNumPy complex128相当。具体的な発生passや寄与量は未確定。

次案として、保存phase2値とwrapper P角度の正確なscalar和から、determinant位相をmod π/4096で比較する案を導出。runner未実装・実数計算未承認。determinantはunitary証明ではなくbranch aliasもある。phase補正や旧FAIL受入へ自動接続しない。

人工構造5＋厳密記号3の8件PASS。2223保護file/1831科学seal不変。旧STOP・58軸・raw1e-9未満・AD03保留を維持。local commit後mandatory STOP、pushなし。次はユーザーのscalar台帳準備/数値実行の判断。[報告](../track_a_h6_m5_phase_ledger_static_20261019.md)。
