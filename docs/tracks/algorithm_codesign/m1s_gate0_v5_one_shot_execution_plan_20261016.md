# M1-S Gate 0 v5：初回one-shot実施計画（2026-10-16）

正式採択の固定・公開後、次のv5科学計算には別途承認が必要と案内したことに対し、ユーザーは別turnで「次の作業に進んで大丈夫」と指示した。
今回の発言と直前の境界を[実施承認文脈](../../../artifacts/track_b_m1s_gate0_v5_execution_preparation/2026-10-16/v1/user_execution_approval_basis.json)へ保存し、新しいv5専用one-shotの承認として扱う。IDE fieldsや消費済みv4 authorityを流用しない。

正式契約C=50b3669b9253eb889fa4daa87e793646df2d5138、SOURCE S=69de022dc8142f703423b1ce0bd42f05e123954e、proof=b5a185cc0c3aae0e477bfc52b93061e86feb7dec。
H4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12、保存DF順prefix G3/G6/G9、非冗長signed Pauli-tail、identityをHDへ移動、cutoff0、保存数値sector oracle、delta/finite-step評価なし。
17精度項目、alpha_R=gamma_(713+r)S_R、global/local分離、1e-8 Ha+B_res、条件付きcoverage、runtime/resource、one-shot/失敗分類を変更しない。

SOURCE8/proof2/raw入力/契約hash、Python3.11.0rc1、NumPy1.26.4配布物と6build hashをmetadataだけで照合する。
CPU1 worker、5 thread環境各1、GPUなし、4 GiB AS、result/log各256 MiB、context4 MiB、Pauli65536、global contributions1000000、scientific wall capなし。
保存v4 timingと静的SOURCEからPauli集約と256次元reference/Pauli作用を主負荷と見込み、既存bilinear/block/reference/mask cacheとtail尺度再利用を維持する。
比較条件を変える並列化・追加benchmark・SOURCE改変は行わず、初回認可run内のreference/correctness gateで検証する。speedup主張なし。

準備を固定した後、authorization-only commit Aを新規作成し、proof < S < C < Aとraw bytesを照合、pushする。
clean HEAD=Aでguardを非数値確認してから、外側launcherでA固有の/tmp exclusive markerを一つ作りfrozen runnerへ一回だけexecする。
outputはartifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot。存在済みdirectoryやmarkerを削除・再利用しない。
guard/runtime失敗時もone-shotを消費し、自動retryしない。

初回入力/基底/JW/reference再構成/Hermiticity/oracle残差・数値correctnessと事前局所予算を確認し、通過した分割だけW/v/rhoを評価する。
全G3/G6/G9行と未到達理由を保存し、PRECISION_UNRESOLVED/TECHNICAL_STOPを科学的no-goやrho=1へ読み替えない。
結果後の閾値緩和、T1c/Q4/Q5、sampling、simulation、spectrum/gap、circuits/compile、総資源評価、追加runなし。
結果/context/log/checksum/消費済markerを保全し、saved-only監査とGitHub公開後mandatory STOP。
