# 入力候補・identity・不足資料

[技術報告](technical_report.md)／[byte/metadata検査結果](candidate_input_identity.json)／[条件proposal](candidate_conditions_proposal.json)。全て候補であり、科学条件の正式採択ではない。

## 第一候補

`artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz`

[固定input commit 34d9a2f1…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34d9a2f1a25bd35c03566ce0635e8b66665ad996/artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz)。3589 bytes／SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。既存preflightのhashと一致。container中の全7 NPYについてnpy bytes・array bytes・header/shape/dtype/endian・SHA256を公開した。科学係数の浮動小数点decodeは実施していない。

| member | ヘッダー上の型・shape | 役割 |
|---|---|---|
| constant | little-endian f8 scalar | 保存full constantを一度使う候補 |
| one_body | little-endian c16、8×8 | full一体係数をそのまま使う候補 |
| lambdas | little-endian f8、12 | 全保存blockのsigned weight。実際の符号は未評価 |
| g_matrices | little-endian c16、12×8×8 | 保存順の全行列、無次元 |
| schema_version | Unicode scalar | connected_cluster_df_hamiltonian_snapshot_v1 |
| hamiltonian_hash | Unicode scalar | 旧hash metadata。丸めを含む旧hashは再計算せず、exact byte hashの代用にしない |
| metadata_json | Unicode scalar | 元metadataを別hashで保持。truncation/energyをscalar/Uへ足さない |

既存metadataのbasisはsto-3g、distanceは1.0、chargeは0、multiplicityは1。molecule_type・rank情報・hermitized・hf_energy等のkeyも存在する。これらは旧記録であり、exact Hermiticity、N、HF/ground、元電子Hamiltonianへの同値証拠ではない。metadataの既存数値を新しい物理評価やUの入力に使っていない。

## SOURCEで確認した表示

`src/trotterlib/rte_connected_cluster_cost_validation.py` のwriterは `hamiltonian.constant / one_body / lambdas / g_matrices` を保存する。`DFHamiltonian` は `cI+Q+ΣλA²` のfull表示であり、correction-only DFModelと同一視しない。loaderのfloat cast/legacy hash経路を新certifierの正本にせず、保存bitsを保つ新adapterを用意した。

filenameにL_D=3等があることを理由に12 blocksの一部だけへA2を適用しない。今回の候補は保存full DF全12 blocksを含む固定順序S2で、元のrandom-tail/cost実験の再実行ではない。rawが全元分子項を再現していることは未認証であり、**固定した保存DFモデル自体**への保証だけを候補とする。

mode axisの位置を `m0…m7` と命名し、元array順から並べ替えず、JW mode0 LSB・creation昇順を宣言するproposalである。化学的spin軸の対応やNα/Nβ、S²の保存をここから認定しない。total N=4はH4名目条件に対するproposalで、正式採択待ち。各fragmentのN保存は一体・平方の文法で後日検査する。

trialは候補としてstored indexの最初の4占有を提示する。HFや近似groundと呼ばず、既存energyを見て選んでいない。どのdeterminantをUの証拠にするかはユーザーの採択事項。

## 小さい候補・保留の理由

調査したcommitの保存パスではH2の係数/検証**集計JSON**はあるが、full constant・一体・全DF平方のraw snapshotを特定できなかった。集計値や以前の分子計算を呼んでrawを再生成しない。PR-2の別NPZは他研究・held-outのscopeを保全し、新たに開いたり解析したりしない。H4候補は、既存preflightで追跡済みで全保存memberが揃うという証拠面からの第一候補で、step結果を比較して選んだものではない。

## 実行前に残ること

- ユーザーによる固定DF target、N/mode順、trial/PF順の採択。
- exact Hermiticity・finite値・全signed平方・scalar/PSD台帳の初回実入力gate。今回未実施。失敗すれば修復せず、別モデル化の必要性と原因を報告してSTOP。
- d/α、sqrt/root解像度・反復条件、出力制限、canonical実行場所の採択。
- H4の実際の語数・有理数bit長・CPU/RAM・runtime。構造と人工scopeだけで完走保証をしない。
- 元電子Hamiltonianとの厳密対応、量子ground準備・QPE/nativeは今回のpilot対象外で、未取得のまま保つ。

**science実行承認・新certificateなし。Mandatory STOP。**
