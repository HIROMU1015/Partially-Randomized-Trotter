# A1：旧opt2来歴の限定静的調査（公開要約）

2026-11-06 JST。基準snapshot：[3354b8b](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3354b8baa70b5ed7a798069db8f2f4c4fdd4b526)。

**記録SOURCEの回収先は特定できたが、旧3hashを生成した完全payloadと原binary環境は回収できなかった。科学再実行は0件。**
この公開要約はSOURCE・既公開provenanceへの参照と非識別的な調査結果に限る。未公開checkpointの本文・path・時刻や内部環境設定の全文を含めない。

## 条件と回収結果

対象は保存linear H4/1.00 Å/STO-3G/legacy DF rank12/4電子/8qubit/L_D3、m0/q1/δ=.02/r1/K2、threshold0、X/Y共通law。旧期待identityはpartition `903439…`、preparation `50c77f…`、tail `c51e35…`。
旧期待hashの記録は[WP06b physical_instance](../../../artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json)と[WP05a physical_instance](../../../artifacts/research_direction_full_scope/2026-09-22/wp05a_full_controlled_interrogation_connection_v1.json)を参照。

| 記録対象 | SOURCE回収先・限界 |
|---|---|
| WP05aの記録4ファイル | SHA256に一致する版を`26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c`で回収 |
| WP06bの記録4ファイル | 同じく`26aa95c7…`で回収 |
| WP06aの記録2ファイル | 同じく`26aa95c7…`で回収 |
| opt2本実行の記録10ファイル | `7753eb71844e429facdb6f32c2def6e37379541f`で回収。WP11の`8b574c54…`も同内容 |
| 初期smoke/timingの旧opt2モジュール | 記録された`f169ec2b…`版は今回のcommit/path履歴/archiveでは未回収 |
| 完全partition・basis操作・component catalog | 旧hashに結び付く完全payloadは未回収 |
| 元binary環境 | version記録はあるが、原wheel/build・BLAS/LAPACK等の完全lockは未回収 |

回収は明示されたmanifestファイルの範囲であり、全推移的依存を証明するものではない。dirty生成時のHEADだけでは原SOURCEを代表できない。
WP05aのmanifestは元`df_partial_s2.py`のSHA256 `2a9ff548…`を直接記録し、その版は既存archiveにも保存されている。

## 旧hash記録とSERVER1 workerの来歴

旧hashを保存するWP05/WP06資料のprovenanceはPython **3.11.0rc1**、NumPy1.26.4、Qiskit1.3.0を記録する。
一方、[SERVER1 opt2 run provenance](../../../artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_fresh32_20260924_232418/run_provenance.json)はPython **3.12.3**、NumPy1.26.4、Qiskit1.3.0を記録する。
この差をhash不一致の原因とは認定しない。

[元opt2 SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7753eb71844e429facdb6f32c2def6e37379541f/src/trotterlib/research_direction_full_opt2.py)の`_worker_preparation`は同NPZからsplit/preparationを再生成する。
WP06b artifactのfingerprintは検査するが、準備のpartition/preparation/tailを旧3hashと照合せず、それらをworker結果にも保存しない。
元WP11の既存checkpoint53件も静的確認したが、その3hashや完全payloadは見つからなかった。本文・環境抜粋はこの公開要約には収録しない。

したがって、**WP06bに保存された旧3hashとSERVER1 workerの実preparationが同一だったとは、保存記録だけから確定できない**。
既存opt2の科学的正否・費用結果を今回改判したわけではない。

## 静的比較で絞れた点

[WP06b](../../../artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json)の9種類のsource basis hashと、[前回保存payload](../../../artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2/PREPARATION.json)のtailの9種類のbasis hashを文字列比較すると、共通は2種類、各側固有は7種類だった。
元の324行のsupport-basis表はhash・support・操作数等の記録であり、全操作列や全component係数ではない。324を実53-bit positive support数とは扱わない。

これはbasis metadataの差である。新しい行列・固有値・演算子を計算した結果ではなく、物理的に有意なtarget差や、許容差内のoperator同値は未認定である。
partition metadataの差だけでtail差を説明したことにもならない。

版差2ファイルについて、`df_partial_s2.py`はexplicit ordered partition対応を追加するが、既定経路のpolicyは維持する。旧/現hash関数が同じpayloadで一致した前回結果は[維持](../o2_a1_q1_q2l_20261106/REPORT_V2.md)する。
`pf_delta_validation.py`の差は検証関数のLD末端条件`>`→`>=`であり、今回の準備経路でその関数は呼ばれない。これらの差が原因と確定したとはしない。

## 次の判断とSTOP

限定探索は完了した。新しい具体的な元payload/環境の所在が得られた場合に追加回収を検討するか、保存DFモデルから新しい一貫したfinite-target契約を定めるかを判断する段階である。
新契約ではraw・層別・CVに同じlaw/target/state/価格policy/誤差条件を与え、旧opt2の価格・biasを無条件に移植しない。

今回、新targetを採択していない。準備・CDF・moment・bias・RNG・state solve・回路build/compile・測定の新規実行は全て0件。
元mainのHEAD/status・既存2,136ファイル、旧A1成果とSTOPを保全した。C1凍結・S2/R1判定を変更しない。詳細科学レビューは未実施。
**mandatory STOP。追加準備計算、Q1/Q2-L、Q3–Q5へ自動進行しない。**
