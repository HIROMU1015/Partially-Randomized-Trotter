# H4 決定論的PFの登録集合と取得範囲

2026-10-19 JST。Phase 0 metadata監査。新科学計算・raw標本の統計集計・費用再会計は0。

正本は2588bb1ff8a03120050fdaa67b9cd30739131becのH4 native D plan/result/receipt。
原本のprivate `signals.json`（96件）は実際に取得し、登録ID集合と照合した。
public resultには全96件のsignal本体が含まれないため、public resultだけで確認したとは表記しない。
private原本のSHA、保存copy、原SOURCE79758d28d89ccaf2f942dc10ad724e70cf513089を別記録した。

## 登録と測定状態

|集合|登録|直接実測済み|費用未取得|
|---|---:|---:|---:|
|打切り B0/B0I|96|8（16軸）|88|
|full-rank B1|8|8（16軸）|0|
|全決定論集合|104|16（32軸）|88|

打切り登録はB0/B0I × prefix 2/3/4/6/9/11 × q1/2/4/8 × S2/S4。
fullはprefix12 × q1/2/4/8 × S2/S4。旧native固定16は「決定論8＋PR8」であり、
強化決定論16は「full8＋追加打切り8」である。両者の16を混同しない。

取得済み打切り8候補：B0I_p4_q1_2nd、B0I_p9_q1_2nd、B0I_p9_q2_2nd、
B0I_p9_q2_4th、B0I_p9_q4_2nd、B0I_p9_q4_4th、B0_p9_q2_2nd、B0_p9_q4_2nd。
各軸の整数RZ・compiler hash・実回路fingerprintは機械可読表に転記した。

## 必要compileと除外の論理

固定した通常signed／経験的非相殺 × 5精度 × allowance倍率1の10 endpointにおいて、
保存済み `cost_acquisition_eligible` が少なくとも1つtrueの打切り候補は63。
このうち8件は測定済みなので、新規対象は55候補・110軸。native controlは最大220 probe。
33候補は全10 endpointで明示的なINELIGIBLE。未解決statusを不適格へ変換していない。

除外の意味を正確に限定する。各endpointの許容集合を保存済み適格性predicateで定義すると、
全endpointでfalseの候補はその集合の最小化に参加しない。この論理は費用proxy・2%経験則・
費用下界を必要としない。全104候補の行を残し、不適格行の費用をnullで保持する。
保存bias/allowance自体は経験的である。従ってこれは固定した経験的評価契約内の除外であり、
真のbiasの厳密証明、物理的不能、大域的劣位の証明ではない。

全96の単価そのものを揃える別目的なら未取得88候補・176軸が必要になる。
本D案は、登録集合内の今回10 endpointの最良を決めるための110軸案であり、
「96候補すべての直接compile完了」とは報告しない。176軸へ自動拡張しない。

## 保存値と来歴の索引

- [104候補の費用・指紋・status](../../../artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/deterministic_coverage.csv)
- [保存signal・bias成分・allowance](../../../artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/saved_signal_catalog.json)
- [104候補×10 endpointの保存適格性・shot](../../../artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/saved_endpoints.json)
- [取得済み軸別metrics](../../../artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/saved_measurements.json)
- [55 compile対象・33除外ID](../../../artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/D_tasks.json)
- [commit:path/SHA/bytesの取得台帳](../../../artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/source_inventory.json)

full8の原測定は0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41由来。
元private metricsのSHAはe9a97c090f4788bba8174bd5d14510a640d1d0ca9a6db3c212e84a7d7a429776。
同じCSV内のordinary control行は除外し、symmetric_directional行だけを使用した。
追加8は2588bb1f…のD_result.json.gzおよびその完了receiptへ結合。
raw D resultのSHAと公開receiptのuncompressed SHAが一致した。

## 再利用条件と別系列

input Hamiltonian/state/snapshot、DF rank/順序/scalar、T、control、wrapper、軸、
compiler設定と対応するSOURCEが一致する保存実測だけを再利用する。
回路fingerprintが異なるB0/B0I、異なる軸やqの単価を単に等しいという経験則で流用しない。
新55候補について、未生成回路のfingerprint一致をPhase 0で確認したとは主張しない。
新旧compiler outputのbit同一性は元資料でも未検証。その限界を維持する。

98cdb35c…のgeometry実測は0.70/0.80/0.90/1.10/1.40/1.60 Å、ordinary control、
S2/structured Gaussian系列であり、今回1.00 Å native symmetric-directionalへ加算しない。
97d25cc4…のT=.8/legacy代表panelは保存測定の再利用であり、追加の独立測定候補に数えない。
tol-only rank10と新T=.4/1.6は別入力・別条件として保持する。

Phase 0完了時mandatory STOP。次の担当：ユーザーがDの110軸案を審査する。
