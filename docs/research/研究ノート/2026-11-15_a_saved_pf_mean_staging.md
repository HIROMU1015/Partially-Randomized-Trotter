# 2026-11-15：Full PF・B0I p7 shortlistの保存meanのpacket化（新規計算なし）

## 行ったこと

- Aのcoverage（`c212cee85`）を引き継ぎ、6候補について保存meanをBのpacketの形に束縛した。
- 使ったのは既存のC baseline経路（`I.plan`→`C.prepare`→`C.b_mean`、provider bindingは`produce_signals`と同じ式）だけで、新しいmeanの算出は0である。
- 6候補のconfiguration IDは、CELL_PLAN_V2とBのprovidersの両方と一致した。
- 束縛した内容：
  - raw＝correctedは保存direct mean、B=1。
  - referenceはbba15083のpair。
  - scalarは規則と成分を束縛した（Full PFは定数、B0Iは定数＋p7 tail identity）。

## 結果

- 受入準備が済んだround（S2）：30。
- S2L（Track Aのm3、4件）：現行のpower・norm gate、backend、reference pairの証拠が足りない。既存記録だけでは受入できず、昇格していない。
- S3（新しいmeanが必要）：44。
- 6候補とも未完成である。段階packetは、`require_complete`でもBの`T.bind_signal`でも拒否された。
- 人工の完成fixtureは、Bの既存checkを通った。取得済みroundは変化しなかった。

## 残る課題

- 承認が必要なもの：44件の計算（S2Lを使わない場合は＋4、m0の由来の判断により＋2）。
- 全13roundが揃った後は、Bの変更なしで既存intakeへ接続できる。

## 参照

[REPORT](../a_saved_pf_mean_staging_20261115/REPORT.md)。

mandatory STOP。
