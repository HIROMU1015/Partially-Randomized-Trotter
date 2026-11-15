# p7/S2・幅hのm1/m2取得（2026-11-15、Codex D）

利用者の固定4軸承認により、保存linear H4/1Å/STO-3G/tol-only DF10、B0I p7/S2・要求幅h=.1、P0除外、元scalar総T一回・native・compiler・測定wrapperでm1/q2、m2/q4のX/Yを取得した。SOURCE be0f81ebdを実H4前に固定し、原作用gateとC guardを通過。新RZはm1各11,333、m2各22,359。保存m0/m3/m4の6軸を再取得せず、元producer/cohortと今回bindingを分けて受け入れた。

raw302d63e90、bound11cb034ef。q1/2/4/8/16の10/26軸受入、対象UNKNOWN0、m5〜m12の16軸は未取得。2 worker、取得wall25.706 s、最大worker RSS401.97 MiB。人工3＋toy回帰2 PASS。[REPORT](../d_pf_p7_s2_h_direct_prices_20261115/REPORT.md)、[HANDOFF_B](../d_pf_p7_s2_h_direct_prices_20261115/HANDOFF_B.md)、26軸価格表へ固定locatorを保存した。

B bec032fcのQ3結果はtriggerとして引用し再計算していない。435,481,090 RZは幅16h/7の達成費用に限る。新fit・長round予測・QPY監査・Q3・shot生成・N/G・winnerは0。5直接点を長q規則や未使用holdoutへ昇格させず、元SOURCE/raw/旧STOPを保全する。ロードマップ差分はHANDOFF_Bへ渡し、本体を編集しない。commit/push・remote照合後mandatory STOP。
