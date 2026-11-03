# Track A：technical resource/application noteへの収束判断

2026-10-05。利用者提示の外部GPT原稿review `CLOSE_AS_TECHNICAL_RESOURCE_NOTE`を採用する。
review対象は原稿v0.1・4主図・Supplement・claim auditを収録した
`4c23453c541700c6a41ba71fc5ec9323b53858d6`。
本記録は人による原稿・scope判断であり、科学結果の独立再現、journal acceptance、science authorizationではない。

## 採用する判断と理由

Track Aは新しいHamiltonian simulation法ではなく、既知手法の特定DF-native実装における
matched-accuracyな定量的resource/application case studyとして仕上げる。
原稿で残せる中心成果は、近接discardを補った有限比較、B2内部の要求精度依存、
元精度の固定構成transferと事後精度感度の区別である。
最接近のGüntherらとの重なりを認め、partial randomization・factorized Hamiltonian・
量子化学資源評価・cost×shotsの原理を新規性にしない。

残る弱点は方法論的新規性の性質であり、strong synthesisやH6を一件追加することを完成条件にしない。
既存の証拠でtechnical resource/application noteを完成させ、投稿形態を別に決める。
Track Bのmethod deltaをTrack Aへ統合して新規性を救済する方針も採らない。

## 固定scopeと中心claim

対象はlinear H4の1.00 Å developmentと1.30 Å固定構成transfer、STO-3G、
保存DF rank12・8 system qubits・参照状態、T=0.8、二次DF-prefix PF、canonical finite-RTE。
q=1/2/4/8、δ=T/q=0.8/0.4/0.2/0.1と登録r/K、対称軸Hoeffding十分shot規則、
Qiskit1.3.0・opt1・rz/sx/x/cx・seed17・指定wrapperを維持する。
primaryは状態準備なしのshot-weighted compiled RZ、他5指標と共通仮想P感度はsecondaryである。

- C1：development登録218構成、ε=0.05、P=0。近接discard L_D4/5を補っても、
  B2 L_D3 q1 r4 K2のprimary点推定は最良登録discard B0 L_D5 q1より43.07%低い。
  全決定論法への優位、厳密winnerではない。
- C2：同じ保存値による一つのε=0.005〜0.1感度解析。302表示点では点最小methodはB2のまま、
  内部設定がq4→q2→q1へ変わる。302独立実験や連続精度の厳密切替根ではない。
- C3：developmentから結果前に固定した5構成の1.30 Åへの移送は元ε=0.05で支持された。
  同じ保存値の事後精度感度では厳しい側でB2固定構成が現行shot規則に不適格となる。
  held-out method optimumや新しいblind evidenceではない。

補助claim、証拠階層と非claimは[固定claim/evidence map](track_a_post_pm2_claim_evidence_map.md)を維持する。
pure discard/PF成分、実準備費用、strong synthesis等の未評価量を埋めない。
finite-time coherent-signalの比較をchemical-accuracy energy/QPE/RPE総資源へ拡張しない。

## v0.2の編集範囲

[本文v0.2](../manuscripts/track_a_resource_study_v0_2.md)では、Abstractを
実施内容・観測結果・scopeの三段構成にし、Introduction末尾へ有限候補resource map、
baseline stress test、applicability analysisの三寄与を明示する。
Related WorkではGüntherらとの差分を限定し、Discussionをscientific limitationsとscope choicesに分ける。
stage名、内部status、commitは[Supplement](../manuscripts/track_a_resource_study_supplement_v0_2.md)へ寄せる。

v0.1、科学値、4主図と図S1、表示CSV、Methodsの式、候補、閾値、原判定は保存する。
補足のPM-2 launch commit転記誤りだけは、元manifestと実在blobを確認してv0.2で明示訂正する。
[claim audit](../manuscripts/track_a_resource_study_claim_audit_v0_2.md)でC1〜C3の強さを各章間で確認し、
[別editorial audit](../../artifacts/resource_applicability/track_a_manuscript_v0_2/2026-10-05/editorial_audit.json)で不変性を追跡する。
文章作成skillに沿って寄与・限定・再現性情報を分離したが、科学設計や元結果は変えない。

## STOPと次の判断

新signal、trajectory、compile、分子計算、quantum shots、GPUは0。
NPZ/NPY/pickle、runtime/cache/checkpoint/registryを開かず、既存科学runner/testsも実行しない。
旧M1/M2/PM-0/PM-1/PM-2 result/status/runner manifest・global validation manifestを変更しない。
原科学結果の`research_decision=null`をこの編集判断で上書きせず、
[別のmachine-readable判断記録](track_a_technical_resource_note_scope_decision_20261005.json)へ分離する。

次はv0.2を材料に投稿形態・投稿先、著者・所属、投稿要領への整形を決める。
投稿可能性reviewを採択保証としない。追加科学計算、PM-3、追加trajectory、別geometry/分子、
strong synthesis、高次PF、energy/RPE、Track B統合は認可しない。
commit/push・投稿もこの判断記録だけで自動実行しない。
