# 証拠制約と今回の判断可能範囲

旧資料の欠損と、科学的にまだ取得していない保証・性能証拠を分ける。**追加の別環境探索・SSH・原本回収・原本代替の再計算は行わない。** 未発見は存在しないという意味ではなく、今回の判断では利用できないという意味である。

## 旧GAPの現在の扱い

|GAP|利用できること／できないこと|課題への影響|今回の扱い|
|---|---|---|---|
|001 UWC|第2環境で指定JSON3件＋theta sweep JSON/CSV4件の候補を発見。FOUND_UNVERIFIED、元runのSOURCE/command/runtime/入力への独立bindingは未完了|RQ-01の旧UWC成功率・改善幅を定量根拠にしない。静的な未実装範囲と他の表現研究の反証は使える|公開報告/statusだけ確認。private原本・fingerprintを取得／転載せず、探索を終える|
|002 旧H4 event/normalization/親exit|O05実event未保存、未丸め172件の来歴、O06親native wait/exitのUNKNOWNを保持|旧H4のevent→価格joinや完全独立監査は主張しない。RQ-03のbasis寄与を補間しない|後のH8層別成果で旧欠損を埋めない。新samplingで代用しない|
|003 M1-S full-H/HD・旧review|旧Gate0 v5の全scope照合はUNDETERMINED。一部旧レビューは二次帰属のみ|RQ-04/06の理論と保存signalの限定記述は可能、全入力の再現済みとはしない|後続finite-stepモデル会計を旧原本照合へ格上げしない|
|004 historical H01入力|PF旧C0.5 source missing/C0.6 reconstruction mismatch。別new-source系列は別identity|RQ-06でhistoricalとnew-sourceのtruthを混合しない|旧復元路線は再開しない|
|005 PF絶対energy tuple|H-chainで対応が閉じた範囲とHF/HCl/prospective indeterminateを分ける|RQ-06の一般energy bridgeの成功例として不足部分を使わない|原tupleが未取得のまま。旧系を足して成功を合算しない|
|006 PR-2旧pilot snapshot|旧S0 hash STOPの入力は新V1–V4/S2/M1/M2の入力と別|RQ-02で旧pilotとM2をsame-input又は独立反復にしない|M1/M2の限定結果は保持。旧原本を探さない|
|007 C1旧partition・旧P5 exit|旧partition差の唯一原因・旧P5処理trial/直接exitは不明。R1/P5Rは別系列|RQ-04/05で旧失敗の0費用・正常完了を仮定しない|旧STOP/failureを保持、後の成功と区別|
|008 stale DF|第1環境で旧無効化37件のhistorical identityを確認。第2環境でも修正canonical Cgs/元anchor rawと元run bindingは未発見|RQ-01/02で旧screeningのrank/bestを科学的baselineへ戻せない|DO NOT USE / stale維持。新fit・screening・原本再生成なし|

根拠：[P2-GAP1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dfdfa6460f9e7d25a9595132a672e8de2a5dc7e/docs/research/pr_algorithm_improvement_landscape_20261023/local_environment_search_20261023/SEARCH_REPORT.md)、[P2-GAP2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/096c293a85a0e756574245673ecf29d5245ddca1/docs/research/pr_algorithm_improvement_landscape_20261023/second_environment_search_20261023/SEARCH_REPORT.md)、[P2-GAP2S](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/096c293a85a0e756574245673ecf29d5245ddca1/docs/research/pr_algorithm_improvement_landscape_20261023/second_environment_search_20261023/STATUS.json)、[BASE-GAPS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/EVIDENCE_GAPS.md)。旧探索報告の「次環境へ引き継ぐ」という提案は、今回の最新ユーザー指示により追加作業にはしない。GAP-002〜007の記述は前段階の固定整理を継承し、新しい全原本探索をしたとはしない。

## 条件不一致と未実施検証

1. Hamiltonian、geometry、basis、rank/policy、prefix順序、L_D、δ/T/q/r/K、state/sector/scalar、compiler/precisionが違う実験を、単一の資源順位や因果効果へ合成しない。synthetic有限P5と分子exponential/energy taskも異なる。
2. G9のmatched-native有限P5はその固定条件で既に比較済み。同じm=5対照を未取得扱いして取り直す案にはしない。一方、C1 H4にはstrict native、共通十分N、deterministic halves、完成coherent estimator、全wrapperのbindingが未取得。
3. C1 CPU表はsetup/online、cache、worker、保存／監査境界に差がある。公平に最適化した算法CPU順位ではない。B3限定pilotも全N古典費用や一般API互換性の証明ではない。
4. 集約compiled RZからbasis/central/deterministic比を逆算しない。加法primitive T、literal相殺モデル、raw logical数、actual compiled wrapperを区別する。T/CX/RZ/depthを根拠なく一単位へ換算しない。
5. analytic sufficient shotsは実行した量子shots／必要最小shotsではない。経験的moment、保存max、SE、engineering ±2SEを母分散・一様range・formal CIへ格上げしない。C_useは明示δ窓の経験的envelope。
6. H8希少RTEは既存の層別有限標本・残余条件付き感度scopeで正式終了。無条件期待費用の保証・残余母費用は未知のまま。今回の改善課題整理を理由に追加sampling/compileを要求しない。根拠：[EV-0077](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md)。
7. H2四方式は既に固定比較完了。H4 readinessは人工準備であり、H4の科学結果を得た証拠ではない。H2/H8のrare-order取得・評価法整備そのものを別の新研究として本作業に取り込まない。
8. 多roundの共通E0/branch/overlap条件、実状態準備、実backend、全費用、独立先行研究差は未確立。H12係数を小系から推定しない。最終総費用評価完了を主張しない。

新しい証拠が必要という記述は、必要性・概念的な識別範囲の説明に限る。新数値、再解析、再ranking、非自明新証明、simulation、sampling、build/compile/synthesis、実装へ自動進行しない。
