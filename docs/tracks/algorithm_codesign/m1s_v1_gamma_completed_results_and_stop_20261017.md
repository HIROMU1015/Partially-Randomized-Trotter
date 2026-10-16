# M1-S γ依存V1：解析完了・技術修正来歴・STOP

2026-10-17。承認済み保存信号解析は完了した。statusは`GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT`、failure=null。72方式行を評価し、36保存zero controlは再解析せず保持、48対照すべてに分類を保存した。新たな科学的GOや研究方針の採択は行わない。

## 対象と固定条件

H4 linear1.00 Å／STO-3G／8 modes／DF rank12、G3/G6/G9、保存δ=1/8,1/16 Ha^-1、R=2,4、T=1,4 Ha^-1。shared ±／ordinary／independent ±を同じ十分shot設計で比較するTask Aの事後保存信号解析である。raw無補正の円周位相を扱い、Task B多round energy RPEは対象外。

`PER_METHOD_ROW`、α=0.05、fresh-IID／Hoeffding／X/Y等配分、p=128、保存数値の条件付きengineering coverageを維持した。各方式行95%は72行同時保証ではない。γは固定せず共通領域 `max(b_sh,b_cmp)<γ<π/2`、双方の正visibility下界を要求する。`M_m(γ)=A_m_lower sin(γ−b_m_upper)`の連続優劣と境界を評価した。保存raw-H総biasにはモデル差を再加算しない。

## 保存された分類

| 分割 | 対照数 | 解析上の分類 | 読み方 |
|---|---:|---|---|
| G3 | 16 | UNIQUE_TRADEOFF_BOUNDARY | 全対照で内部境界を囲い込んだ。境界区間より下はshared marginが低く、上は高い。区間内は順位未解像。 |
| G6 | 16 | UNIQUE_TRADEOFF_BOUNDARY | 全対照で同じ向きのtradeoff。各cellの共通領域・境界はraw JSON／CSV参照。 |
| G9 | 16 | ALL_COMMON_GAMMA_SHARED_HIGHER | 採択済み保守的margin設計の形式的順序。元の絶対visibility差は未解像のため`ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE`を全件維持。科学的な改善の認定ではない。 |

この結果は保存された条件付き半径から構成した同じ十分設計の比較である。連続marginが大きくてもceil後の整数shot数が同じ場合がある。strictなshot削減は保存された記号条件 `ceil(K/M_sh²)<ceil(K/M_cmp²)`を別途満たす必要がある。今回、絶対shot数・ceil削減数・新γ grid・native回路費用・RPE simulation・総資源・科学的GOは計算していない。有限step signalやHamiltonianも再計算していない。

## 不具合修正と実行来歴

旧試行01（S=f2a5611c、C=9aafa8b7、A=b0b48836）は、Python3.11.0rc1に存在しない`sys.set_int_max_str_digits`呼出しで科学入力decode前に停止した。旧失敗、null/未評価結果、消費済みmarkerを変更しない。

ユーザー最新指示「しっかり完了するまで行ってよい　途中で止まった場合は修正して行って」を、技術的修正と別試行による完了までの実施承認として記録した。新namespaceの互換helperではAPIが存在する場合だけ25000桁を設定する。数学カーネル・解析・supervisorは旧SOURCEと同じbytes。採択済み科学契約から変更したmachineフィールドは`source_identity`だけであり、科学条件・上限を緩和していない。

- 新SOURCE S2：`677dd6957e0e1913b70ca5f999458835ae8a2e23`
- 再固定契約 C2：`2cb5160d9051a6568b8acc0cfb29fc169bdfca09`
- 試行02専用authority-only子 A2：`161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c`
- 契約SHA256：`e877ada90db46cc6c7f525b118aaa649e769e1160d93e040d8e55bb76cb558f4`
- 結果SHA256：`bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d`
- 保存入力commit：`025c5a90888b02b73d2815fbb10d6bd1d2823602`、SHA256：`ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`

今回の新SOURCE起動は1回、成功、当該authorityのretryは0。旧試行を含む総起動は2回である。新旧authority・SOURCE・結果・markerは独立し、旧one-shotを復活させていない。

## 検査・資源・保証範囲

登録外技術テスト38件PASS。pinned Pythonでoptional API有無、最大65536-bit整数の往復変換、実workerのresource設定とatomic checkpointをmetadata stubで検査した。数学SOURCE同一性など静的23件、契約・入力・SOURCE照合48件もPASS。実行後の保存結果schema・全grid・boundary区間のdomain整合・provenance・resource上限は515件PASS、保全監査は11件PASS。旧非index7681ファイル・親の追跡1284ファイル・既存208worktreeのHEADを保全。詳細は同directoryの監査JSONを正本とする。これらはローカル記録であり、独立したH4境界数値検証やCI再現の代替ではない。

CPU1 worker、GPUなし、memory2GiB、result/log各16MiB、CPU160秒／wall320秒を維持。worker CPUは約0.319秒、RSS最大16084 KiB、supervisor wall約0.402秒、外部wall0.48秒、外部最大RSS23040 KiB。3335 bound stepsで上限1000000未満。実測値はこの実行の記録であり一般runtime保証ではない。aggregate virtual memoryのparent reserveとchild capは合計2GiB。中断・失敗なし。

IEEE／BLAS／trig等について元の保存signal半径が依存する未検証条件は残る。今回の有理数区間算術や技術テストで元signal、保存oracleの厳密固有状態性、platform、物理測定を認証したことにはならない。証拠区分は`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。

## 証拠と残る判断

[新契約・実施記録](m1s_v1_gamma_runtime_repair_and_completion_contract_20261017.md)、[採択済み条件を引き継ぐmachine契約](../../../artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/execution_contract_adopted.json)、[raw result](../../../artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json)、[48対照CSV](../../../artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/gamma_comparisons.csv)、[108行の保存bounds](../../../artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/method_bounds.csv)、[evidence index](../../../artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/evidence_index.json)、[publication receipt](../../../artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/publication_receipt.json)を参照する。

未解決の技術障害はない。残るのは、このγ依存tradeoffがTask Aの目標精度に実用的か、整数shot数・native費用・Task Bへの次段階をどう定めるかという科学的判断である。今回はその判断を代行しない。必要な科学レビューの開始や次期条件の採択、新規解析実行はそれぞれユーザーの判断を要する。追加計算へ自動移行せず、mandatory STOP。
