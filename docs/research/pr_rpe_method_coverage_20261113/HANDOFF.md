# A/C → B：4方式のRMSE比較受入仕様

今回の技術変更は[REPORT](REPORT.md)、保存metadataは[COVERAGE](COVERAGE.json)、再利用元は[SOURCE_MAP](SOURCE_MAP.md)。実N/C/Gは未実行。

## 候補・N/C

既存APIは`pr_rpe_rmse_controller.generate/assess/cost/select`。`method`は従来の`PF/PR/FULL_H_RTE`を維持し、`comparison_method(spec)`とPF candidateの`comparison_method`で`FULL_PF/TRUNCATED_PF/PR/FULL_H_RTE`を区別する。

PFの`split.deterministic_scope`は`FULL_H`又は`TRUNCATED_H`。truncatedには`retained_hamiltonian={source, construction, prefix, target_hamiltonian}`を必須とし、targetは元Hのbindingと同一。κ/K/RTE mappingは付けない。A adapterは元B0/p7をTRUNCATED_Hへ、FULLをFULL_Hへ、PRの決定論側をPREFIXへ写す。元A科学SOURCE・physical IDは保存する。

PF candidateにtarget/retained-Hを追加し、schemaを`PR_RPE_RMSE_V2`へ更新したため、**B configuration/candidate IDは基点から変わる**。同じ旧IDに異なるidentityを載せない。過去のN/CやA preflight出力へ新IDを付け直さない。A/Cは新B generateの`identity`と`law_binding`全体をechoし、元physical ID/SOURCEとの対応証拠を別に残す。A物理schedule・fixed grid・H6早期例外自体は不変。

N packetは既存の`configuration_id, policy, source, status, evidence_kind, rounds, rmse_evidence`。十分Nは正のPython int、`N_X/N_Y`。nominal/旧HP Nを代入しない。科学証拠にはgeometry/basis/state/prior/decoder、全round対象のRMSE契約とbound、採択状況が必要。truncated PFはさらに下記を要求する。

```python
rmse_evidence.update(
    truncation_status='QUALIFIED', covers_truncation_error=True,
    target_hamiltonian=configuration['spec']['binding']['hamiltonian'],
    retained_hamiltonian=configuration['spec']['split']['retained_hamiltonian'],
)
```

この宣言だけで科学的証明は成立しない。source-boundの全体RMSE boundが切捨て誤差を含む根拠をAが供給し、ユーザーが科学条件を判断する。数値boundはBが生成しない。

固定A510向け`n_from_a`は従来通り十分N=nullだけを受け取り、`MeanPriceLookup`はA PF/PR用のまま。将来の適格NやFull-H RTEは、新SOURCEの評価器がBの既存供給record形式を満たした後に接続する。native Full-H RTEはevent_count/R_outer=Noneを明示するsource-bound mapperのmetadataを受け取れるが、物理samplerやNモデルを呼ばない。未知mapperの候補はNOT_EVALUATEDで、Gへ進まない。

Cは既存schemaの`identity/law_binding/axis/source/cohort/classification/expectation/mean/model/uncertainty`。測定・compiler・full-wrapper/P0/unitsもidentityと一致させる。`quantity`を指定する場合は`FULL_WRAPPER_MEAN_COMPILED_PRICE`のみ受入可能。旧compiled recordの未指定は既存意味を維持する。

- DIRECT deterministic、DIRECT finite sample、MODELは区別する。有限cohortは明示`price_policy='finite_cohort_scenario'`の条件付き比較だけ。
- MODELは既存のcandidate/domain/binding/compiler/q_variable/validation/coverage資格を要求する。C1のfalseを保持する。
- `price_status=OUTSIDE_DOMAIN/FAILED/MISSING/NOT_EVALUATED/UNQUALIFIED_MODEL`を有効価格として受け取らない。nullは取得済み価格にしない。
- S0aのpath S、構造proxy、局所期待、単一回路価格を全law平均Cへrelabelしない。未知のsystematic/covariance/CIはnull。

保存C1のnominalなpaired mean-price covarianceは元INTAKEに存在し、coverageにはその存在と参照、paired/calibration groupを残す。全schedule・共有校正・cross-candidateの未知共分散まで既知とは扱わず、Bで新しいCI計算は行っていない。

`assess()`が返す`coverage`は各m/X/YについてidentityとN/Cの`availability/value_status/qualification/source/reasons`を持つ。Cは種類・cohort・期待値区分・不確かさも保つ。存在AVAILABLEと資格AVAILABLEは別欄。assessはGを計算せず、cost/selectも明示呼出しのみ。

## 方法別の探索範囲

`select`の既定coverageは4方式。限定比較には`required_methods`を明示できるが、欠けた方式を一般比較の完了と表示しない。固定formulaが異なるmenuを比較する場合のみ、次の明示引数を使う。

```python
formula_menu = dict(source=menu_source, by_method={
    'FULL_PF': ['2nd'], 'TRUNCATED_PF': ['2nd'], 'PR': ['4th(m5_best)'],
})
selection = select(configurations, supplied_costs, formula_menu=formula_menu)
```

これは人工test例であり正式なPF baseline採択ではない。指定がなければ従来のfixed-formula条件。finite-K/κ定義/mapperは各方式内で固定し、PRとFull-H RTEに同一mapperを強制しない。全H/state/decomposition/SOURCE、h、RMSE target/policy/prior/decoder、compiler/control/wrapper/Pと価格policyを共有する。Mが異なる候補は各々全scheduleのRMSE資格が必要で、`schedule_menu`にMと全Tを保存する。

## 次の成果と担当

| 担当 | 次に供給するもの | Bが受け入れない代替 |
|---|---|---|
| A | 新B identityへのpreflight pin追従。2時点pilotは診断packetとしてsource/対象round/未資格を保持 | 2時点のPASS、nominal N又は旧HP Nを全round RMSE十分Nにすること |
| A | Full PF、元H対象truncated誤差、Full-H RTEのN評価経路と全round誤差契約 | B0をFull PF、B3をFull-H RTEにすること |
| C | 既存C1意味のまま返す平均compiled価格。将来は新B候補・lawと対応するfull-wrapper mean/domain証拠 | S0a path Sから平均CやMODEL資格trueを作ること |
| B | 新しい固定成果のmetadata/source照合と、必要最小限のadapter更新 | 不足N/Cの0埋め、固定R又は小qモデルによる新R/長q補完 |
| ユーザー | 正式RMSE/error契約・Full PF formula/幅menu・対象truncation範囲・純RTElawと有限科学scopeの判断 | 技術test PASSだけによる科学条件採択 |

## 最小の次科学scope案（未承認・未実行）

1. まずユーザーが1系・1外側条件・1つの固定(M,共通κ)と比較対象PFを決める。H4 B0/p7とFull PFは別候補。Pauli/DFも同じ数値stateとみなさない。formal search範囲は未採択。
2. Aの既存2時点案を再利用する。H4 m0/m6又はH6 m4/m8は保存mean/reference照合の参考点であり、今回採択しない。完全identity/law/time/source一致なら保存mean/referenceの許可付きRMSE再解析だけ、不一致時だけ当該2時点の不足mean/内部対照を取得する。局所pilotであり全round資格は与えない。
3. CのS0a案は固定4セル×X/Yの8 path構造照合に限る別診断。これはN/C計画の価格取得数に含めない。新sampling/compile/mean-price資格へ自動拡張しない。
4. 総費用比較用のCはAと同じ物理候補だけを対象とする。C1の小qを長qへ移さず、law平均・希少event・共有校正の不確かさを含む方式を先に定める。保存directが正確に適合する場合はcompileを追加しない。不足wrapper価格又は独立validationだけを具体的な有限planで承認対象とする。
5. Full-H RTEを入れる場合は既存DF/Pauli producerをA/Cが再利用し、全非scalar項・一体項・λ/law・finite-K・event単位を固定する。Bのdraft event metadataは物理的検証の代用にしない。
6. 局所診断の後、正式に選んだ有限比較menuの不足全round N/Cと全体RMSE/decoder証拠を別scopeで取得する。完全coverageと価格policy・cross-candidate covariance等が揃って初めて実G/共通κ・M/外側比較を別途承認対象にする。現在のG/winnerはnull。

科学再解析、mean/N/RMSE、S再計数、sampling/compile/fit、実G/選択はそれぞれ具体的SOURCE/候補/有限上限を定めた明示許可が必要。承認証明書や新grant発行機構は不要。**今回の公開確認後mandatory STOP。**
