# Track B DF三次補正：現generic whole検証系列の区切りと証拠索引

2026-10-12 JST。利用者が採択した[GPT科学レビュー原本](../../research/track_b_df_correction_et_scientific_review_20261012.md)を既存記録へ反映する文書である。
Codexによる新しい科学レビュー・アルゴリズム採択・計算結果ではない。

**現generic whole実装を新規PRアルゴリズムの主線として拡張しない。取得済み成果を保存して、この検証系列を区切る。mandatory STOP。**
Track B全体の終了、whole方式一般の不可能性、Bの全PR資源優位を意味しない。
今回の文書整理だけを理由に、同じ科学レビューを反復する必要はない。

## 1. 原本・結果の来歴

| 資料／実行 | 固定identity | 意味 |
| --- | --- | --- |
| GPT科学レビュー原本 | 39,039 bytes、SHA256 `00f19797037936c2c4e7ac2bf41610fec636113cca3df9ab3c91a87d40276076` | 利用者から受領した研究判断。全文・原本内の公開前記述を変更せず保存 |
| 原本公開commit | `1d39fc687bf328c8f8b5a9c5f7251950f0fee814` | 原本と受領provenanceだけのcommit。remote再取得一致 |
| E/T結果公開commit | `901cafcd594b0399b8638ee1d4ea612bb56fc69e` | 今回の判断が参照する保存済み結果。独立再現ではない |
| 元科学source S7 | `74682c40d0cdcca92d244a9edf4bb2852361eef5` | native/Tを実行した凍結source |
| 元authorization A7 | `84562f8c5af9249a902662324200456498a83d15` | S7の直接子、当時の一回実行認可 |
| 元result R7 | `ad4b4a334171cc39053c0804876a63c93093bfa3` | 原terminalは`TECHNICAL_INCONCLUSIVE`のまま |
| 復元source S9 | `3afef9e418a8d2173a2780f5534404390bfbe6fb` | 同じ元post-T ASTを保存値だけで処理 |
| 復元authorization A9 | `eea9c497421c5dcfc541dcaa508f607112dbd0d5` | saved-only処理の認可。新科学実行の認可ではない |
| 復元status | `SAVED_E_T_ACCOUNTING_COMPLETE` | 元runの再分類を伴わない別の保存値集計 |

元science marker SHA256は`28e8d00ab4d8703e3693c29d946011c27c17cab2ddb3650d783ddeb8197b9dc2`、
復元marker SHA256は`452325155c8f5166398753b36ac852863b11e3a976210ffabb1a41bd3f25bf7d`。
logical accountingは1,135,179 bytes、SHA256 `b9a6d6aa684f4df4808c1b59d33782cd83f36afad1d510dec5c89a758b0f6ad4`。
元会計JSONは保存されなかったため、失われた元bytesとの一致を証拠として主張しない。

## 2. 設計から判断までの証拠索引

| 段階 | 一次資料 | 証拠／限定 |
| --- | --- | --- |
| 原設計 | [DF少数軌道・希薄三次補正設計](../../research/track_b_df_sparse_correction_design_review_20261011.md) | 監査入力。設計書だけを実装・資源優位の証拠にしない |
| 独立監査 | [A–F監査](df_sparse_correction_independent_audit_20261011.md)、[claim別先行研究比較](df_sparse_correction_prior_art_comparison_20261011.md) | BCH／理想平均を検証。最大6軌道は指定primitiveクラスの条件付き結果。production DF access・総資源優位は未確定 |
| 比較への限定 | [監査後GPT判断](../../research/track_b_df_sparse_correction_post_audit_scientific_review_20261011.md) | wholeと局所Pauliに同じ局所情報を与える。局所性をAだけの利点にしない |
| 結果前仕様 | [比較契約v2](df_correction_comparison_preregistration_v2_20261011.md)、[E/T再開準備](df_correction_et_resumption_preparation_20261012.md)、[v7契約](../../../artifacts/track_b_df_correction_et_phase_preparation/2026-10-12/v7/contract_v7.json) | CDとEを別系列に固定。native最大22／T最大64。未使用上限を追加計算義務にしない |
| 元実行・保存停止 | [v7停止記録](df_correction_et_v7_storage_stop_20261012.md)、[原630ファイルinventory](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v4/original_run_inventory.json) | native22／T key39を取得後、512 KiB保存制約で停止。原terminal・結果を保持 |
| 保存値集計 | [完了handoff](df_correction_et_completed_saved_accounting_and_gpt_handoff_20261012.md)、[exact 4行](../../../artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/comparison_rows.json)、[全取得索引](../../../artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/evidence_manifest.json) | backend33／Clifford fast path6。全Aのactual Tはunknown。原runの成功へ書き換えない |
| 科学的判断 | [最新GPTレビュー](../../research/track_b_df_correction_et_scientific_review_20261012.md) | 3条件の局所固定計画分離を認め、現generic wholeへの追加投資を優先しない。全PR非優位の証明とは区別 |
| 今回の文書反映 | [closure manifest](../../../artifacts/track_b_df_correction_et_scientific_closure/2026-10-12/closure_manifest.json) | 原本identity、保護対象、文書差分、公開確認。新科学計算0 |

過去資料の「未評価」「次は比較」は当時の状態として保存する。現在の系列状態は本記録と最新GPTレビューを参照する。
外部文献の主張は既存監査／GPTレビューの確認範囲を継承し、今回新たに全文監査したとは扱わない。

## 3. 固定した比較scopeと4行

Eは保存済みsynthetic r3／r6、rank2×3、seed7101107の元生成順、primitive signs(+1,-1,+1)、
theta=1/32、kappa=4、u=1/128、uncontrolled／MSB-controlledの局所centreである。
Qiskit1.3.0、rz/sx/x/cx、optimization0、compiler seed20261011。
native数値2e-7／synthesis8e-7／full1e-6、pygridsynth2.0.0、strict phaseを維持した。
分子geometry・production DF rank・PR splitを登録した実験ではない。

| active modes | controlled | A固定加法計画completion下界 | A actual total T | B完全計画worst-event T | B条件付き期待T | A未知RZ出現数 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 3 | false | 576 | 未確定 | 112 | 112 | 57 |
| 3 | true | 692 | 未確定 | 236 | 236 | 179 |
| 6 | false | 576 | 未確定 | 116 | 116 | 5,519 |
| 6 | true | 未確定 | 未確定 | 232 | 232 | 11,622 |

上表は保存されたexact 4行の転記であり、新たな費用計算・最適化・採点ではない。
取得済みsequenceを保持し、gate間cancellationを行わない固定加法計画において、最初の3条件でA下界>B上界が成立する。
最適compiler・whole方式一般に対する下界ではない。4行目はunsupported scalar phaseによりA下界も欠測であり、非優位・優位の証拠へ変換しない。

CD6taskは別の3-mode／3-block、係数0.45／0.35／0.25、seed6121107、t=1、delta={1/8,1/16,1/32}、
gamma=1.25、s=1、paired cutoff4/P5、complex epsilon=1e-4、alpha=0.05の保存結果である。
CDの補正頻度にEの別fixtureの単価を掛けて、実測PR総費用を生成しない。

## 4. claim・証拠・未確定事項

| 記録する内容 | 支持の範囲 | 越えない限定 |
| --- | --- | --- |
| 条件付き四次精度 | 固定順序・有界入力・正時間・独立draw・P5以上・digital/native予算等、元独立監査の前提 | 個々のsampleが四次PF、一般channel保証、energy精度保証を意味しない |
| 最大6軌道の支持 | number-conserving Gaussian基底の高々2軌道parity積primitiveに由来する三重交換子 | 任意DF registry、production importer、認証付きrank切捨ての完成ではない。同じ局所情報をBも利用可能 |
| Aの追加weight1 | 受理whole指数回転の補正ensemble | 元RTE normalizationは残る。shot負担1やPR総資源優位を意味しない |
| 局所T費用の3条件分離 | 固定generic分解・取得済みsequenceを保持する加法plan | 同じ最終信号精度でのPR勝敗や高速化率ではない |
| A/Bの平均・biasの違い | 理想受理Aは指数回転、Bはweighted線形補正 | finite-stepで同じ平均ではない。gate合成精度とlinearization biasを混同しない |
| 復元値を証拠に使用 | GPTが限定scopeでの使用を支持。原AST・保存入力・独立count照合に基づく | 元v7成功、独立再現、無条件exact certificateへの読み替えを行わない |
| 総費用・新規性 | 未確定のまま保存 | Bを勝った新手法として採択しない。独立論文の主貢献確立とは扱わない |

## 5. 採択済みの研究運営上の判断と次の境界

現generic whole検証系列をここで区切る。残り全T、r6 controlled phaseだけの完成、別compiler探索、
同入力PR総費用への接続を自動継続しない。この決定はGPTレビューを利用者が採択したもので、Codexの新しい科学判断ではない。
数学的反例や全PR非優位を証明したための停止ではなく、現研究目標に対する追加投資を優先しない判断である。

今回認可された文書反映・取得性確認は完了後STOPする。同じ範囲の追加GPTレビューは求めない。
Track Bの新候補を進める場合は、具体的な新機構・native access・保証・取得計算量等の差に基づく研究方針をGPT側で決める。
その新しい目的・比較条件・scopeの採択と、新しい科学実行の明示認可があるまで、Codexは次系列の実装・実行を開始しない。

今回、新しい科学実行source・runner・contract・authorization・science markerは作成しない。
既存source／contract／authorization／raw／marker／STOPを保持し、レビュー原本・文書・metadataのみを公開する。
追加matrix／compile／T合成／CD再計算／sampling／LP／DF／molecule／GPU／G10／RA-D0は0。
