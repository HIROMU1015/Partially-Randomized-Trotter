# Track B 方法ノートv1.1：変更・証拠・未確立事項・handoff

2026-10-20 JST。基点 `78cb65be373b0808641b5a5e44d9b69913bf84e9`。
**既存成果の原稿化。新科学レビュー・新証明・科学コード実行・保存値再解析ではない。**

本文：[方法ノートv1.1](../../manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md)。
変更元：[v1.0固定版][V10]。v1.0は変更しない。

## v1.0からの主要変更と理由

| v1.0 | v1.1 | 変更の理由・保持した内容 |
|---|---|---|
| 表題・要旨、§1 | G6を核とする構成・保証・実装・資源条件の要旨と証拠層 | G10負結果だけを中心statementにせず、後続F/W/Pilotまで含む方法仕様にする。§1では有限構成と別taskのD1–D3引用を明確化。独立新規性・一般優位は未認定 |
| §§2–3 | 数式(1)–(15)を保持、§3.4を追加 | 既存の集約・正値性・Green・pairing・envelope証明を保ち、zero-fill二次momentとaccepted量子実行費用の単位を明記 |
| §4 | 数式(16)–(19)を保持、§§4.4–4.5を追加 | F1の全到達parentへの提供precision十分条件、F2の標準bit費用、F3接続済み事項を反映。既に完成した仕様を未解決へ戻さない |
| §5・付録A | P5式(20)–(27)、群構築・selector留保を保持 | 低次数の強い対照とgeneral local器との差を省略しない。新しいP5証明・実行なし |
| §§6–8 | G10契約・全17行・既存図・式(28)–(37)を保持 | 条件付き資源上の負結果、hard/expected/tail、sampling-only下界を失わない。歴史的契約であることを明記 |
| 未収録 | §§9–10：G7/G8正結果、D1–D3、P_U/W2b/W2c | 比較task・provider・confidenceの違いを残し、正負双方を説明。新N・価格・順位・交点は計算しない |
| 未収録 | §11：16,896件Pilot | 実API・古典費用の固定観測を反映し、compact、未分離費用、順序/共有負荷/IID制約を明記 |
| §9 関連研究 | §12へ拡張、PTSCを追加 | 既知部品と全pipeline接続を区別し、直接確認範囲・未確認文献・PDF頁画像取得失敗を明記 |
| 旧§10/付録Dの結論 | §§13–15、本文と本表へ再構成 | BF/BM/SP・Parent・DF・M1-Sのtaskを分離。旧G10 endpoint/STOPは付録Bと原文参照で保持し、Track B全体の自動終了や再開としない |
| 付録B/C | 既存来歴・正確表・監査の所在を保持 | 旧失敗・marker・SOURCE・原結果を残す。旧checkerコマンドは履歴であり今回実行しない |

削除した科学結果はない。旧結論の適用域を広げる箇所は、本文で条件・出典を示す。図や数値は既存の再掲で、新作図・再集計・丸め値からの差/比/順位の判定は行わない。

## 主張と証拠の対応

下表は原稿で使う既存主張の所在である。一般証明、固定SOURCEの対応、既済有限検査、観測値を区別し、Codex自身の編集確認を外部独立レビューと呼ばない。

| ID / 本文 | 既存主張 | 固定根拠・SOURCE / 証拠の型 | 限界 |
|---|---|---|---|
| C01 §§1–2 | 同じ有限Taylor meanの全形式return集約 | [E1]§2、[KERNEL]のmass/word_series、[MATRIX] M1 | channel/exponential一致ではない |
| C02 §3.1 | Green再帰で局所係数を取得 | [E1]§5、[E2]§2、[KERNEL]、M2–M3 | 既知free-product式の特殊化。native取得費用を含まない |
| C03 §2.1 | short-step正値性 | [E1]§§3–4、M4 | 正p、0<x≤1、odd m。fixtureを証明としない |
| C04 §2.2 | parent/childがmeanを保存 | [E1]§6、[KERNEL] Parent、M5 | 単一Qi回転、位相・語順保持 |
| C05 §§3.2–3.3 | 全B_full事前取得なしの生成、U上界 | [E1]§§7–8、[GENERATOR] sample/upper_normalizer、M6–M8 | raw非reducedをzeroにし、reduce先へ再配分しない |
| C06 §§3.2–3.4 | E[X²]=B_ord B_full、全N、期待NZ | [E1]§8、M7、[W2C] | momentは古典trial単位。digital/ideal/実現数/上限を分ける |
| C07 §4.1 | πW̃=α̃、近似meanのbias/range | [E2]§§3–5、[KERNEL] rational_event_weight、M9–M10 | 元irrational meanへのexact一致でない |
| C08 §4.4 | P0–P5の事前十分条件・Full F3 | [F1]§§2–7、[F3]、[F3_PRECISION]、[F3_GENERATOR] | 全入力固定bit成功、最小precision、物理IID/provider保証でない |
| C09 §§4.3/4.5 | rational/bit/selector/全N会計 | [E2]§2、[F2]§§1–9、SOURCEのsort/走査/再sum | CPU秒・一般classical優位でない |
| C10 §§5–6 | P5少数群とcontrolled条件 | [E3]、[PROVIDER]、M12 | L²+1群は全online O(L)を意味しない。無料controlを仮定しない |
| C11 §9.1 | G7固定P5の条件付き期待T利益 | [G7]の保存値・affine表 | exact CQ/helper/取得Rz会計、登録3対照。physical provider未取得 |
| C12 §9.1 | G8のon-demand接続 | [G8]の契約・実行報告 | 固定trace、小support、同backend。全角・独立再現でない |
| C13 §§6–8/9.1 | G9/G10同次数closed対照・native限界 | [G9][E4]、[G10_TABLE][G10_ARITHMETIC] | 固定provider/合成/policy。新ランキングでない |
| C14 §8 | m7 fixed-policyのsampling-only分離 | [E5] affine_policy_lower、既存保存算術 [G10_ARITHMETIC] | 辞書・価格・精度・保証固定。物理query下界でない |
| C15 §9.2 | D1–D3：保証/価格/rの役割 | [D1][D2][D3]、[D3_ROWS] | 共通exponential taskの固定例。ledger間の新同時保証なし |
| C16 §10.1 | P_Uは反証済み | [PUQ]の全量化命題、[PU]のexact certificate | Uと真のB_fullは別。全費用の一般NO-GOでない |
| C17 §10.2 | W2b N_F≤N_Oと他対照の条件 | [W2B]、固定G7 budget [GENERATOR] | 同じroot/source/precision。P5 budget互換性は未確立 |
| C18 §10.3 | W2c全trial古典・accepted量子費用条件 | [W2C]、[F2] | 未取得費用を0にしない。一般CPU/native順位なし |
| C19 §11.1 | 固定Pilotの16896件とprefix接続 | [PILOT][PSUM][PCHECK] | 実量子shots0、独立反復・一般correctness認証でない |
| C20 §§11.2–11.3 | CPU/wall・setup/online・保存と限界 | [PCOST][PSTORE][PINV][ORG][PCLOCK] | coarse/fine二重加算なし、compact原文復元不可、因果差/IID/Full-N未確立 |
| C21 §12 | 既知部品とpipeline差の未確定性 | 一次資料P1–P7、[PRIOR][MATRIX] | 網羅的priority/完全同値・非同値を判定しない |
| C22 §13 | BF/BM/SP、Parent、DF、M1-Sを別成果として保持 | [BF][BM][SP][PARENT][DF][M1S] | 異なるtask・費用・誤差を総合改善率へ結合しない |

固定ファイルのbytes/SHA256、Git blob、公開再取得照合は[source_provenance.json](source_provenance.json)、原稿の参照・保全確認は[編集検査](editorial_verification.json)。数理の機械検証や保存値checkerの再実行ではない。巨大raw archive・全回路IR・全証明・全testを今回独立に再監査したとはしない。

## 未確立事項と次の担当

| 項目 | 現在の状態 | 次の担当・必要な判断/承認 |
|---|---|---|
| 結合全体の独立性・論文中心claim | 同input/access/target/precisionで既知手順の合成と全保証・取得費用の対応が未決 | **ユーザー・GPT**が本文と既存対応を評価。新しい詳細科学レビューを開始する場合はユーザーの開始承認 |
| 投稿形態・投稿十分性 | 技術方法ノートは作成、原著新規性/投稿GO未認定 | **ユーザー・GPT**が中心claim・投稿形態を判断。外部投稿は別明示指示 |
| 一次文献の未確認範囲 | P6本文は今回未再取得、PDF頁画像照合失敗、網羅的priority未実施 | **Codex**は次の限定文献確認指示で版/頁を補完可能。新数学・還元証明は別scope |
| 一般normalizer/cost順序 | P_U偽は確定。一般B_full/対照順位や総費用優位は未確立 | **ユーザー・GPT**が研究上必要な命題を選ぶ。新理論や新数値を自動開始しない |
| F1/F2/F3の保証と実用性 | 一様な提供precision条件、bit会計、Full F3入口/angleは既存到達点。最小selector・全合法入力実用成功は未確立 | 追加実装が必要と判断された場合のみ**ユーザー**がscopeを承認し**Codex**が担当 |
| physical controlled-Q/native/全task | 条件付きoracleと小provider事例のみ。state/readout/全round・分子PR/RPE/QPE総会計は未接続 | **ユーザー・GPT**が対象と費用taskを定める。新取得/実行は別承認 |
| Pilot測定の一般化 | 順序交絡・共有負荷・observer未分離・反復不足・compact制約を保持 | **ユーザー・GPT**が必要な主張を決める。新統計解析/追加Pilotは別指示・実行承認 |
| Full-N | 未実行。既存構成と有限prefix記述の完成を妨げる必須条件ではない | **ユーザー・GPT**が実測全workloadを主結果として要するか判断。選ぶなら新固定SOURCE/契約と**ユーザーの新実行承認** |

新規計算なしで進められる作業は、本文の記号・論旨・引用精査、既存文献との対応説明、中心claim/投稿形態の検討、既存証拠の取得説明の補完である。この一覧は新しい定理・実装・実験を採択した計画ではない。旧STOP/CLOSED/HOLDや消費済みgrant/claimを再利用しない。

## GPT・ユーザーへのhandoff

**読む順序**：本文の要旨・§§1–4（構成と保証）、§§9–12（正負の費用結果とPilot/既知法）、§§14–15（限界と中心statement）。P5とG10の詳細・正確値は§§5–8/付録と固定リンクに保持した。

**既存証拠が支持する核**：同じshort-step有限Taylor operatorに対する局所return係数・正値なparent/child mean・全B_full事前取得不要の生成・有限bit補正/bias/rangeと認証予算・条件付きbit費用。実装証拠は固定入力・SOURCE・保存範囲に限定する。

**判断を返す点**：この構成的接続をどの中心claim/投稿形態で扱うか。G7正結果、G10の強い対照と下界、P_U反例、Pilotの有限観測を併記した本文を使う。Full-Nや追加Pilotの要否は本稿で独断採択していない。

**次の担当はユーザー・GPT。Codexは公開・保全確認後に停止する。** 続く文書編集は新しい具体的指示で対応できる。詳細科学レビュー開始・新科学計算/再解析・新証明・回路/native・追加Pilot/Full-Nは対象に応じた別明示承認が必要で、原稿公開を許可に読み替えない。

## 編集・取得・保全の範囲

ユーザー提供 `Track_B_統合科学レビュー_2026-10-20.md` は全文読解した。研究方針の主要参考として用い、レビュー固有の新しい式変形・推奨を本稿の新定理/正式採択へ移さず、重要な記述は固定証明・SOURCE・結果へ照合した。原本の内容・複製・全文引用を公開しない。別の私的Claude/GPT G6原本は未取得のまま扱う。

[共通運用指示2026-10-17][OPS]と共通/専用AGENTSを適用。作業はprimary Codexのみで行い、AIへの委任なし。科学moduleは静的な文字列読取とGit/HTTP/hash照合だけで、import/実行・新test・統計・数値差/比/順位計算は0。標準Pythonは文書生成・link/bytes/hash/Git照合だけに使用する。

既存v1.0、科学SOURCE、raw、grant、claim、ledger、marker、STOPは基点から変更しない。独立branchの新規本文/本記録/出典metadataと、PROJECT_MAP/docs索引への導線だけを公開する。現況synthesis・validation manifestの科学状態は変更しない。Git差分とv1.0 SHA256を確認し、公開後はremote HEAD・新規公開ファイルbytes/SHA256・clean状態を照合する。

**公開・保全確認後mandatory STOP。**

[V10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md
[E1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[E2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[E3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[E4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcd3ea6217bc00b667180cec149a70102d75f07e/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[E5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_saved.py
[MATRIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/research/track_b_g6_construction_paper_evidence_20261018/mathematical_claim_evidence_matrix.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/handoff.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md
[PU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/850ac89e1a6ea955def805e5462b8694ec0840b4/docs/research/track_b_g6_pu_analytic_proof_20261018/proof_or_counterexample_certificate.md
[PUQ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/general_order_comparison_question.md
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/handoff.md
[W2C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/handoff.md
[ORG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/research/track_b_g6_w2c_b3_pilot_result_technical_organization_20261020.md
[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/README.md
[BF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md
[BM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
[SP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d2bb1fa439748b02084bd9fbc9b10a705328f8a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/research/track_b_df_correction_et_scientific_review_20261012.md
[M1S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
[PILOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md
[PCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/cost_tables.md
[PCHECK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/saved_evidence_check.json
[PSUM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/result_summary.json
[PSTORE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/storage_reconciliation.json
[PINV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/raw_inventory.json
[PCLOCK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/clock_calibration.json
[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[GENERATOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[PROVIDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[F3_PRECISION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/src/trottertracks/algorithm_codesign/g6_f3_precision.py
[F3_GENERATOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/src/trottertracks/algorithm_codesign/g6_f3_generator.py
[D3_ROWS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/resource_rows_exact_v2.csv
[G10_TABLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/resource_table_exact.json
[G10_ARITHMETIC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f41d5e3d6c12a46c7ea387c064346e160a79936a/artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/saved_arithmetic.json
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7eff8a33bef4ee1f29698ab10c6da46988dc8db5/docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md
