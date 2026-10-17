# 新アルゴリズム研究：H1〜H4探索の一時停止記録

2026-10-16 JST。ユーザー指示「新アルゴリズム研究の探索範囲再設計」に基づく追記型の研究記録。既存の科学判定を再実施した記録ではない。

**H1〜H4を対象とする現在の探索を `PAUSED` とし、新アルゴリズム研究全体は研究課題選定段階へ移行する。** 新アルゴリズムの正式採択はない。

## 固定された完了記録と今回の変更

起点は `HIROMU1015/Partially-Randomized-Trotter` の固定commit `0d460c54f3bf54ea46d4d8129f6702703c4bb217`。次の3資料を今回GitHubから独立取得し、byte数・SHA-256・Git blob SHAを確認した。identityは [provenance.json](provenance.json) の S1〜S3。

- [静的K0技術報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d460c54f3bf54ea46d4d8129f6702703c4bb217/docs/research/qc_new_algorithm_static_screen_k0_20261016/technical_report.md)
- [文献対応表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d460c54f3bf54ea46d4d8129f6702703c4bb217/docs/research/qc_new_algorithm_static_screen_k0_20261016/literature_correspondence.md)
- [取得・作業provenance](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d460c54f3bf54ea46d4d8129f6702703c4bb217/docs/research/qc_new_algorithm_static_screen_k0_20261016/provenance.json)

| 項目 | 採用する記録 | 適用範囲 |
| --- | --- | --- |
| H1〜H4静的K0 | `COMPLETE` | 完了済みの限定的な静的screen |
| 総合技術判定 | `NO_GO_NO_INDEPENDENT_GENERATION_RULE` を維持 | 今回調べた記述・生成規則で独立した新回路生成規則を特定できなかった |
| H2個別判定 | `UNDETERMINED_DATA_REQUIRED` を維持 | leakage・分母・情報取得・費用が未確定。総合NO-GOで上書きしない |
| H1〜H4の現在の探索 | **`PAUSED`** | 今回のユーザー承認により、一時停止を追記 |
| 当該系列のK0.5・K1 | `NOT_AUTHORIZED / NOT_EXECUTED` | 本作業はK0.5/K1ではなく、読取り・限定文献調査・研究範囲整理・資料公開 |
| 新アルゴリズム正式採択 | なし | 本資料の問題領域を候補やGOと扱わない |
| 新アルゴリズム研究全体 | 継続・研究課題選定段階へ移行 | 全体終了を意味しない |

既存報告にある正式状態更新待ち等の記述は、その時点の履歴として保存する。本書が今回のユーザーによる一時停止指示の参照先であり、過去のdecision・method status・レビュー本文は変更しない。

## 一時停止理由の範囲

完了K0の記録では、H1は既知の固有状態commutator恒等式・補正式、H2は既知の相関／filterと元Hamiltonian測定、H3は既知のinteraction-picture/Gaussian frame、H4は位相参照と入力アクセスの条件に帰着した。新しい有利な回路生成規則と、その情報取得を含む費用改善は特定されなかった。これは上記固定報告の要約であり、新しい不可能性証明ではない。

H2の未確定性、有限stepの実用性、より一般の代数構造は残る。したがって、この技術的NO-GOをD1・D2内の全アルゴリズム、全分子、全入力モデルへのno-goと記録しない。単なる名称変更や未実装を、独立した生成規則の証拠ともしない。

## 固定条件・他研究の保全

D1はblock-encoding、qubitization、それらを中核とするQSVT・量子walkを新規研究対象から除外する。D2は元の電子Hamiltonianの絶対基底エネルギー推定を主タスクとする。定数項・energy origin・対象粒子数sectorを失った別タスクを、元タスクの達成と扱わない。

[既存PF系列の正式状態](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json)と [SOS-PF A3正式採用](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/artifacts/qc_sos_pf_a3_state_adoption/2026-10-16/decision.json)を保持する。第一候補の現固定構成CLOSED、H-E1現定義CLOSED、M-a/M-b STOP、PF小改変PAUSED、旧D HOLD、SOS-PF現分割生成探索CLOSED、A2 NOT_ADOPTED、SOS-PF K0 `UNDETERMINED_DATA_REQUIRED`、K1-F未認可・未実施を変更しない。N1 STOP／N2 HOLD／N3 CLOSED／THRIFT現template CLOSEDも維持する。

Track A/B、第1・第2研究は独立している。特にTrack Bの後続M1-S G0-T1限定理論契約とCheckpoint結果を、古い統合記録だけで一括PAUSEDとしない。これらの状態は本書の適用対象外であり、[探索範囲表](research_coverage_map.md)で固定資料の検討範囲のみ参照する。

既存SOURCE・results・authorization・manifest・STOP marker・レビューは変更しない。新規科学計算、保存科学数値の再集計、実装、数値テスト、回路生成・compileは0件。

## 引継ぎと停止

[探索範囲表](research_coverage_map.md) → [科学的問題と未確認事項](open_problem_landscape.md) → [GPT判断資料](gpt_decision_handoff.md)を次の研究方針選定用とする。具体的な新アルゴリズムは成立していない。

新規5資料の専用branchへの公開・独立取得照合後 **Mandatory STOP**。新候補の採択・設計・実装、新K0/K1、停止系列の再開、科学計算、GPT/Claude詳細科学レビューを自動開始しない。

## 2026-10-17追記：Exact Encoding限定案の区切り・課題選定への復帰を正式採択

ユーザーは、固定commit `830be75335bb9b599488cb365213d555b0dc18bb` のExact Encoding × Controlled Evolution限定構成案を区切り、④「PRに限定しないアルゴリズム改善」の**研究課題選定段階へ戻る方針を正式承認**した。この一案は `REDUCES_TO_KNOWN_METHOD` として保存し、追加探索・実装・数値検証を行わない。次の具体的な研究課題は未採択であり、Encoding一般の不可能性や④全体の終了ではない。

[正式方針・日付ノート](../研究ノート/2026-10-17_qc_exact_encoding_direction_adoption.md)、[machine記録](../qc_exact_encoding_direction_adoption_20261017/decision.json)。上記2026-10-16本文と `provenance.json` は当時の記録として保全し、旧状態の意味を変更しない。①②③、H1–H4 PAUSEDおよび④の既存CLOSED/STOP/PAUSED/HOLDは不変。SOURCE・結果・科学契約・authorization・grant・ledger・markerと直前報告を変更しない。今回の記録・公開で新規候補探索・採択、科学レビュー、科学計算、SOURCE実装は0件。公開同一性・保全確認後mandatory STOPし、次の課題選定や検証には自動移行しない。
