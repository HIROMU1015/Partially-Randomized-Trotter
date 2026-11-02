# PF次数逆転C：H2共通政策の実行前準備

2026-10-23。固定監査commit `366b2fd24f7e86aa82b5b518b6e3774520f65596` から作成した独立・detached worktreeの未commit資料。
**準備完了／科学実行・政策採択・比較CのGOは未承認。mandatory STOP。**

- [技術報告・比較可能性の更新](REPORT.md)
- [allowanceのSOURCE定義・3案の比較・推奨](ALLOWANCE_OPTIONS.md)
- [固定4候補の最小実行計画（未採択・実行不可）](EXECUTION_PLAN_DRAFT.md)
- [不足資料とlocator・受入条件](missing_evidence.json)
- [保存4行の無計算転記](candidate_records_4.csv)、[保存入力抜粋](saved_input_excerpts.json)
- [SOURCE索引](SOURCE_INDEX.md)、[取得manifest](source_manifest.json)
- [Git provider照合の範囲](provider_git_comparison.json)、[正確な既知locatorの確認](exact_locator_checks.json)
- [技術検証](technical_verification.json)、[終了状態](completion_status.json)、[新規資料inventory](file_inventory.json)

今回のスクリプトはstdlibによるcopy/hash/JSON・CSV・AST検査だけを行う。歴史的SOURCEをimport/実行しない。
科学結果・研究上の採択・validation statusを更新しておらず、新たな科学検証経路も追加していないため、既存の研究文書・索引・manifestは変更していない。

新PRのprivate資料についてユーザーから「一旦なしで進めて後で渡す」との回答を受領した。実receipt未取得を維持し、追加科学計算で補っていない。commit・pushは今回の指示対象外。
