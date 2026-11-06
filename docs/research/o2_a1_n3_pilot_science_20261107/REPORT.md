# 752-shot予備検証：SOURCE固定

採択済み752-shot条件のactivationと人工検証を準備した。人工44件PASS（新activation10、準備版pilot/design23、既存campaign11）。保存入力/認証SOURCEと依存環境を静的に照合済み。現時点は実入力前、pilot scientific callsは0。
[実行仕様](EXECUTION.md)、PROFILE_LOCK.json、INPUT_BINDING.json、人工テスト結果を実入力前に公開・照合してから、持続managerで一度だけ開始する。

独立worktree/branch、fresh output/cache。旧測定・係数・乱数・cacheは流用せず、旧SOURCE/STOP、main/dirty/untracked、C1凍結を維持する。scientific retryはしない。
成功/最初のSTOP後に結果・全費用・未解決事項を追記して公開照合し、mandatory STOPする。小pilotは精度保証や期待費用改善の証明ではない。
