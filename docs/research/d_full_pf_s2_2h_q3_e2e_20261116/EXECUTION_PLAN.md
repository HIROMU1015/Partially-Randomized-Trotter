# Full PF/S2/2h：通常探索入口からのQ3一件検証

最新利用者指示の範囲で、configuration 9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981だけを対象にする。公開D352539の通常CLI、保存13信号・26価格、既存PF worker/AST guard/decoder/target/law/loss/exact tauを用いる。B2統合版検収は別件として扱う。

§10.5の照合はDUPLICATE_PREFLIGHT.json。B保存244 job、B2既知FullPF領域、D既存round、C固定catalog、関連processで対象一致はない。総予算0のproposal_onlyも対象だけからB_PRICEを返した。配分は研究者の入力ではなく、通常探索器の出力をFIXED_INPUTS.jsonに記録したもの。ROUNDには整数shot列や手動seedを埋め込まない。

実行はROUND.json・新run01で、FULL_PFのみ、configuration allow-list一件、新規実Q3の総上限1。pf_completionを起動せず、signal/sampling/build/compile/fitは追加0。priceは保存direct10+conditional16、未知の長q誤差/CIはUnknownのまま。Q3適合/不適合はどちらも検収上有効。条件付き費用を正式G/winnerへ昇格させない。

native-dependency-v2のPython3.11.0rc1で元PF process-workerとpinned validatorを人工2test確認済み。実科学call0で元AST guardとAPI/partial保存を通過した。実Q3では既存の8-worker DP・stage/roundログ・process終了情報を使う。CPU/RAM/OS/他利用者への設定変更はしない。one-Q3 jobのため同時job数1とする。

同一run・同じROUND・同一総予算で通常CLIを再開し、累積LAUNCH<=1・追加起動0を確認する。技術障害時は既存worker成果を回収して復旧し、科学worker完了後に再評価しない。partial/source/数値gate不成立はUNKNOWNを保存する。対象外jobを投入しない。

原結果をcommit後、既存Cが読めるgeneric PF publicationで配分・MSE判定・候補固有価格を結合する。不適合を達成費用/最安適合へ加えない。Cの通常周期での反映を確認し、旧D/受入テスト/停止記録は保全する。本一件の検収・公開後に追加探索へ自動進行しない。
