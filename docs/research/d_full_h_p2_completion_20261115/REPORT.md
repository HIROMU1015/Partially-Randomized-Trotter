# G36223：修正版への保存結果引継ぎ

**最終状態：MANDATORY STOP。** 修正版SOURCE `0dc1eeaee3d9d6465f18edc33d4e17ed312bd6f4` への切替は実施したが、新workerが4:14.99でchild receiptを書かず異常終了し、parentはSTOP_CELL/exit2となった。追加sampling/compile0。D7の18 unique compileは全て保存済みだが、最終path検査は16軸PASS・2軸UNKNOWN、較正cellは4完了・G36223未完了。再投入・条件緩和・seed差替えはしていない。[STOP_DETAIL.json](STOP_DETAIL.json)とrun01の原記録を参照。

ユーザーが「修正版に切り替えた方が早そうなら切り替えてもよい」と指示した。旧SOURCE887のcheckpoint wrapperでは内部fingerprintが旧writerへ戻る問題を人工で再現済みで、dfab06ee0の修正後は同じcanonical bytesのfast writerを全wrapper内部検査にも適用する。旧残検査は2時間以上続いているため、修正版で残検査を行う方針とした。実対象での速度改善率はまだ未測定。

## 固定入力・変更点

[FROZEN_COMPLETION.json](FROZEN_COMPLETION.json)はD7 G36223の同plan・physical ID・cohort・seed・compiler/profileと、d323fee1cのpath0 native checkpoint、887c3b9d0の完了path1 receipt/個別compile hashを固定する。path0のsampling/compile/既存7数値gateを再実行しない。path1は元receiptを検査付きで読み、全検査・価格・event列を再利用する。**追加sampling0・追加compile0、D7全体18 unique compileを維持する。**

既存P2のexecutor・record producer、SOURCE/input checks、phase/parity/XYを再利用する。childではsampler/compiler入口を禁止する。CODE変更はcheckpoint wrapperのfast writer適用、段階別progress callback、完了path再利用、保存済み入力からの制御付き引継ぎだけ。科学条件・閾値・Qiskit1.3.0/rz/sx/x/cx/opt1/seed17/P0/new ZZ/scalar once/測定wrapperは変更しない。

## 検証と移管条件

[TESTS.txt](TESTS.txt)：11人工test PASS。通常X/Y process並列、小toyの実transpile/QPY/7gate、checkpointの旧writer禁止・sampler/compile0・元event/fingerprint一致、完了path再利用の同一性・出所・hash確認、段階log、稼働worker/科学失敗からの再試行拒否。初回はtest fixtureのJSON tuple/list比較と排他的ファイル書込みで失敗し、fixtureだけを直して再確認した。H4追加試運転0。D1本人のlocal検査であり独立レビューではない。

旧managerとpath0 workerのUID/開始時刻/commandを照合し、保存path0 native・完了path1・旧jobのhashとgateを検査する。旧path0又は全体が先に完了した場合は切替えず既存結果を使う。未完了ならmanager→workerを一時停止して境界を再確認し、worker→managerの順で終了する。元のterminal STOP_CELL/exit2は計画的な切替として保全し、科学gate FAILと混同しない。旧SOURCE/raw/QPYを変更しない。

新SOURCEは別worktreeで固定し、一回だけ起動する。privateのPROGRESS.jsonlに段階開始・直前段階完了・秒数を記録し、STACKS.txtに120秒ごとにworker内部stackを記録する。観測のためにroot/sudo・OS設定・他processを変更しない。

切替の実績はrun01/TRANSFER*.json、実行job、console/time、最終G36223 receiptに記録する。移管実績は [TRANSFER.json](run01/TRANSFER.json) に保存した。旧887 runは2:22:05で計画的終了。新runは4:14.99、報告peak RSS5557360 KiB。旧887のGNU timeは未回収のworker資源を含まないため、全jobのpeakとして扱わない。元d323のpath0 nativeは5022.000351489987秒、peak RSS15116922880 bytes、7数値gate PASS。重複確認は既存df0e11fd1とDELTA_PREFLIGHT、今回の旧job・path1保存結果照合を継承し、受入可能な今回自身の4 compileを再取得から除く。

完了後は14軸の公開済み証拠へG36223の4軸を加え、Cへ固定locator付きで引き渡してmandatory STOP。C科学受入・fit/凍結、m5/m6 HOLDOUT、長時間外挿、law E[C]/N/Gは行わない。ロードマップ本体は編集しない。

## 異常終了の確認範囲と部分引渡し

最後に開始した段階は `build_new_evolution`。最初の120秒stackではfast writer内、次の定期dumpはframeの途中で切れている。Python例外の記録、子receipt、子終了signalは得られなかった。service cgroupのoom/oom_kill/oom_group_killは0、apportの自分の該当crash報告は見つからなかった。これだけで終了原因を断定しない。同じPythonの単純な3秒人工loopでは0.01秒間隔の定期dumpを発火させてもexit0で、実workerの異常は再現できなかった。[人工診断](SYNTHETIC_DIAGNOSTIC.json)。

新たな数値gate不一致は観測していないが、path0のparity/scalar-event phase/X-Y/fingerprintの最終receiptがないためPASSへ昇格させない。parentの `error:null` は成功を意味しない。今回の予期しない終了を、旧d323/887の計画的切替によるSTOPと区別する。

G36223の価格はpath0 X/Y各5552901、path1 X/Y各5572406。[OBSERVED_COMPILES.json](run01/OBSERVED_COMPILES.json)とCOMPILE_RECORDSに元JSON bytes/hash・producer SOURCEを保存した。path0のX/Yは保存wrapper QPYのhashと末尾の測定軸を読取りで確認した：[軸対応](SAVED_PATH0_AXIS_METADATA.json)。これは新しいbuild/compile・作用検査ではない。path1は完了時の元receiptをhash完全一致のlossless gzipで保存し、[実装証拠](run01/PATH1_IMPLEMENTATION_EVIDENCE.json)へ結ぶ。path0の元event receiptと7数値gateも保全した。

G36223の完全なPRICE_CALIBRATION_RECORD_V1は作成しない。既存4 cell・8個のC入力recordと14軸の公開済み証拠を維持し、G36223の保存4価格・path1 PASS・path0未確認を部分引渡しとして渡す。D1の公開後技術照合とC科学的受入NOT_PERFORMEDを区別する。canonical-round guardは呼ばない。

ロードマップへの影響：D7全較正引渡しの前提は未達であり、C fit/凍結と、その後のHOLDOUTには進めない。原因特定と未確認path0の扱いが後続課題。価格モデルの否定や量子的correctness失敗とは結論しない。ロードマップ本体は統合担当が更新する。

公開結果SHA：`1fdb5dd42f65d6286848dc76e686f62c5b3accf8`。保存済みraw/compile・部分証拠を公開commitから読取り照合しPASSした。旧4 cellの結果は7d0c39066へ固定済み。C向け固定locatorと実際の検査範囲はHANDOFF_C.json／PUBLICATION_CHECK.jsonを参照。これは公開後のD1技術照合であり、path0残検査やC科学受入のPASSではない。

[C向け固定locator](../d_full_h_p2_calibration_resume_20261115/HANDOFF_C.json)／[公開後検査](PUBLICATION_CHECK.json)。
