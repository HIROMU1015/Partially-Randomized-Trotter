# PR-LD1/LD2：明示target付きRPE caller・C signal binding

2026-11-13 JST。基準B `91d730ac3f49bbbd8ab16b6aceda33cd44701b1f`、既存branch `codex/rpe-decoder-statistics-20261113`を再利用。技術実装・人工テスト・保存metadata照合を完了した。新しい実H4 MC、保存科学値の数値再解析、実RMSE/必要N/compiled価格/G評価は実行していない。

## 実装と再利用

[rpe_signal_statistics_caller.py](../../../src/trottertracks/rpe_signal_statistics_caller.py)に、既存処理を呼ぶ4関数を追加した。

| API | 処理 |
|---|---|
| `bind_bundle` | 元A bundle/carrierとC requestを照合し、原値・target・資格を変えない受入viewを作る |
| `accept_saved` | 既存SavedArtifactsで固定Git JSONを読み、Cのmetadata-only `plan`と上記bindingを使う |
| `prepare` | 明示evaluation_targetを検査し、既存whole-shot `W.prepare`とC `signal_input`を使う。MCなし |
| `diagnose` | 元targetから独立したevaluation_targetで既存`S.simulate_prepared`→`S.summarize`を呼ぶ明示実行入口 |

W=`rpe_shot_allocation`、S=`rpe_decoder_statistics`、R3=`rpe_decoder_fast.decode_sampled_counts`、元controller・count/PCG64/binomial・decoder・loss/CI・費用会計は変更していない。Cの大きなprovider graphを複製せず、既存`rpe_optimization_pipeline`を`integration`引数で受ける。新科学runner/CLI、N資格producer、optimizer、承認証明機構は追加していない。

## targetの区別

A bundleのtargetには`reference_solver_energy_ha`があるが`energy_ha`はなく、proxy採用はfalse。旧`W.diagnose`を直呼びするとMC後にtarget lookupで失敗する可能性があった。新経路は値・意味・SOURCE・qualification scope・元target SOURCE/H-state bindingを持つ**別のevaluation_target**を要求し、欠落時はMC呼出し前に拒否する。solver値のfallbackやground資格の付与は行わない。

元target/mean/error contract/prior/decoderはそのまま保持する。使用したevaluation_target、統計契約、whole-shot allocation、C形式の実消費signal_inputを結果に記録する。`n_packet/sufficient_N/global_RMSE_bound/G/winner`は常にnull。元Sのconditional診断・未解決trial・全trial loss/CIの意味は変えない。

## 保存A → Cの照合

対象はA `91349e703e0d09ef702ccd108745dfaf1acc29f7`のB_MEANS_LD1/LD2。configuration `a9abf8de75bf3432fcd714e868e5c14c8884b8f8`、実装 `fefd548ecd56f60d72031b3d385f6f5e6ee95d06`、raw carrier `16210932e5dabe7d12d41f35fea0e0a0d7a9b271`を区別した。

両構成とも全13roundでcanonical/physical ID、m、law、SOURCE、C physical_context、raw/corrected/reference/exact_tail、normalization、bias、経験的allowance・数値観測・state/reference evidenceのコピー一致を照合した。新しい信号値・比率・位相・半径・RMSEは計算していない。rawのRe/ImをX/Yの観測meanとして使う既存経路を保持し、normalizationや価格標本数の追加乗算はない。

受入packet.sourceは元numerical_source、各provider_binding.numerical_sourceはCの既存契約どおり**mean.sourceのcarrier参照**。実装SOURCE、input/reference SOURCE、carrierの資格情報を別欄で保全する。carrier.qualificationは辞書、bundle mean.qualificationはUNQUALIFIED文字列であり、同一schemaとして比較・上書きしない。

LD1/LD2のIDはそれぞれ`a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`、`1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`。元target/proxy=false/全UNQUALIFIED/科学null/status/STOPを保全した。binding成立は物理資格を意味しない。Cの正式会計gateは閉じたままである。

## 検証と限界

[TESTS.txt](TESTS.txt)に実行コマンド・範囲を記録した。新規9＋既存38＝**47技術テストPASS**。人工MCの明示target・scalar/R3の全trial/summary一致、13round/X-Y配分、巨大整数の保存とbinomial int64域外拒否、zero/tie、未解決loss、欠測・不正shot・SOURCE/law/config混在拒否、診断nullを確認した。

保存AのテストではMCと実G入口を禁止した。C `5cc7c07e112ce97205948465c94441cba0042416`のplan/price_context/signal_input/compareとA metadata plannerを、変更していない関数本体として固定Gitから選択し、受動providerのみを注入した。C.compareの未資格gateとG/winner=nullを確認した。これはC全runtime/provider graphや実H4科学実行の検証ではない。他担当のworktreeやDのT2実行には介入していない。

## 次の担当とSTOP

使い方・受入仕様は[HANDOFF](HANDOFF.md)、残る採択事項は[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)。ユーザーが評価target・全事象loss・risk水準・whole-shot vector・反復/precision/confidence/seedと有限scopeを明示採択した後にのみ、Bがその範囲を実行する。Cはこの受入view/signal_inputを既存pipelineへ接続し、Dは独立に価格証拠を供給する。本納品は仕様共有資料であり、他担当への送信・作業変更は行っていない。

commit/push・remote SHA/tree/主要成果物照合後、**mandatory STOP**。今回の人工PASSを科学的精度・十分N・正式Gの認定へ昇格しない。
