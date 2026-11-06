# 研究C：H4線形Z再配置後の登録下流比較・実行完了

2026-11-06 JST。固定H4・直鎖1.00 Å/STO-3G、number CDF8、Nα=Nβ=2/sector36、元frame順LD2。線形Zを既存κの一体headへ移し、frame identityとZZを維持した同じ表現を全候補へ適用した。partial S2/s1/δ=h=.1/K0/m0..12、own-DF ground oracle・状態準備除外、固定P0/fenced serial/shell込みの条件付き総RZを比較する。新DF fit・JU再最適化・H2/H6等は0。

**登録した主比較A/Bは、ともに近似IIDモデルの名目判定でENTRY由来DFの方が低費用だった。** これは今回の有限候補・辞書・価格proxyでの結果であり、一般的な最適性・実CA・全law・hardware/FT資源優位を保証しない。

[比較原本](../../artifacts/research_c_h4_handoff/2026-11-06/science/run01/comparison.json) ／ [公開要約](../../artifacts/research_c_h4_handoff/2026-11-06/publication/summary.json) ／ [800価格標本CSV](../../artifacts/research_c_h4_handoff/2026-11-06/publication/native_price_samples.csv) ／ [保存物監査](../../artifacts/research_c_h4_handoff/2026-11-06/execution/result_audit.json) ／ [STOP](../../artifacts/research_c_h4_handoff/2026-11-06/science/run01/STOP.json)。

## 実行範囲と来歴

PはC0/64更新のPR-aware γ=.1、E1は同じC0/64更新のENTRY γ=.1、E2はENTRY γ=.03。比較の両側に同じpartial PR/RTE実装を使う。ENTRYという表示はDF候補の目的armを表し、決定論法を意味しない。3候補の保存DF予算と元入力は元契約のまま再利用した。

元の科学条件・16入力は[固定a670契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a67036a6baafc00deaa38a1b0b801d313d519ad5/docs/research/research_c_h4_relocated_downstream_contract_unapproved_20261106.json)。実装切替はユーザーが明示許可した一回だけで、[科学SOURCE cfc64b708](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cfc64b70861a6482dddd042da8a110abebd6ad7a/artifacts/research_c_h4_handoff/2026-11-06/technical/SOURCE.json)、[引継ぎ契約](research_c_h4_handoff_contract_20261106.json)へ固定した。元のSOURCE/契約/結果/過去STOPを変更しない。

元processはSIGINTで直列compileの回収待ちになったため、検証済み標本0の処理中E1/r128 batchを保全してSIGTERMで終了した（実際のexit143）。元wrapperのSTOP/completionは作られず、その欠落状態を維持した。[別の中断観測](../../artifacts/research_c_h4_handoff/2026-11-06/execution/termination.json)に明記し、旧PASS/STOPを捏造していない。新processのSTOP.failed=false、completion.failed=nullは今回の継続実行についての事実である。

元9 batch・288の検証済み標本と完了D/13roundを再利用し、残り512標本をprocess分離で計算した。未検証の古い途中生成物はPASSに流用せず、同じsample index/seedで全検査を完結した。乱数SOURCEは元の `bb102f16c3bf757853f9e9fc98d811dc230592d71cd6bad57bcce0363fa2046e`、コードSOURCEは `6bd88bc86bd1da2876f87b8ae67be509b15727f01517cea9c7219f91c350c4cc`。中断したr128のevents JSONは旧新32/32でbyte一致し、全800標本のdomain・必要ファイルと再利用288件のbyte同一性を監査した。

## 主比較：共通r32と公平retuning

r8/16/32/64/128を各候補に同じ機会で評価し、各selection32だけで選択を固定した。P/E1/E2は全てr16を選択し、PR-or-Dの選択点も全てPRだった。r16はgrid内部であり、端点拡張やfinalに基づく再選択は行っていない。独立final64はE1/r16、E1/r32、E2/r16、P/r16、P/r32の5組。

| 主比較 | PR-aware由来Pの総RZ | ENTRY由来DFの総RZ | P/ENTRY点比 | P−ENTRYの名目97.5% Welch区間 | 判定 |
|---|---:|---:|---:|---|---|
| A：P/r32 対 E1/r32 | 92,464,828,249 | 86,352,772,151 | 1.070780079733 | [3,233,654,155.710, 8,990,458,040.290] | NOMINAL_MODEL_ENTRY_LOWER |
| B：P/r16 対 E2/r16 | 87,514,417,699 | 75,621,395,471 | 1.157270599860 | [9,453,375,634.390, 14,332,668,821.610] | NOMINAL_MODEL_ENTRY_LOWER |

2主比較のfamily α=.05、各名目97.5%区間、各arm N=64を事前固定した。両比較はprecision_met=trueで、登録されたskewness/influence診断も通過した。区間はSciPyの固定N・近似IID Welchモデルで、coverageは検証されていない。Ratio区間は登録されておらずnullのまま。

旧辞書の公平retuningではP/ENTRY=.545306だったが、[その旧固定結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_h4_100_results_20261106.md)の45.47%PR低費用を、今回の再配置後の辞書へ移せない。今回の順位逆転は保存3候補・登録下流条件に限定する。再配置後の目的で再最適化した比較、線形Z項だけの因果的寄与、PR-aware一般の劣位は検証していない。

## 個別の同J/U D比較と未解決診断

| 候補/r | 条件付き総RZ | planned shots | W | 同J/U D総RZ | PR/D点比 | 個別名目95%判定 |
|---|---:|---:|---:|---:|---:|---|
| E1/16 | 76,465,702,876 | 15,116 | 13,710,432 | 105,217,244,042 | 0.726741168 | NOMINAL_MODEL_PR_LOWER |
| E1/32 | 86,352,772,151 | 14,376 | 11,454,752 | 105,217,244,042 | 0.820709314 | NOMINAL_MODEL_PR_LOWER |
| E2/16 | 75,621,395,471 | 15,118 | 13,718,624 | 105,217,244,042 | 0.718716748 | NOMINAL_MODEL_PR_LOWER |
| P/16 | 87,514,417,699 | 15,862 | 16,034,512 | 103,256,682,350 | 0.847542413 | UNRESOLVED_DISTRIBUTION_DIAGNOSTIC |
| P/32 | 92,464,828,249 | 14,630 | 12,214,432 | 103,256,682,350 | 0.895485175 | NOMINAL_MODEL_PR_LOWER |

P/r16のPR/D点比は.8475424、名目95%比区間[.83211765,.86296718]だが、delete-one影響度のhalfwidth比.25118664が登録閾値.25を越えたため、固定規則どおり `UNRESOLVED_DISTRIBUTION_DIAGNOSTIC` を維持する。外れ値除去や追加標本は0。他の4組は個別名目モデルでPR/D低費用だが、同時のsecondary優位を保証しない。

選択固定後のbest-policy比較P対E1/P対E2も、個別95%では同じP/r16の影響度診断により両UNRESOLVED。主比較97.5%のhalfwidthは異なるため、主比較A/Bが診断通過したことと矛盾しない。この相違を用いて有利なconfidenceへ切り替えていない。

既存統計helperの `selection_scope` はr16の個別PR/Dにも「preselected r32 only」と表示する。これは継承された説明ラベルの誤記で、実際のr、sampling namespace、frozen_selection、N64と数値計算は登録どおりである。rawは改変せず、公開要約のactual_selection_scopeに実際のselection32固定／独立final64を明記した。科学条件や数値の変更・再計算は行わない。

## 全検査・実行時間・公開範囲

15 selection政策×32＝480、5 final組×64＝320、合計800の一意な価格標本。再利用288＋新計算512、同J/U D3件を完了した。検証対象はraw/compiled/actual raw inverse/actual compiled inverseの全512入出力、phase/return/XYを維持し、correctness FAIL0。16入力、64コードSOURCE、旧checkpoint3608ファイルの照合が通った。検証済みQPYは価格3200＋D12＝3212。中断途中の未検証生成物やcache再構成があるため、算術callの実累計を3212と同一視しない。

引継ぎ後の実行は **22:55:29〜23:02:51 JST、442.15秒（約7分22秒）**。親CPU37.84秒、子CPU11370.41秒。これは新計算512標本・旧288標本の復元等に要した時間であり、旧processでの計算や開発・切替待ちを含む全実験の所要時間ではない。E1/r128の新32標本は64.17秒で完了した。新runnerのretry/resume0、一回のplanned handoffだけを記録した。129人工・回帰test PASSは技術証拠であり、科学結果の性能証拠とは区別する。

公開するものは、比較・選択・D・各政策とfinalのscalar JSON、800価格標本CSV、監査、開始/終了/STOPおよび[raw file hash一覧（gzip）](../../artifacts/research_c_h4_handoff/2026-11-06/publication/result_files.json.gz)。raw QPY/NPZ等の9318ファイル・2,761,495,069 bytesはローカルに保全する（worker用pickle/mmap cacheは一覧から除く）。全raw binaryのGitHub公開や外部独立再現完了は主張しない。

価格標本はclassical calibrationであり、shots/Wへ標本数を掛けない。planned shotsを実量子shot実行数とみなさない。actual CA、全law synthesis bound、implemented-law誤差、ideal-law価格transfer、FT資源、厳密DF enclosure、一般化・新規性・論文化GOは未資格／未判断のまま。主agent自身の実装・保存物監査であり、独立科学レビューは行っていない。

計算は登録範囲の完了でSTOP済み。公開照合後にMANDATORY STOPし、追加実験・r拡張・標本追加・H2/H6・重要レビューへ自動進行しない。次の作業なし。
