# H6 full m5_best：保存4再生の精度診断完了・旧raw FAIL維持

2026-10-19。**一回解析は正常終了。拡張精度でもraw差は残り、旧STOPは維持する。** 新しいcompiled費用の受理は0、追加compile・再会計・retryは行わずmandatory STOP。

SOURCE：`99e8723d402245a9ad923e7dbc0800f47bc575b6`。plan：`b2b2a818dd318a9a981cacae3eeb59617501992e9d6e60ff45d1c151dc288254`。branch：`track-a-h6-m5-saved-phase-run01-20261019`。

[実行契約](track_a_h6_m5_saved_phase_readiness_20261019.md)、[起動記録](track_a_h6_m5_saved_phase_run01_start_20261019.md)、[結果bundle](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_run01_results/2026-10-19/summary.json)。保存値の新しい後処理を追加せず、承認済みrunnerの出力をそのまま転記・監査した。

## 1. 終了と固定範囲

15:04:07 JST起動、15:17:11 JST終了。外部wait4のwallは783.5414437744766秒（約13分04秒）。launcher native exit0/reap、4 worker全てexit0/reap、signal0、原runのSTOPとは別の正常完了記録である。

対象はH6直鎖・隣接1 Å/STO-3G/tol-only1e-8 DF19、T=.8/P=0、full m5_best/prefix19/q2、symmetric-directional/full Hadamard cosine、保存low状態1つ。保存wrapper/compiled各1つを通常/拡張で計4再生した。新SCF/DF/state/full-H reference、sampling、回路合成、transpile、signal探索、sine、shot/費用会計、fit、rankingは0。

起動時のidle CPU gateから1 workerで順次実行した。過去のworker上限を流用せず、共有環境や他ユーザーprocessを変更していない。

|再生|wall秒|保存vectorの再現|
|---|---:|---|
|WRAPPER_ORDINARY|62.00116214342415|差0、完全一致|
|COMPILED_ORDINARY|172.15716240368783|差0、完全一致|
|WRAPPER_EXTENDED|95.6272793635726|通常再現gateの対象外|
|COMPILED_EXTENDED|380.449434151873|通常再現gateの対象外|

通常2本の保存再現が通過してから診断結果を受け入れた。拡張精度の実mantissaは64 bit、binary64は53 bit。保存角度のbinary64値を正確に移し、同じgate順序で解析式を作用させた。係数の再最適化や状態の正規化・位相補正はしていない。

## 2. 保存された診断結果

下表は[runnerのJSON](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_run01_results/2026-10-19/diagnostic_report.json)の転記。今回の終了監査で再演算した値ではない。

|指標|通常精度|拡張精度|
|---|---:|---:|
|wrapper/compiled raw差|1.504043707445665e-9|1.5039128868467355e-9|
|共通位相 rad|1.5028193689500676e-9|1.5026891953807317e-9|
|phase-aligned残差（診断専用）|6.067672593889436e-11|6.065602469072673e-11|
|wrapper norm|0.9999999999999337|1.0|
|compiled norm|0.9999999999987826|1.000000000000002|
|元raw基準1e-9未満|FAIL|FAIL|

同一回路を通常/拡張で比較した差は、wrapperが7.772378217648106e-14、compiledが1.2631160114195734e-12。これは上の「wrapper対compiled」の差とは別の比較である。JSON内の`precision_sensitivity.*.raw_gate_PASS=true`はこの同一回路・別精度の診断値であり、旧compiler等価性や費用受入をPASSにしたものではない。

Hadamard、枝別の差・位相、overlapも同じJSONへ保存されている。数値を有利に選び直す追加解析はしていない。

## 3. 言えることと未解決事項

通常経路では保存vectorを完全再現したため、今回のQPY復元/通常再生によって新しい不一致を作った状況ではない。拡張精度でnormの数値ずれは小さくなった一方、wrapper対compiledのraw差・共通位相・aligned残差はほぼ残った。

**この保存初期状態では、演算精度を広げるだけで元のraw不一致は解消しなかった。** 保存gate列・位相帳簿などの表現差を調べる候補が残る。ただし、元のtranspile中の角度/位相の丸めや合成表現は今回やり直していない。原因を単独の丸め、global phase帳簿、compiler bugのいずれかに確定した結果ではない。

対象は1状態のみであり、全状態のunitary等価性や厳密な誤差上界を証明しない。phase-aligned残差が小さいことを理由に旧raw基準を置換したり、phase補正した回路を採用したりしない。旧H6 full m5のFAILは残り、旧58受理軸と未受理cosine費用、未投入sineを維持する。

## 4. 終了監査と保全

[終了監査](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_run01_results/2026-10-19/final_audit.json)は26項目PASS。SOURCE/入力/runtime1831file、新grantとmarker、正確な4 job/ledger、4 resultとvectorのSHA、report SHA、native exit0と全worker回収を照合した。監査ではQPYやNPZを復元しておらず、結果の新数値解析も0。

既存684 fileと準備側10 fileが不変。原SOURCE・raw・grant・ledger・marker・STOP、元のDRAFT/INVALIDも保持した。旧STOPを遡及的にPASSにしていない。他worktree・共有環境は変更しない。自己確認であり、独立レビューではない。AI subagent0。

全バイナリvector・実path・有効grant/configはprivateに保存し、公開bundleには既計算の指標・receipt・metadata・hashのみを置いた。新科学計算や人工テストの再実行は追加していない。

結果資料をlocal commitし、公開pushは行わずmandatory STOPする。後続Phaseや追加compileは起動していない。

## 5. 次の担当

**ユーザー**：保存gate列・位相帳簿の静的調査へ進めるか判断する。この診断は完了したが、元のraw不一致は未解消である。

**Codex（次の指示がある場合）**：既存証拠の静的調査方針を具体化する。新しい数値診断・再compile・基準変更・費用会計を自動開始しない。AD03の主比較採用と後続Coreは保留を維持する。
