# H8 bias相殺・非相殺診断：保存証拠の監査と実装計画

**184決定論候補の保存証拠を監査した。172候補はsigned成分を両軸に保存し、旧B0+Iの12候補は成分未保存だが復元に使う端点値を保存している。**
これはschema・来歴・bytesの監査結果であり、実H8のbias再計算・非相殺適格性・費用・ランキングの新結果ではない。
今回それらの数値解析、signal/compile/sampling/fitは全て0。新しい科学的結論や厳密bias保証を追加しない。

基準：Phase A結果 `01309059636f2432e29b663bcd2eb0e400f5ae0a`、B2準備 `3a450af0fb431269ac94152807992883c1490b26`。
H8直線鎖1Å/STO-3G、保存tol-only DF28、16 modes/sector4900、T=.8、q1/2/4/8、δ=.8/.4/.2/.1。
旧B0/B0+Iはprefix7/14/21のS2、旧full PFはprefix28のS2/S4。
追加はB0/B0+IのS2 prefix8〜20（旧14を除く）とS4 prefix15〜21。

[候補・軸ごとの分類CSV](../../artifacts/resource_applicability/track_a_h8_bias_evidence_audit/2026-10-17/candidate_axis_inventory_v1.csv)／
[詳細metadata監査v2](../../artifacts/resource_applicability/track_a_h8_bias_evidence_audit/2026-10-17/evidence_audit_v2.json)／
[実行計画・SOURCE/tree hash・人工検証](../../artifacts/resource_applicability/track_a_h8_bias_evidence_audit/2026-10-17/README.md)。

## commit固定の追補（2026-10-17）

利用者の「コミットして、プッシュコマンドを出して」という指示により、監査時の独立snapshotを資料専用branch `track-a-h8-bias-evidence-audit-20261017` へ固定した。
metadata監査器・人工prototype・tests・入力索引のSOURCEは `efc7e00f5032d3003347e843b4929eb57b7ed47e`。
[commit準備の照合記録](../../artifacts/resource_applicability/track_a_h8_bias_evidence_audit/2026-10-17/commit_preparation_receipt_v1.json)にSOURCE結合とB2保全確認を保存する。
pushは行っていない。以下の「未登録snapshot」「Git更新なし」は元の監査段階の履歴である。

資料固定前の読取では、B2は `RuntimeError:MEMORY_PRESSURE` によるSTOP、launcher記録はdriver native exit -9・all_reaped=true・cleanup_errors=[]、元launcher/driver PIDは不在だった。
launcher自身のnative exit receiptとB2最終PASS gateは未取得で、B2成功や4軸PASSを認定しない。原STOP・結果・grant・ledger・markerは変更しない。
このcommit作業では人工30検査を新worktreeで再確認しただけで、実H8の新数値解析、非相殺会計、ranking、signal/compile/sampling/fit、新grantは全て0。
実H8数値runnerとそのSOURCE・実行認可は別途必要であり、診断計画は `approved=false`、`runtime_authorization=false` のまま。

## 証拠分類

|保存record群|候補数|real/imagのsigned成分|独立forward端点|本監査の分類|
|---|---:|---|---|---|
|旧B0|12|discard/PF、両軸あり|corrected/exact_truncatedあり|保存成分を用いた将来の数値照合が可能|
|旧full PF（method B1）|8|PFのみ、両軸あり|correctedあり|固定full Hamiltonianのためdiscard成分を仮設しない|
|旧B0+I|12|total_signedのみ。discard/PFは未保存|overlay recordに直接はなし|保存端点からの復元と来歴照合を別承認後に実施する候補|
|追加B0|76|discard/PF、両軸あり|corrected/exact_truncatedあり|保存成分を用いた将来の数値照合が可能|
|追加B0+I|76|discard/PF、両軸あり|親forwardを同じscalarで回転した値あり|共有scalarを明記した経験的な照合が可能|
|合計|184|172候補344軸に成分あり、12候補24軸で未保存|存在は数値的一致の認定ではない|368軸全てを一覧化|

「保存成分あり」は今回closureや独立成分一致を再計算して合格したという意味ではない。
`complex_closure_discrepancy` の保存欄と過去の照合記録は参照したが、新しい引算・加算・絶対値和は実H8値に適用していない。
旧12候補はcorrected、exact_truncated、full-H reference、total_signed、full-Fock scalar receipt、対応する親B0 recordが存在する。
その意味で端点自体の欠測候補は0だが、**非相殺診断に受理済みの184成分集合はまだ存在しない**。

## 来歴と照合した固定bytes

- 旧H8 summary：原92候補の保存JSON（決定論20＋PR72）。公開済みD0 `3d910ee1de4597b8895e8506500553dbad5a5085` のinput receiptへ結合。1,392,287 bytes、SHA `dac86904b9a9d5d72146956ce2ed7aa3f9da52545169f5bbecb2af45c8ae6706` に一致。
- 旧B0+I：結果 `5f79257f432bc7e0317f122090f62fec379e2a1b` の `case_scalar_signal_receipts_v1.json` 12件とprovenance/終了監査を固定Git objectから受領。SOURCE `54ed98c5fa5b7190f77255271f4e3bec8db15630`。
- 追加152：Phase Aのgzip 138,903 bytes、SHA `3b067c5b28aa7b5b63fbe4bc4617e338aedc84861d6a06a020bc5f594213decd`。展開byte列2,734,696 bytes、SHA `1dcce70d78e7e53058acb43a1764f46c050ea6ca2ae26bbefbad4f4e9468a7a9` が元JSON export receipt・原出力inventoryに一致。SOURCE `fc463d9e07825f34e496acac6b40db7aec1aed6d`、provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。
- full-H参照metadataはPhase A input receiptと一致。新しいfull-H referenceや科学arrayは生成・読込していない。
- 184候補のmethod/prefix/q/formulaと固定登録集合を全件照合した。来歴資料とproducer sourceを含む26入力receiptを固定した。raw科学値は公開用成果物へ複製せずprivate snapshotに保持する。

Phase Aの35 workerは保存receipt上exit0/reap。`prefix_19` の実行時 `result_sha256` 欄だけが未記録で、公開時の別SHAが存在する。
この欠測は原記録のまま保持し、実行時にhashを観測したと遡及補完しない。
prefix19に依存する16候補（S2/S4×q1/2/4/8×B0/B0I）の来歴は、この明示した限定を引き継ぐ。
分類表の「成分あり」はruntime hash欠測が解消したという主張ではない。

旧B0/full PFのruntime profileには元記録の範囲を維持し、現profileを後付けしない。
旧B0+IとPhase Aが固定したprofileは `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。
科学条件・元SOURCE・compiler・結果・grant・ledger・消費・STOPを変えていない。

## 「成分を独立に確認できる」の範囲

producerの `differences()` はB0について次を保存する。

```text
discard_axis = truncated_reference_axis - full_H_reference_axis
PF_axis      = corrected_axis - truncated_reference_axis
total_axis   = corrected_axis - full_H_reference_axis
```

full PFはPFのみ。追加B0+Iはcorrectedと打切り参照を同じfull-Fock scalarで回転し、full-H参照は回転させず、同じ分解関数を呼ぶ。
旧overlay実装は同じ端点を保存しつつtotal_signedだけを書いたため、旧12候補では成分欄がない。
`exact_truncated` は変数名であり、数値計算された打切り参照を厳密演算・証明付き真値へ昇格させない。

correctedのnative/spin系とoccupation CSR/forward系は別経路の値を保存する。
しかし、`independent_forward_signals.exact_truncated` は同じoracleの打切り参照を共有しており、独立した別の打切り真値ではない。
CSR参照の実装はsplit-timeとeigen residualで経験的に検査するが、同じ数値libraryを共有し、成分別の独立証明書にはならない。
追加B0+Iの別forwardも同じCと位相回転を共有する。scalar誤差・reference誤差を独立な確率変数として扱わない。

旧B0+Iには8構成のnative correctness公開projectionもあるが、signal全12構成の別signed成分は保存されていない。
残4構成にはこのnative projectionを捏造しない。費用取得対象外であった履歴と、bias端点の保存有無を区別する。
保存成分の出所、経路差、成分別誤差証拠を確認するには、将来の承認済み照合が必要である。

## 通常bias・非相殺診断・厳密保証の区別

|区分|各軸の量|現時点の意味|
|---|---|---|
|通常signed bias|`abs(total_signed_axis)`|相殺を含む保存complex signalと参照の差|
|経験的非相殺診断|`abs(discard_axis)+abs(PF_axis)`、full PFは`abs(PF_axis)`|指定した分解に対する感度診断。数値成分同士の相殺を取り除く|
|厳密bias保証|真の対象signalとの差を上から保証する証明付き量|今回の保存証拠からは未認定、null|

通常量が小さい理由を「相殺」と断定する候補別数値判定や、相殺の度合い・ランキングは今回計算していない。
非相殺量も分解の粒度に依存し、PF成分内部の全相殺を解消するとは限らない。
経験的uを足しても未知の物理的truthに対する厳密bias上界にはならない。
H6 V1で旧B0+I12候補の分解欠測により `UNRESOLVED_FULL_DOMAIN` とした規則を、H8でも維持する。
H6の未解決事項を今回解決したと報告せず、H6係数・分解値も移送しない。

## 未解決事項と追加検証の必要性

1. 旧B0+I12候補24軸の分解を、保存端点から別承認後に復元する必要がある。成分sumとtotal、親B0・scalar/reference来歴を照合する。保存端点だけで進められる可能性があり、現時点で新signal/compileを要求する欠測は確認していない。
2. 172候補の保存成分についても、端点定義・closure・別forwardとの差を新しい数値処理として検査する必要がある。metadata監査合格をその代用にしない。
3. 共用reference・共用scalarと成分別誤差証拠の限界を維持する。新しい厳密certificateを要求する研究へ拡張する場合は別設計・承認が必要で、本準備には含めない。
4. prefix19のruntime SHA欠測を明示した来歴限定を保持する。後付けのPASS書換えや追加hashの時刻偽装はしない。
5. 費用比較にはB2全4軸PASSと正常終了の固定receiptが必要。B2不合格・途中STOPを未検証S4モデルで補わず、費用比較へ自動接続しない。
6. PR側にも非相殺のouter_PF/finite_RTE会計が必要。確認union6の全12軸に両成分・forward端点の欄があることだけを今回確認した。非相殺用shotやpaired費用・SEは未計算。

## 別承認後の実装計画

[未承認計画JSON](../../artifacts/resource_applicability/track_a_h8_bias_evidence_audit/2026-10-17/diagnostic_plan_UNAPPROVED_v1.json) に順序・STOPを固定する。
まず保存成分と必要な旧12復元を来歴付きで検証し、その後に初めて非相殺量を評価する。
各axisを別々に評価し、axis平均やcomplex normへ置き換えない。分解closureはH6の固定 `isclose(rel_tol=1e-12, abs_tol=1e-12)` を参照する案で、許容差緩和や実入力での不一致救済を行わない。

5精度ε=.05/.01/.005/.001/.0005、allowance倍率は**1と10**、α_axis=.025、u/headroom≤.01、既存normalization/shot算術を保持する。
通常biasの保存shotは非相殺診断にそのまま流用できないため、別承認後の保存値数値解析で診断用shotを再計算する必要がある。量子sampling追加とは別操作である。
保存uは経験的なcombined allowanceとして別に扱い、成分ごとに同じuを繰り返し加算したり、独立と仮定して二乗和で縮めたりしない。この扱いは厳密な誤差伝播証明ではない。

PR比較は確認union6/n32を維持し、outer_PF/finite_RTEの非相殺に対応したshotで保存paired wrapper費用を重み直す。
通常biasのSEを診断用へコピーせず、必要なpaired raw費用が欠ける場合はSEを欠測にする。探索n8とは混合しない。
費用はB2実測優先・B1固定model・保存実測の区分を保持し、P=0のRZを正確な有理数で比較する。
旧32集合と全184集合を分け、いずれも1候補の分解・適格性が未解決なら、その集合全体の最良とPR比はnull。
診断可能なsubsetの参考表は別名にし、全184候補の保守的最良・全候補実測最良とは呼ばない。

保存JSONを一度parse/index化し、component/axis結果をcacheして精度・倍率・比較集合で共有する。
現在は数MBのmetadataと小さい人工算術であり、worker poolを作るほど重い処理ではない。実測test wallは約0.17秒。
将来重い解析になった場合は実際の空きCPU/RAMとB2/M1-Sの非干渉を確認して並列化を選ぶ。Codex独自のCPU/RAM/worker上限は新設しない。
B2に既に固定された資源上限はそのrunの履歴・契約として保持し、本準備から変更しない。

## 人工検証・分離・STOP

[metadata監査module](../../src/trotterlib/h8_bias_evidence.py)、[監査CLI](../../scripts/audit_h8_bias_evidence.py)、
[人工限定の診断prototype](../../src/trotterlib/h8_bias_diagnostic_fixture.py)、[tests](../../tests/test_h8_bias_evidence.py)。
**人工30件PASS**。184/368件coverage、成分/端点欠測、片軸欠測、重複・scope違い、nonfinite、実入力を人工算術へ渡すことの拒否、相殺・非相殺・closure、axis別適格性、倍率1/10、厳密保証への昇格拒否、全domain欠測時nullを確認した。
metadata監査器は実値の和・差を計算しない。人工prototypeも正式な実H8用実行runnerではない。
実H8の数値pipeline・新OS observer統合・fresh実行gate・実行grantは未実装/未発行で、今回の成果をその実行認可へ拡張しない。

監査・testsはhome配下の独立snapshotで実施した。temp/logも同領域に置き、subagents0。
新worktree登録・Git ref/index/commit/pushの更新・B2/M1-Sへの書込やsignal送信・CPU affinity変更・既存venv変更・install/GPU query/useは0。
資料のsource-tree SHAとfile bytes/SHA、別途反映するpatchだけを固定し、公開や統合はB2終了後の別判断に委ねる。
今回の準備を完了し、mandatory STOP。

次の作業は、**ユーザー**が案内するpush commandで資料branchを公開すること。保存値の追加数値照合や実H8診断runner準備は未着手で、実施する場合に**ユーザー**が範囲を指定し、**Codex**がその指示内で対応する。
非相殺費用再会計・順位付けは、さらにその固定実装とscopeに対する明示承認後に限る。
