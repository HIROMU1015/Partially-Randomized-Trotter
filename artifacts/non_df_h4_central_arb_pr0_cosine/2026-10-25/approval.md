# PR0 cosine Arb one-shot 実行承認

2026-10-25、同じchatで準備完了後の「PR0 cosine / 64→96→128 bit、各C-target1＋compiled1、最大6状態作用」実行案に対し、ユーザーは「その方針で、次の作業に進んで大丈夫」と回答した。この承認を今回の限定pilotへ適用する。

対象は固定linear H4 / 隣接1.00 Å / STO-3G、Pauli-JW184項 / retained27、S2 / T=.8 / q8 / r4 / K4、PR index0 cosineだけ。保存32 event・raw state/ray・compiled QPYと元C exact operandsを使用する。新規compile/sampling/Hamiltonian/state生成は0。

64 bitから開始し、各精度でselected C-targetとcompiledの状態作用を各最大1回、初回科学計算は最大6状態作用。失敗callも非返還。精度ごとの2作用は独立したprocessで並行し、共通exact νで正規化する。B_savedは直接差へ一度だけ。

保存ρ予算 `19792741170690161541/7631344236317346665332736` を上端が満たす、または下端が超過した時点でSTOP。跨ぐ/非有限の場合だけ96→128へ進み、128でも未確定ならINCONCLUSIVE。幅1e-8は参考であり、判定後の高精度化は禁止。

科学SOURCE/runtime/実入力/契約を固定して新one-shot grantを発行・消費し、call ledger・native exit/reap・mandatory STOPを保存する。技術的中断は既存失敗call/SOURCE/grant/STOPを保全し、新SOURCE/未使用grantによる限定回復だけ許可される。成功済み作用の不要な再実行はしない。

PR1/sine/DF/R2/N_G再解析/正式certificate受理は禁止。旧SOURCE・結果・grant・ledger・marker・STOPを保全。局所数値包含を正式受理やPR全law保証へ昇格しない。commit/push・remote確認後mandatory STOP。

最新版policyの§10.3は提供fileに存在しない版不一致を記録し、実在する§8.4/10/10.1/10.2とユーザーの具体指示を適用する。CPU/RAM/worker数の恣意的な上限、共有環境・他ユーザーへの変更はしない。
