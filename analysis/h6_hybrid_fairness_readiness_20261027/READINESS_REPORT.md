# H6 Hybrid PR：固定4候補のPhase A準備

2026-10-27 JST。基点`f7757c4320ba34dd36a64de4250c29a69bf38390`、branch `codex/h6-hybrid-fairness-extension-readiness-20261027`。
**Phase A完了。技術接続・人工テスト・静的入力照合を終えた。実H6の新signal/B/N/G、標本数の数値算出、sampling、build/compileは0。Phase Bは未承認・未実行のままmandatory STOP。**

## 対象と成果物

[FIXED_CANDIDATES.csv](FIXED_CANDIDATES.csv)の4物理候補：p78/q8/r16,K2、p78/q8/r32,K2、p78/q8/r64,K2、p684/q1/r16,K4。すべて未実行。5精度の新適格性や費用をまだ判定していない。
固定H6/1.00Å/STO-3G/918非identity項＋constant、S2/T=.8、同じraw ray/reference/exact norm、central control、Qiskit1.3.0/opt1/seed17、仮想ρ=0の条件を維持する。

- [EXECUTION_PLAN.md](EXECUTION_PLAN.md)：Phase Bの順序、精度/call範囲、標本設計、旧SE floor、停止条件。実行無効の設計であり、sampling数値capは未確定。
- [SOURCE_INPUTS.json](SOURCE_INPUTS.json)：再利用code SHA、固定入力ref/SHA、保存norm、公開用runtime版、将来の独立出力先。
- [SAVED_COMPARATOR_STATISTICS.csv](SAVED_COMPARATOR_STATISTICS.csv)：旧2 Hybrid anchorと純RTEの保存G/SE等15行をそのまま引用。
- [TEST_RESULTS.json](TEST_RESULTS.json)：人工/mock27件の結果。費用優位の科学的証拠ではない。
- [専用adapter](../../src/trotterlib/h6_hybrid_fairness_extension.py)、[接続テスト](../../tests/test_h6_hybrid_fairness_extension.py)、[包含oracleテスト](../../tests/test_h6_hybrid_fairness_extension_mean.py)、[静的準備script](../../scripts/prepare_h6_hybrid_fairness_extension.py)。

## 再利用と差分

元`h4_pauli_pr_search.law_descriptor`をprivate namespaceへ抽出し、validator接続だけを追加した。実H6公開entrypointは固定4候補に限定し、旧r≤8 validatorとprefix0専用のevent-extension wrapperは変更していない。2/3qubit人工entrypointは別に設け、保持/tailが両方空でない系を使った。

再利用した科学処理は、PauliSignalAdapter、finite-RTE sampler/CDF-grid、AuditedPauliSignalAdapter（同じseedでq*rを一度だけsampleしq blockへ分割）、tight L2 mean、D2 central control、Hadamard wrapper、default compiler、H6任意精度整数shot会計である。
価格とG統計は元`cost_summary/paired/_statistics`を再利用する設計。free-CPU dispatch、price_job、QPY-aware pack/restorer、native終了処理も既存を次段階で用い、新しい管理基盤は作っていない。

descriptor内のlegacy `cost_samples=8`は元configとの互換metadataで、今回の探索n_Eを決めた値ではない。実sampling数はPhase Bの設計で別途固定する。
SOURCE入力のbyte SHAを確認し、元modelの12qubit/918項/全Z-prefix78と保存norm文字列の一致を確認した。Normを新しく計算していない。event拡張のinputコピーはarchive内にあるため、元Git blobのSHAと既存SOURCEのコピーSHAを照合し、巨大rawを復元したとはしていない。
初回の静的scriptはarchive内コピーをloose Git fileとして読むpathで停止した。保存SOURCE照合へ修正して再実行しPASS。科学callは発生していない。

## 人工検証

| 検証 | 件数 | 結果・範囲 |
|---|---:|---|
| 新専用adapter | 8 | 固定4 dispatch、H6 identityのmock、旧r8 law一致、保持/tailとCDF、cohort addressの引数、extended bodyの元sampler一致、両軸compiled signalとraw norm、shot/paired covariance |
| 新mean接続 | 2 | 4つの2qubit Hybrid fixtureでtight L2が既存128bit Arb oracleを包含。B一回、同じexact norm、再正規化なし。実寸入力は人工entrypointで拒否 |
| 再利用D2 control | 12 | 小系の非可換順序、符号、order0/2/4、global→relative phase、wire、wrapper、誤実装検出 |
| 再利用H6 shot | 5 | 既存guard・allowanceを変えず、大きな整数・exact ceil・G恒等式を人工検査 |

計27件PASS、各processはexit0・回収済み。人工testだけでlawを構築し、2/3qubitのbuild/compile/状態作用を行った。**実H6の新law・signal・B・N・G・seed生成・sampling・回路生成/compileは行っていない。**
Compiler側runtimeは既存Qiskit1.3.0/NumPy1.26.4/SciPy1.14.1を読取再利用した。旧数値venvの参照先はこの環境にないため、python-flint0.9.0だけを新worktree内の非公開依存directoryへ追加し、人工oracleを同じ固定版で検証した。共有venv/他研究環境は変更していない。Qiskit1.3の既存deprecation警告は科学的不一致ではなく、版変更もしなかった。
これはprimary Codexの技術照合であり、独立した科学レビュー、実H6性能検証、全law実compiledρ保証、正式certificate受理ではない。

## 標本計画の準備状況と残り

旧Hybrid/RTEの確認n48、Gとpaired SE、旧Hybridの軸別SE/paired price配列の所在を確認した。元metadataには分散・共分散の独立fieldがないため、今回は値を新計算せず、その復元をPhase Bに残した。保存G/SEや既存14/15順位の再会計もしていない。
計画は、新signal適格性の確認後、sampling前にn_Eと上限、確認対象/nominee規則、n_C決定式とcapを固定し、探索のpaired分散から最終n_Cを一度だけ決める。確認結果を見た恣意的top-upは行わない。
目安は名目95%半幅1%だが、旧比較側SE floorと希少event・有限標本を明示し、判別不能ならINCONCLUSIVEを残す。数値n_E/n_C、比較差区間、break-even、新G予測はまだ算出していない。

## 次の担当・必要な承認

**ユーザー：Phase Bの固定4候補に対する限定科学実行を許可するか判断する。** 許可範囲には候補固有law/mean/B/bias/u/N、保存価格からの分散・共分散と標本数の数値設計、適格候補の新探索・独立確認、G/SE比較・保存・公開を含める必要がある。
許可後のCodexは、sampling前の数値capとSOURCEを固定し、同じ範囲の通常段階では再承認を挟まず実行する。Phase A commitをそのまま実行許可とは解釈しない。
旧SOURCE/raw/grant/ledger/marker/STOP、最新14/15資料、別研究/B3/他worktreeは変更していない。今回の準備を公開・remote照合したらmandatory STOPし、実行・主要claim採択・詳細レビューへ自動移行しない。

## 固定資料

- [公平性調査と4候補案（f7757c43）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7757c4320ba34dd36a64de4250c29a69bf38390/analysis/h6_hybrid_search_fairness_20261027/REPORT.md)
- [元Hybrid研究（8fd1ceb2）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/docs/pauli_h2_h6_resource_comparison_20261025.md)
- [H6 fairness（288d2206）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/docs/h6_rte_fairness_20261025.md)
- [純RTE event拡張（10cb3e5c）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md)
- [総合資料（b6c89969）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md)
