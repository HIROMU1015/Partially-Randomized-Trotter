# P3 Stage 3：Givens数値安定化とユーザー指示による新attempt

2026-11-02 JST。基点は失敗結果`aab0b483fdd8e57fe17305a11c5cb750a9c80a4d`。
ユーザーの追加の「実行してよい」を、原因調査・必要な技術修正と同じStage 3初期scopeの新attemptへの許可として扱う。
旧run01/STOP/科学SOURCEは保全し、失敗をPASSへ書き換えない。新attemptの保存先はstage3/run02。

## 修正対象と維持する条件

qiskit-nature0.7.2の数保存Bogoliubov分解には、要素の`np.isclose(...,0)`による省略と、
Givens cosineからの`arccos`角度変換がある。前者は小さな非零成分を落とし、後者はcosineが1へ丸まる微小角を失い得る。
人工例でこれらの弱点を検査し、project内の限定した数保存Givens関数を追加する。
元の分解順、`U.T`規約、RZ/XXPlusYY gate familyを維持し、exact-zero判定と`atan2(|s|,c)`だけへ変更する。
共有qiskit-nature/NumPy/package/venvは変更せず、dense fallback・新solver・高精度化を使わない。

これはnative合成の実装variantの変更であり、旧経路と同じ価格programと呼ばない。
初回は価格未取得なので、旧価格との混合・再ランキングはない。両armの保持とDF residual basisに同じ修正版を使う。
旧H4差の原因を、人工例だけで単一機構へ断定しない。新attemptのpreflightで実H4への適合を確認する。

Hamiltonian、保存辞書・U/η、時間、q/r/K、seed8件、compiler、P=0、wrapper、F/E/R/Emptyは変更しない。
特にone-particle d8のτ/4・4τ、full-space operator・長いprogramの誤差政策を緩和しない。
新しい科学条件・判定閾値の採択は行わない。

## 実装・新SOURCE・実行範囲

[安定化関数](../../src/trotterlib/stable_gaussian_basis.py)は既存Apache2.0の数保存Givensアルゴリズムを最小修正し、出典・変更点をSOURCEに明記する。
[薄いrunner](../../scripts/run_h4_p3_stage3_stable_basis.py)は既存Stage 3 runner/kernelを再利用し、
自分のprocess内のlowering依存だけを差し替え、新variantを来歴へ記録する。
元のrunner/kernel/共有libraryを書き換えず、旧SOURCEの再現可能性を保つ。
[人工tests](../../tests/test_stable_gaussian_basis.py)と既存Stage 3 testsを同じ修正経路で実施する。
SOURCE固定前の最終人工検査は、新規安定化8件＋既存Stage 3の12件＋bit-copy回帰1件の計21 PASS。
近零8-mode回転、複素回転・reflectionで、同じ厳しいoperator基準への適合を確認した。
これは人工実装検査であり、H4差の原因同定や科学的PASSの代用ではない。

人工検査後に新しい科学SOURCEをcommit固定して公開し、run02を1回起動する。
最初に共通保持/basis/scalarを検査し、全PASSなら同じ8 trajectory・256 event位置を一度sampling、
native body18件のPASS後だけ同じ36 wrapper compileを行う。
ユーザー指示による追加attempt1回として記録し、automatic retryは0。run02も失敗した場合は追加retryせず停止する。

QPY・結果・preflight・未到達範囲・SOURCE来歴をrun02へ保存し、公開・remote照合後mandatory STOP。
新signal/B/N/G、finite-bit実効平均、追加標本・Stage 4・主要claim採択・論文改訂は行わない。
