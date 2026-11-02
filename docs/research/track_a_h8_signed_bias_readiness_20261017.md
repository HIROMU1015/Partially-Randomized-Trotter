# H8 signed bias成分の復元・数値整合性検証：実装準備

**数値検証runner、人工検証、実行契約、無効grant案を準備した。保存H8値での復元・数値照合は未実行で、別途明示承認が必要。**

- SOURCE：`64ec6cbba7e48ac7c04a6ad0b38c3e76f0f14526`
- 起点の証拠監査：[`b75039c1d5ead475d8c0ff8b62dba04a9978a6fa`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b75039c1d5ead475d8c0ff8b62dba04a9978a6fa/docs/research/track_a_h8_bias_evidence_audit_20261017.md)
- Phase A結果：`01309059636f2432e29b663bcd2eb0e400f5ae0a`
- [実行契約・無効grant・SOURCE closure・人工検査・保全記録](../../artifacts/resource_applicability/track_a_h8_signed_bias_readiness/2026-10-17/README.md)
- [数値module](../../src/trotterlib/h8_signed_bias.py)／[契約・入力読取](../../scripts/h8_signed_bias_contract.py)／[runner](../../scripts/run_h8_signed_bias_validation.py)／[人工tests](../../tests/test_h8_signed_bias_validation.py)

H8直線鎖1.0 Å/STO-3G、保存tol-only DF28、16 modes/sector4900、T=.8。
旧32決定論候補と追加152候補、計184候補368軸の登録集合を維持する。
旧B0/B0+Iはprefix7/14/21・q1/2/4/8のS2、旧full PFはprefix28のS2/S4。
追加B0/B0+IはS2 prefix8〜20（旧14を除く）とS4 prefix15〜21、q1/2/4/8。
データ内のmethod `B1` はfull PFを指し、費用fitのPhase B1とは別の名前である。

## 今回実行したこと

人工端点と人工byte fixtureだけを用い、新38件＋既存回帰30件、**全68件PASS**。
test全体wall約1.55秒、child peak RSS約55.3 MiB。保存H8を使った実行時間やメモリ実測ではない。
人工suiteの結果は[限定検証記録](../../artifacts/resource_applicability/track_a_h8_signed_bias_readiness/2026-10-17/limited_validation_v1.json)に固定した。
追加の実行保護として、draft grantを実runnerに渡したときに、入力読取・出力作成より前に拒否することも確認した。

SOURCEと旧producerのtext、および必要なJSON fieldのschemaを読取調査した。
保存値に対する差・和・位相適用、closure、forward差、非相殺shot、費用・ランキングは計算していない。
SCF/DF/state、新signal/compile/sampling/fit、新有効grant、GPU query/use、venv変更は全て0。
レビューは実装者自身による限定確認であり、別担当の独立reviewや追加GPT/Claude科学レビューとは呼ばない。subagents0。

## 別承認後の固定処理

|対象|候補／軸|行う検査|
|---|---:|---|
|旧B0+I|12／24|保存scalar・親B0・corrected/truncated・full-H reference・totalを照合し、未保存signed成分を復元|
|既存成分|172／344|保存成分と端点差、component closure、保存forward端点を各軸で照合|
|全登録集合|184／368|重複・欠測・scope違いを拒否し、各軸の結果と来歴を記録。候補を除いて全体PASSにしない|

各real/imag軸で固定する定義は以下。

```text
discard = exact_truncated - full_H_reference
PF      = corrected - exact_truncated
total   = corrected - full_H_reference
```

full PFは `PF = corrected - full_H_reference` だけを持ち、discardを仮設しない。
保存されたtotalと各signed成分をそれぞれ端点差へ照合し、component和とtotalのclosureも別項目で検査する。
誤った成分が相互に相殺してclosureだけ通る場合も不合格になる。

旧overlayでは保存端点から得た成分を `RECONSTRUCTED_FROM_SAVED_ENDPOINTS` として扱い、元rawへ追記しない。
既存172候補の保存成分と、今回新たに復元する成分を出力上も区別する。
不一致があっても184候補の診断記録を揃え、最初の不合格理由を保持した上でterminalをSTOPとする。
入力bytesや全体decodeの不合格は数値照合前にSTOPし、未実行を記録する。

## scalar・SOURCE・reference

旧監査の26入力receiptを再利用し、SOURCE/入力SHA/旧compiler profileを結合する。
今回、それらの保存H8値に数値処理を適用していない。実行時には全入力bytes/SHAを再照合する。
Phase Aのgzipは展開byte列の元SHAも確認してからparseする。原producerのPython textもbyte照合するがimportしない。

- Phase A SOURCE：`fc463d9e07825f34e496acac6b40db7aec1aed6d`
- 旧B0+I SOURCE：`54ed98c5fa5b7190f77255271f4e3bec8db15630`
- scientific provider SOURCE：`c31ef424175c02379eafebe8fc6c7548c1d9aabe`
- 保存profile：`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`

保存されたfull-Fock scalar `C`、exact stored rational、scalar error receiptを照合する。
親B0のcorrectedとtruncatedへ `exp(-i*0.8*C)` を適用した値と保存overlay端点を比較する。
full-H referenceは回転せず、親の参照および固定full-H referenceとの一致を確認する。
DF行列を読み出してCを再導出する処理、scalar抽出や化学入力の再生成は含まない。
保存rational receiptへの一致は元scalarの独立な再導出・物理的厳密証明を意味しない。

各軸には入力role/SHA、元SOURCE、親・scalarとの関係、保存経路の範囲を記録する。
旧B0/full PFの未記録runtime profileに現profileを後付けしない。

## 許容差と証拠の限界

旧監査のclosure案 `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` を継承する。
この新契約では、端点差・total・成分・closure、scalar位相の端点照合、非回転reference、forward端点にも同じ値を事前固定した。
real/imagを独立に比較し、complex normや軸平均による救済をしない。
保存H8値の結果を見て許容差を調整していない。将来の不一致を緩和や別参照への置換で救済しない。
scalarのstored-rational算術については、保存済みerror receipt内であることをFractionで検査し、別の誤差幅を新設しない。

forwardのcorrectedは別経路の保存値だが、truncated referenceとB0+I scalarは共有される。
追加B0+Iでは保存forwardを親forwardの位相回転にも照合する。
旧12 overlayには別forward欄がないため、保存親B0のforward整合性を検査し、overlayの独立forwardが保存されていたとは報告しない。
旧native correctnessの公開projectionは8構成のみ。残4構成にそのPASSを補わない。

Phase A prefix19の元runtime SHA欠測（16候補）は、公開時の別hashと区別して保持する。
数値照合が将来PASSしてもこの来歴限定は消さない。
成功statusは経験的数値整合性のPASSであり、厳密bias保証や完全に独立した成分証明はnullのまま。

## 実行契約・資源・STOP

既定CLIはscope表示だけで保存H8を読まない。

```bash
python -P -B scripts/run_h8_signed_bias_validation.py
```

実処理には別途認可済みgrant、固定plan digest/SOURCE、実装6 closure filesのbytes/SHA、既存Python fingerprint、全入力receipt、fresh gateが必要。
draftは `approved=false`、`runtime_authorization=false`、CPU未予約、input/output未指定、absolute command nullのまま。
fresh gateの有効期間300秒は起動前確認の鮮度であり、数値処理の実行時間上限ではない。

処理は数MiBのscalar JSONを一度parseし、候補でindex化して両軸に使う。軽い処理のためworker poolや新observerは設けない。
CPU/RAM/worker数・phase/total wallにCodex独自の上限を設定しない。既存ジョブの資源契約も変更しない。
出力見積4 MiB・新規4 inode、RSSの概算128 MiBは予約上限ではなく、実H8実測でもない。
SOURCEとPython fingerprint、CPU/memory/pressure/OOM・空きblock/inode/quota等の非干渉を直前確認してgrantに結合する。
runnerは新たなCPU affinityを設定せず、共有環境・他ジョブを変更しない。

新exclusive outputと永続one-shot lockを使い、raw・旧結果・grant/ledger/markerを上書きしない。
自分のSIGINT/SIGTERMに対してSTOPを保存し、child processは起動しない。
最終input/source再照合を通るまで成功扱いにせず、Pythonのnative exitは呼出元がその場で別途記録する。
terminal.jsonを過去のnative wait証拠に見立てない。自動retryなし、成功・失敗いずれもmandatory STOP。

B2の終了/PASSは、このsigned照合の前提条件ではない。
今回の保全確認時のB2はMEMORY_PRESSURE STOP・全回収記録・元PID不在であり、既存worktree/7主要証拠のhashを保全した。
B2 STOP・SOURCE・結果・grant・ledger・markerを変更せず、M1-Sや他worktreeにも書き込んでいない。
非相殺shot・PR費用・最良候補・Phase Cへの接続は本scope外。

## 次の担当と必要な認可

**ユーザー**：上記SOURCE・184候補368軸・固定許容差による保存成分復元と整合性検証のone-shot実行を、別途明示承認するか判断する。
**Codex**：承認後にinput root・新output・fresh gate・grantを結合し、一度だけ実行して結果/欠測/STOP/native exitを保存し、mandatory STOPする。
この準備依頼を実H8数値再解析の認可には読み替えない。
