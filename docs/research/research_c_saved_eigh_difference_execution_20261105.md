# 研究C：保存H／H_eigh独立差分・固定4-call実行（2026-11-05）

**4 attempts／4 COMPLETE／FAIL0、retry/resume0。保存HとH_eighはexactには異なる。差のFrobeniusノルムをArb192で包囲し、独立診断を完了してMANDATORY STOPした。**

\[
8.696771106319332\times10^{-16}\ \mathrm{Ha}
\le \|H_{\rm saved}-H_{\rm eigh}\|_F
\le 8.696771106319333\times10^{-16}\ \mathrm{Ha},
\]
\[
\boxed{\|H_{\rm saved}-H_{\rm eigh}\|_2
\le 8.696771106319333\times10^{-16}\ \mathrm{Ha}}.
\]

十進表示は下向き／上向きに丸めた外側区間。正本は以下のexact rational endpointと [result.json](../../artifacts/research_c_saved_eigh_difference/2026-11-05/science-run01/result.json)。operator normの厳密値は求めていない。物理的無視可能性・原因の帰属・ground距離・信号影響は判定していない。

## 固定SOURCE・入力・実行範囲

独立branchは `codex/research-c-saved-eigh-difference-execution-20261105`、起点 `b7ebd03c087f986abdf24abd8be4303a28f02308`、tree `adc6982a8d5d164cf7ce55c60b7bd1dacdd83b64`。独立worktree `/tmp/prt-c-saved-eigh-difference-execution-20261105` で、ユーザーの明示実行許可後に固定scopeを一度実行した。

- 科学SOURCE：`4a3228881173710f9c6b4d0d13085a77fbc59668`。
- [SOURCE原本](../../artifacts/research_c_saved_eigh_difference/2026-11-05/technical/SOURCE.json)：SHA256 `0653edb67d03da3e1ea49abfb6780d9d36bbbbea996cc15e0d4305a7e0ec86ca`。
- [固定契約原本](research_c_saved_eigh_difference_contract_unapproved_20261105.json)：SHA256 `9b5c16785bebc3c047ef3f5feadee6fbacdbf8ebcdb475d1bddf5305703eca6b`。
- [準備・数学仕様・人工検証](research_c_saved_eigh_difference_readiness_20261105.md)。SOURCE14／producer9／input・provenance9件は無変更。

保存直鎖H4・1.00 Å・STO-3G、Nalpha=Nbeta=2、number CDF8・元順LD2、sector36、PR endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。関連固定政策はpartial ordered S2／s1/r32/K0／h=.1／m0／own-DF oracle・状態準備除外。この診断自体は保存行列の差だけで、時間発展・delta窓の評価は非適用。

Hは `artifacts/research_c_near_practical/2026-11-03/run01/physical/PR_update016.npz` のarray `H`、complex128／36×36、ファイルSHA256 `ebdebe8f2c3f156c8202834181e143298d667490b3f33fc611dc0fab97b976cd`。各real/imag binary64をexact rationalとして解釈した。基底は同run `physical_reference.npz` の `sector_basis_indices`、SHA256 `92a8d27bdebb53937f6361c106a13ef90b9110930d4bf4ddb577796cfb208739` へbindし、固定順序を検査した。

既存NumPy1.26.4／UPLO=Lの入力semanticsを表すBは、`B_ii=Re A_ii`、`B_ij=A_ij (i>j)`、`B_ij=conj(A_ji) (i<j)`。これは既存固有値問題の数学的定義であり、**H_eighを研究H_targetへ採択していない**。平均化、元Hの上書き、旧Direct m0の再実行は行っていない。

## 結果・数値資格

`D=H_saved−H_eigh` のexact Frobenius平方和は

\[
S=\|D\|_F^2=
\frac{32943178113}{43556142965880123323311949751266331066368}\ \mathrm{Ha}^2.
\]

Arb192の正本区間 `[L,U]` は、共通分母

`7067388259113537318333190002971674063309935587502475832486424805170479104`

に対し、下端分子

`6146345800899910140305754671777030870649494985287095518637`

と上端分子

`6146345800899910140305754671777030870649494985287095518639`

を持つ。固定runnerがexact arithmeticで `0<=L<=U`、`L²<=S<=U²` を検査した。`difference_exactly_zero=false`、`identical=false`、`rounding_enclosed=true`、status=`VALIDATED_SAVED_EIGH_DIFFERENCE_ONLY`。大きさによる成功閾値は使っていない。

|matrix identity|値|
|---|---|
|H_saved|`fd4c78d1a657e3995fb679da548e5cd2fb22a3b741a9a5b671d76b5b45a9d6d2`|
|H_eigh|`54403f56e771c7d984b4f856aa2646069278aae3537546d30558cad569d0c1c7`|
|D|`f6be67c5ee9f806ff85537c104c6b8648dffd6ac9545320523267b087d774d23`|

非zero差の存在とそのnorm boundが今回の取得事項。どの生成演算・入力が差を生んだかは未確定。gapを評価していないので、この値からground-state距離やRPEの十分条件を認定しない。実H4のeigensolve、ground certificate、spectral gap、X/Y、sampling、build/compile、fresh IID校正、Stage C、actual CA、重要科学レビューはいずれも0。対応するresult fieldsはnull／falseを保持する。

## Calls・原本・資源

実行開始はjournal時刻 `2026-11-05T03:26:37.276228+09:00`。同じ保存bytesをhash確認してからdecodeし、既存bridge・exact arithmetic・Callsを固定SOURCEのまま用いた。

|kind|attempts|COMPLETE|FAIL|
|---|---:|---:|---:|
|metadata_load|1|1|0|
|arrays_load_exact|1|1|0|
|eigh_semantics|1|1|0|
|difference_norm|1|1|0|
|合計|4|4|0|

Raw出力は固定fresh [science-run01](../../artifacts/research_c_saved_eigh_difference/2026-11-05/science-run01/)。11ファイル：contract、SOURCE、producer_binding、arrays、H_saved_exact、H_eigh_exact、difference_exact、result、science_calls、STOP、resources。入力NPZを追加ロードして再診断せず、数値はこの実行の出力だけから報告した。

[STOP原本](../../artifacts/research_c_saved_eigh_difference/2026-11-05/science-run01/STOP.json) は4/4、failed=null、retry/resume0、mandatory_stop=true。終了後のpassive preflightは `FRESH_NO_RETRY_RESUME` で拒否し、消費済みscopeを再実行しない。[実行確認](../../artifacts/research_c_saved_eigh_difference_execution/2026-11-05/execution_verification.json) にjournalと出力hash・SOURCE/input保全を記録した。

実行直前CPU affinity32、available RAM約52GiB、GPU deviceなし、disk空き約0.7GiB。単一36×36のO(n²) rational処理と依存4段階なので、固定CPU pathを使い、process poolやGPU転送を追加していない。CPU/RAM/worker/wallの人工上限や共有環境変更なし。Raw [resources.json](../../artifacts/research_c_saved_eigh_difference/2026-11-05/science-run01/resources.json) はwall `0.13872612174600363` 秒、CPU `4.034274175` 秒、peak RSS `52448` KiB。別の[launch終了記録](../../artifacts/research_c_saved_eigh_difference_execution/2026-11-05/launch01/exit_status.json) はreturncode0、wall約0.286秒、起動1回。これは今回の小行列経路の記録で、性能改善率のbenchmarkではない。

## 保全・証拠status・次の判断

SOURCE・契約・固定入力とraw内snapshotのbytes一致を確認した。旧Direct m0のSOURCE/input等41ファイルも無変更で、旧15-callの3 attempts／2 COMPLETE／1 FAIL、残12転用禁止、retry/resume0のSTOPは維持。準備worktree・元dirty checkout・並行作業のcache/lock/processを変更していない。[保全確認](../../artifacts/research_c_saved_eigh_difference_execution/2026-11-05/preservation.json)。

準備時の人工32 PASS／関連127 PASS・継承4 FAILは準備報告の証拠であり、今回の新しい回帰実行とは呼ばない。今回は不変SOURCEによる実4-callの完了と、bytes・journal・出力statusの確認を行った。LocalのArb包囲結果であり、CIや外部独立再現ではない。研究概要、VALIDATION_STATUS、dated note、manifestへこの限定結果だけを追記する。

公開後にremote SHA/tree、報告全文、主要blobを照合し、[public_verification.json](../../artifacts/research_c_saved_eigh_difference_execution/2026-11-05/public_verification.json) へ記録する。

**MANDATORY STOP。次はユーザー／研究Cが保証対象H_targetと参照条件を判断し、Codexは具体的依頼後にground-state certificate等の独立契約を準備する。新しい科学実行と重要科学レビューには、それぞれ適用する明示許可が必要。今回から自動進行しない。**
