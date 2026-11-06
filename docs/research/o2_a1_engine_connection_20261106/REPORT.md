# A1 N2補完：測定engine技術接続・一様誤差評価の準備

2026-11-06 JST、SERVER1 GPUA100-gleap-01。
**最小adapter・一様保証方法・人工correctnessを準備した。実科学入力でのgate誤差／総engine上界は未評価でUNKNOWN。公開照合後mandatory STOP。**

保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子number sector/L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、理想gate・保存DFモデル限定。
N1 actual finite-grid Q・selected basis policy、N2の保証済みgroundとstate/spectral witnessを維持する。

## 固定資料と確認範囲

| 資料 | 固定commit |
|---|---|
| N1結果・SOURCE・完全payload | `744e1a09661ebba6841fe5faf456f88c41c47780` |
| N2結果・SOURCE・state/witness | `5e9bd08068738cdf1070bdc43bee8194b113ae33` |
| N2科学SOURCE | `a4141795bd3e23670612d0f0d4a43678457a1a37` |
| 新finite-target契約 | `2b2e6c1bc3b2d201d1a6d1238172187d68aea6ab` |

N1/N2の各REPORT・SOURCE・保存STATE/ENGINE結果を読んだ。[STATIC_INPUTS](STATIC_INPUTS.json)に固定commitの25ファイルのbytes/hash・identity照合を保存した。
実データのmatrix復元、norm計算、上界評価、state作用関数を禁止した状態で静的検査した。NPZ load、preparation/CDF、state solve/eighは再実行していない。
前回N2のREPORTが参照するRUN.logは固定N2 commitに含まれていなかった。原worktreeの[既存log](N2_RUN_LOG_RECOVERED.txt)をbytes/hash付きで回収した。旧commit/結果/STOPは変更せず、科学再実行もしていない。

## 実装・保証方法・人工結果

[ENGINE_CONNECTION](ENGINE_CONNECTION.md)に、保存basis復元、既存q1 boundary_optimized・selected plan・RTE/Hadamard builderの再利用、ordering/phase、state normとBorn/bitへの対応を記録した。
[新adapter](../../../src/trottertracks/algorithm_codesign/a1_engine_connection.py)は共有coreや固定SOURCEを変更しない。
保存RZ/XXPlusYYのnamed parameterと局所matrix hashを照合し、選択・係数・qubit順序を保存値へbindする。未対応gateや不一致ではSTOPし、basis再構成/eighや別targetへのfallbackはしない。

[ENGINE_ERROR_PLAN](ENGINE_ERROR_PLAN.md)は実装と対応する全supportの保守評価を定める。
全saved full/support basisとinverse、固定D半step、有限RTE sign/order/support variant、wrapperの局所gate集合を検証し、no-cancellation列長capで全pathを覆う。
各局所matrixと同じexact-angle ideal gateとの差δ*、最大gate列長LからE≥(1+δ*)^L−1を外向きに評価し、gate生成・最終正規化の座標誤差を4Eで抑える。
標準Fractionの小さな局所contractionとexact Born比により、gate適用・Born積和/除算の丸めはこの実装では0となる。未知のBLAS誤差を0として代用する方法ではない。
初期normはBorn比で相殺し、最終normの違いは4Eへ課金する。確率範囲外をclipしない。

総上界の取得式は **既存b_state≤7.81e-7＋4E＋条件付き2/2^53**。
この式は方法であり、実N1 gateによるδ*・L・E・総上界は今回求めていない。実分子の代表path比較から一様性を認定しない。
草案のb_engine≤1/32は未採択の候補のまま。新しい精度・失敗確率・測定予算は採択していない。

新規人工17件と既存A1人工55件、[計72件PASS](TESTS.json)（[log](TESTS.txt)）。
人工2-mode DF fixtureの全path・両軸について、既存のfull repeated/selected/Hadamard builderとliteral gate名・parameter hex・qubit位置・順序が一致した。
独立statevector作用との照合、β≠0のXXPlusYY intervalの独立高精度exponential照合、known phaseのX/Y、norm処理、確率range/bit-grid endpoints、誤差の包含、identity/matrix変更時の棄却、N3禁止callの隔離を確認した。
これは人工correctnessであり、実分子数値取得・CI・独立外部再現の証拠ではない。

## 残る作業と次の承認対象

**次は、以下の限定的な実数値qualification一回だけの別承認が必要。今回未実行。**

| 項目 | 次回の限定scope |
|---|---|
| 入力 | 固定N1 PREPARATION/POLICY/CONFIG/LAWと固定N2 identity/state保証・保存vector/witness。NPZはloadしない |
| 関数 | 公開固定SOURCEの `qualify_fixed_bundle(root,fresh_output)` 一回 |
| 数値処理 | 保存bundleの復元一回、D block4件のhalf-step局所列を各一回、scalar列一回、有限局所gate集合のunique keyごとにinterval検証一回、δ*/L/E/総上界の合成一回 |
| 0回 | 実state norm・信号・作用評価、実path build/全列挙、preparation/CDF、solve/eigh、RNG/shots、compile、variance/N/費用評価 |
| STOP | identity/profile/matrix/selector不一致、未対応gate、非有限interval、E≥1、fresh出力衝突。条件緩和・代替kernel・retryなし。成功時も公開照合後STOP |

正確な関数・内部回数・出力は[実行仕様](ENGINE_ERROR_PLAN.md#別途許可が必要な限定科学実行提案)と[機械可読plan](NEXT_EXECUTION_PLAN.json)に残した。
N2のstate保証を再利用するので、新しいstate solveやspectral検査は不要。
実stateに対する`probability`呼出しやfresh-bit facade、train/cal/prod、価格評価はこの次回qualificationにも含めない。

数値取得後、総engine保証・実行モデル・精度/予算の採否を判断し、N3を別に許可する必要がある。測定facadeはengine内部のp/vector/E/gap等をCV側へ渡さず、fresh bitだけを渡す接続が必要である。
exact kernelを別の浮動小数点kernelへ替える場合は本保証を流用できない。実科学入力での性能・古典費用も未評価。

## 保全・停止

[SOURCE](SOURCE.json)、[PROFILE_LOCK](PROFILE_LOCK.json)、[保全記録](PRESERVATION.json)、[STOP](STOP.json)を保存した。
元mainのHEAD/statusと既存dirty/untracked 2,136ファイル、N1/N2の固定payload、旧A1 SOURCE/結果/STOPを保全する。C1は凍結のまま。
共有venv/package/cacheや他ユーザーjobは変更せず、自分専用cacheを使用した。CPU/RAM/worker数の人工上限は設けていない。
実科学入力の新規numerical call・N3・重要科学レビューは0件。g2/g3の希少性、control集合・pilot数を維持する。
専用branchでcommit/pushし、remote SHA・本文/blob/treeを照合してmandatory STOPする。
