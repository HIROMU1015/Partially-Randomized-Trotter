# N3保存帰属v1：数値事後評価・選択証拠再認証の準備契約

2026-10-13 JST。完了済み科学レビュー第12.6節の認可境界に対応する。**source・仕様の準備のみ。実行は未認可。**
原run1一回の消費済み認可を使わず、別scopeに結合した明示指示を受けた場合だけ実行する。

## 固定データ・対象

原結果payload6d8f769b、公開basef07591a9、数値source3eba0814を維持する。
[保存データregistry](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/saved_input_registry.json)の40 filesをSHAで結合。
元6H、full5/6、ν2/3、非負density、一外部mode、候補2mode/H、元ε_proj=.01。geometry/basis/DF/L_D/PF窓なし。
OCC・SDP_EQUIVALENT・DENSITY_SOSの216保存mode-path、432保存solver events、696保存候補を解析対象とする。

1. 保存phase totalsの合計、row totalとの差、backend solve/setupとの混合未割当残差を求める。sourceで分離不能な原因を推定しない。
2. 保存候補のg=c−U、q=κU−s、box=max(0,κ−g)、root/box/zero活性をexact scalar arithmeticで整理する。
3. SOSは保存丸めℓとeからq=(κU−ℓ)+eを分ける。OCCは保存τ/Zからs=ντ−trZを照合する。修復前proposal/rがないため、repairへの完全因果帰属は未判定とする。
4. 既存の[小型証拠接続](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/SOS_OCC_existing_proof_connection.json)の3点を凍結checkerで再認証する。external4のSOS正証拠、同mode/δ=1/400のOCC負証拠、同Hのexternal0のOCC正証拠。

証拠を変えず、元SOS witnessの妥当性と同じtargetでのscalar条件を確認するだけ。新κ/δ探索や証拠生成をしない。
同modeでの条件族分離をH単位追加縮約や物理的不能へ読み替えない。
旧runの冷時間を新workerの実行時間で置き換えない。今回のposthocは新しい確認的移送実験ではない。

## source・実行gate

moduleは`n3_saved_attribution_v1.py`と`n3_saved_attribution_execution_v1.py`、runnerは`run_n3_saved_attribution_v1.py`。
defaultはsource/input/planのidentity検査だけ。数値事後評価は`--execute --authorization`、専用clean execution branch、別認可を必要とする。
execution branchは`n3-saved-attribution-v1-execution-20261013`、fresh outputは`artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1`。
producer、NumPy/SciPy/CVXPY/SCSをimportしない。新solver/eigensolver/最適化/元batch再実行/新入力は0。

CPU2spawn workers/thread1、AS1GiB/worker、aggregate sampled RSS6GiB、開始前available8GiB。
標準library JSON/Fractionを使用し、一caseを一回decodeしてphase/scalar整理へ共用する。
既存のexact checkerとMemoryWatchをsource SHAで固定して再利用。GPU driverは照会で利用可能性を確認できず、GPU対応を主張しない。
input総量64MiB/file8MiB、output総量32MiB/file4MiB/log4MiB、停止記録64KiB予約、rational4096bits、候補はcaseあたり最大1000、選択証拠再認証3点。
既存checkerのoperation/bit上限も適用。wall/CPU capなし。attempt1/retry0。

## 検証・完了・STOP

synthetic23 PASS：未評価literal抽出、tier slot欠落拒否、source/pointer/copy意味論、rational/bool/bit拒否、box/root/zero/tie、残差会計、反実仮想費用を出さないこと、認可/path/overwrite拒否。
非負densityの合成例でSOS正/OCC負/他mode正の3証拠を再認証し、改変を拒否。実2spawn/AS1GiB経路とserialの合成結果が一致した。
testsは登録raw/registryへのreadを拒否するguard付き。登録6Hの新数値事後評価・PSD再認証は0。
実行認可を受けた場合も、一次事後結果・失敗・監査・公開・remote確認後にMandatory STOP。
同機能/同品質の次benchmark、研究の収束・中心claim、task変更はGPTの科学判断へ戻す。
