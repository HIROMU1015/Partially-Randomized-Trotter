# H6 full m5_best：保存scalar位相台帳の一回照合

2026-10-19。**一回照合は正常終了。保存角度・global phaseから求めたdeterminant位相差は約1.486×10⁻⁹ radで、保存された共通位相と同程度だった。発生passは未確定で、元のraw FAIL・STOPは維持する。** 追加compile・回路再生・費用受理は0。結果資料をlocal commitし、mandatory STOP。pushは行わない。

SOURCE：`f0022cec13be6f959b1a7b9425a9ba55657542fa`。plan SHA256：`b24b9fea32985e90354a5e3d92d1e71f87c72e5ef9ca981f3d0a96943e00b28e`。branch：`track-a-h6-m5-phase-ledger-run01-20261019`。

[前回の静的調査](track_a_h6_m5_phase_ledger_static_20261019.md)、[保存4再生の結果](track_a_h6_m5_saved_phase_run01_results_20261019.md)、[今回の結果bundle](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/summary.json)。以下の実数は承認済みrunner出力の転記で、終了監査・資料作成での再評価ではない。

## 1. 承認と実行範囲

直前の「保存gate列・位相帳簿の静的調査」に続く利用者の「次の作業に進んで大丈夫」を、この固定scalar照合の実装・人工検証・fresh gate後の一回実行への承認として記録した。[scopeと承認の公開projection](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/scope_and_authorization.json)は、privateの正本plan・承認receipt・消費済み新grantのSHAへ結合している。公開projection自体は正本planや再利用可能なgrantではない。

対象はH6直鎖・隣接1 Å/STO-3G/tol-only1e-8 DF19、T=.8/P=0、full m5_best/prefix19/q2、symmetric-directional/full Hadamard **cosine**の保存wrapper/compiled各1回路。13 qubit・ancilla12・layout None・最後の測定だけという既存構造を維持した。sine・別状態・別候補は対象に含まない。m5公式IDは`PF_M5_BEST_20D64C2_BINARY64_V1`。

新しい[core](../../scripts/h6_m5_phase_ledger_core.py)、[contract](../../scripts/h6_m5_phase_ledger_contract.py)、[launcher](../../scripts/run_h6_m5_phase_ledger.py)、[metadata準備器](../../scripts/prepare_h6_m5_phase_ledger.py)、[人工検証](../../tests/test_h6_m5_phase_ledger.py)を固定した。新scopeは`H6_M5_SAVED_DETERMINANT_PHASE_LEDGER_V1`。旧4再生・元compileのgrant、ledger、marker、STOPは別のまま保持した。

新SCF/DF/state/reference生成、signal、sampling、gate matrix生成、回路作用、transpile、shot/費用会計、fit、rankingは0。既存QPYの復元は2回だが、NPZ・statevectorの復元は0。

## 2. 台帳の式と数値条件

[静的調査で確認したdeterminantの式](track_a_h6_m5_phase_ledger_static_20261019.md)を使う。今回の13 qubitのgate集合ではH/X/SX/CXの全系determinantはいずれも1、RZ/RZZ/XXPlusYYも1である。terminal measurementを除くunitary部分のdeterminant比較に必要なscalarは、保存global phaseとwrapperのP角度だけになる。

\[
 B=\gamma_{\rm compiled}-\gamma_{\rm wrapper}
       -\frac12\sum_{P\;\rm in\;wrapper}\theta_P,
 \quad \delta_{\det}=B\pmod{2\pi/8192},\quad
 \delta_{\det}\in[-\pi/8192,\pi/8192).
\]

保存binary64値をbitどおりの有理数へ移し、P角度は`Fraction`で正確に総和した。binary64による総和、隣接Pの取消、角度の再最適化は行わない。moduloには数学的πを使う。既存mpmath 1.3.0/Python backendを封印し、独立した80桁・120桁のcontextで同じ台帳を評価した。旧replayの通常・拡張common phaseも、保存binary64値を正確に有理数へ移して比較した。

内部guardはbranch整数一致、2精度の差≤1e-60 rad、branch境界からの距離>1e-60 rad。これらは**内部数値安定性の確認だけ**で、物理誤差の許容基準・厳密証明・compiler受入閾値ではない。物理的一致閾値と原因分類はどちらも`null`。元raw差1e-9未満という基準は変更していない。

## 3. 終了と保存結果

16:05:37 JSTに起動。外部wait4のwallは38.794707868248224秒、native exit0/reap、2 workerともexit0/reap、signal0。SOURCE・入力・runtime・資源の起動前22項目を通過した。実効空きRAM・idle CPU・diskから2つの独立抽出jobを同時実行できると確認し、2 workerを使用した。監視には一時的なhost PSIによるkillを設けていない。[fresh gate](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/fresh_gate.json)、[終了記録](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/completion_witness.json)、[外部exit](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/external_exit.json)。

wrapperの233,465個のPを正確に総和し、compiledにPが0であることを確認した。抽出したoperation順序・counts・global phaseのbit表現は、前回の保存構造metadataに一致した。[wrapper抽出](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/extract_WRAPPER.json)、[compiled抽出](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/extract_COMPILED.json)。

|台帳の保存値|結果|
|---|---|
|wrapper P角度総和（正確な有理数）|−518346498055919 / 140737488355328|
|B（正確な有理数）|884279719839943 / 562949953421312|
|固定branch整数（80桁・120桁とも）|2048|
|120桁のδ_det rad（表示を丸めた値）|1.4857234829599567e-9|
|80桁と120桁の差 rad|5.358348126710585e-82|
|branch境界までの距離 rad（120桁、表示丸め）|0.00038349371124792735|

|比較（runner出力）|保存common phase rad|δ_det − 保存common phase rad|
|---|---:|---:|
|通常replay|1.5028193689500676e-9|−1.7095885990110854e-11|
|拡張replay|1.5026891953807317e-9|−1.6965712420775007e-11|

[完全なscalar出力](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/scalar_ledger_result.json)には80/120桁の文字列・exact有理数・保存referenceのbit表現・連続値の差を保存した。保存値へ近づくようbranchを選び直していない。確認後に精度や式を変更した追加試行も0。

## 4. この結果から言えること

保存gateパラメータとglobal phaseの数学的なdeterminant帳簿にも、約1.486e-9 radのscalar位相差がある。このため、元の約1.5e-9の差を、今回の状態ベクトル再生時の丸めだけで説明する状況ではない。拡張replayでも差が残った前回の証拠と整合する。

一方、今回のδ_detと保存common phaseには約1.70e-11 radの差がある。**determinant由来の位相と、特定の初期状態に対する共通位相は別の量**であり、同一視しない。determinantは2π/8192ごとのaliasを持ち、unitary全体の作用・全状態で一様なglobal phase・発生compiler passを決定できない。内部精度の安定性も、厳密な物理誤差certificateではない。特定の加算関数・再合成passを原因として確定したわけではない。

元raw差は通常1.504043707445665e-9、拡張1.5039128868467355e-9で、どちらも旧基準FAILのまま。今回の値を回路へ位相補正したり、raw基準をphase-alignedへ置換したり、未受理費用を受理したりしていない。旧58受理軸・未受理cosine・未投入sine・AD03主比較採用保留を維持する。

## 5. 人工検証・監査・保全

[人工13件](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/artificial_validation.json)が固定Python3.12.3でPASS（1.061秒）。正確な有理数、固定branch境界、内部精度不一致STOP、scope拡張拒否、未承認CLI、独立supervisor、人工13q QPYの2 worker実動作と回収、metadata改変時のscalar未投入、一回限りの拒否を確認した。初回はPython3.12のbuilt-in `sum`が補償加算を行うことを人工比較側が想定せず1件失敗したため、naive参照を明示的な`+=`へ修正した。実数値coreの変更や実回路の再試行ではなく、初回logもprivateに保持した。作者自身の検証で、独立レビューではない。AI subagent0。

実configのDRAFTはexecute/superviseとも、実入力復元前にexit2で拒否した。metadata改変12種も拒否。[起動前22項目](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/preflight_checks.json)。

[終了監査35項目](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/final_audit.json)PASS。新SOURCE・入力・runtime1,932封印file、正確な3stageのledger、新grantの消費、結果/抽出SHA、2 workerとlauncherのnative exit/reapを照合した。監査ではQPY/NPZ復元やscalar再評価をしていない。

[保全2,324file](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_run01/2026-10-19/preservation.json)は全て不変。元科学seal1,831fileと、旧SOURCE/raw/結果/grant/ledger/marker/STOP、以前のDRAFT/INVALIDを保持した。変更拒否対象は使用する証拠に限定し、無関係なworktreeのHEAD/statusを凍結していない。他worktreeや共有runtimeは書き換えない。

全QPY/NPZ、有効grant/config、実path、資源観測logはprivateに保持する。今回のbundleは既計算scalar・metadata・receipt・hashだけで、実行可能な有効grantを含まない。既存科学契約、V1/V2、H5/H7、Codex A/Bの作業にも変更を適用していない。

## 6. 次の担当

**ユーザー**：今回の位相差の証拠を確認し、compiler位相処理の追加調査・修正計画へ進むか判断する。

**Codex（次の指示がある場合）**：保存差を消す事後補正とは分けて、必要な修正・検証範囲を具体化する。旧STOP解除・再compile・費用会計・Core/主比較採用は今回の承認に含まれない。

一回照合の終了後mandatory STOP。後続計算の自動起動、公開pushは0。
