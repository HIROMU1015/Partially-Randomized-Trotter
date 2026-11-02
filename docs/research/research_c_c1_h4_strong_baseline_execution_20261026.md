# H4強baseline：spin基底・相対Gaussian融合の限定科学実行

2026-10-26 JST。**227/227 logical units COMPLETE、correctness FAIL／inflight／retry／resume0。全9局所因子・新3 kernelのfull512／P0／6 actual adjoint／9 complex X/Y gateがPASSした。** s1単一kernelでは、XDFのspin対応基底だけで保存CDFとの全6 native指標の大小関係が逆転した。両armへの相対融合ではCDFのRZがXDFより2.1668%小さいが、CXは同数、depth3指標はXDFが小さい。B3、13round合成、CA、shots、総費用は未評価。公開照合後mandatory STOP。

- 公開準備／今回の科学実行SOURCE：`880006396326a4d46f4a4030564279e92fadb9cf`、tree `4a016a93ab2f49fd1b8c2afc65ccc6cab6211522`。
- 実装SOURCE：`ea5fff03be761056fdf826b4b6eebd442bfc2d89`。runner・library・tests・契約は変更せず一度実行した。
- Branch：`codex/research-c-c1-h4-strong-baseline-execution-20261026`、独立worktree `/tmp/prt-c1-h4-strong-baseline-execution-20261026`。
- [固定227-call契約](research_c_c1_h4_strong_baseline_science_contract_draft_20261026.json)、scope `C1_H4_STRONG_BASELINE_S1_V1`。契約SHA256 `e5ca93221b78a846a3c8c7fb56f2caebf6e30a748fd5f7c366b7efc04eb21f7c`。
- SOURCE manifest SHA256 `728ff6eae2684d1934c2af2aa279f3062db1573c5a0d1bd8f8f3b5ab7cee2510`。
- [全raw60件・QPY21・SOURCE・journal・STOP](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/run01/)／[機械可読summary](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/execution_summary.json)。
- [全native metrics](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/native_metrics.csv)／[5比較](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/native_comparisons.csv)／[correctness残差](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/correctness.csv)／[分類別call時間](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/call_timing.csv)。
- [preflight](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/execution_preflight.json)／[postflight・保全hash](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/execution_postflight.json)／[技術準備・人工101 tests・数学的規約](research_c_c1_h4_strong_baseline_preparation_20261026.md)。

## 固定条件と再利用

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ2／sector36、8 spin-orbital mode、system full256／ancilla-last controlled full512。保存number CDF8とOpenFermion1.6.1 eigh元順prefix8 XDF、元g補正κ・signed λ/G・核scalar一回・元fragment順を保持した。PFは全frame ordered symmetric S2、今回のnew kernelはs1・h=δ=.1、局所B1はτ=.05。PR splitなし、own-DF ground oracle／状態準備費用除外。CA=.00159 Ha／DF reserveCA/8と13round・s4・RPE条件は背景条件として維持したが、今回の評価範囲へ拡張していない。

Gaussian `nature_jw_exact_zero_lapack_atan2_v1`、P0 phase `qiskit_P0_mpfr160_declared_phase_v1`、Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false、既存layout等の未指定条件を共通適用した。同一basis共有・fsum角度結合、head/kernel/tail別compileというscopeを維持し、noncommuting fragmentの並べ替えはない。

原P-H4 SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、CDF export `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`。B0は科学SOURCE `862a97d4213121749448ca716cde820a44b99772`／公開 `f4de9f62c2ff1cbaaa75b6cdaa07217d00dfd957` の保存native結果・元CAR/PF full256 reference・shell・local PASSを使う。保存DF norm＋guardはCDF7.666215916301914e-5 Ha／XDF9.533901455091894e-5 Haと異なり、今回再計算していない。旧共通CA結果はSOURCE `18045333…`／公開 `3a8356a7…` の別scopeとして保全し、新しい総費用を作っていない。

ユーザーのチャット許可後、fresh run01を一度実行した。歴史的なdraft/unapproved名・metadataを変更せず、承認証明・原文保存・新grantは作成していない。**run01は消費済みでretry/resume不可。**

## Gateとfull controlled位相

| Gate | 結果 |
|---|---|
| SOURCE32・保存input45・原packet・契約・依存9版 | pre/postflight PASS、protected77件不変 |
| 保存DF/PF/ground/alias、κ/scalar/rank/frame、B0 scope/来歴 | PASS。DF/PF/groundの新規計算なし |
| B1 base＋8frameのexact interleaved spin構造 | 9/9 PASS。off-spin要素の省略・近縮退cluster化なし |
| B1のG再構成・basis unitarity | 最大成分差2.220446049250313e-15／1.1102230246251565e-15 |
| B1 complete controlled局所9因子 | 9/9 PASS。最大成分差1.5184608422413791e-15、unitarity差2.886579864025407e-15 |
| ordered S2・fusion順序 | 3/3 PASS。B1境界融合0、B2 CDF/XDFは各16箇所 |
| raw/P0 full512、control0/1・unitarity | 3/3 PASS。最大成分1e-10、rtol0、global phase fitなし |
| actual adjoint | 6/6 PASS。実inverseとdagger／reference／復帰を独立に照合 |
| m0 own-ground raw/P0 complex X/Y | 9/9 comparisons PASS、allowance2e-10 |
| native費用 | correctness PASS後の3 metricsと登録5比較を完了 |

局所referenceは**元Gからの独立raw CAR**、kernel referenceは**元CARから作成済みの保存CDF/XDF full256 target**である。新しいbasisやfusionからtargetを自己生成し直していない。512全入力column・512全出力componentを検査し、sector-onlyや有限probeで代用しなかった。

| 新route | raw/ref最大成分 | P0/raw最大成分 | P0/ref最大成分 | P0/ref Frobenius | P0/ref X/Y誤差 |
|---|---:|---:|---:|---:|---:|
| B1 XDF spin | 5.12090e-15 | 3.26226e-12 | 3.26226e-12 | 5.94926e-11 | 3.55698e-13 |
| B2 CDF relative | 5.78303e-15 | 1.96856e-12 | 1.97295e-12 | 3.11810e-11 | 3.61434e-13 |
| B2 XDF relative | 5.12938e-15 | 6.24635e-12 | 6.24978e-12 | 1.29066e-10 | 3.42033e-13 |

adjointの最大成分差はdagger比較1.45450e-14、reference比較6.25263e-12、復帰3.70939e-13。解析的真空位相補正、controlled対角I項・核scalar、Gaussianの向き・frame順を含む数値pilotのPASSである。P0/ref Frobeniusと最大成分差を混同しない。登録gateは最大成分1e-10で、XDF B2のFrobeniusが1e-10を超えることによる事後FAILや許容差変更はない。

full512 Frobeniusは測定した数値残差として保存した。reference形成・Operator復元・norm評価の厳密丸めenclosureは未資格、`rigorous_enclosure`／`CA_achieved`／`shots`／`total_cost`はnullである。13roundへの合成誤差伝播は実施していない。

## 同一s1 kernelのnative資源

**測定単位はh=.1のcontrolled S2 kernel一個。** B0の2行は保存値を引用し、新しいbuild/compileはしていない。新3行は同じP0とGaussian精度による今回の測定値である。

| route | RZ count | RZ depth | CX count | CX depth | total depth | circuit size |
|---|---:|---:|---:|---:|---:|---:|
| B0 CDF8（保存） | 12,272 | 5,873 | 7,388 | 6,440 | 12,806 | 23,543 |
| B0 XDF8（保存） | 13,944 | 6,005 | 7,756 | 6,468 | 13,113 | 26,740 |
| B1 XDF8 spin | 11,821 | 5,793 | 7,176 | 6,328 | 12,556 | 22,376 |
| B2 CDF8 relative | 11,062 | 5,577 | 7,008 | 6,280 | 12,127 | 21,011 |
| B2 XDF8 relative | 11,307 | 5,564 | 7,008 | 6,220 | 12,122 | 21,706 |

| 登録比較（left/right） | ΔRZ | ΔRZ depth | ΔCX | ΔCX depth | Δtotal depth | Δsize | RZ比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 XDF / B0 XDF | −2,123 | −212 | −580 | −140 | −557 | −4,364 | .8477481354 |
| B2 CDF / B0 CDF | −1,210 | −296 | −380 | −160 | −679 | −2,532 | .9014015645 |
| B2 XDF / B0 XDF | −2,637 | −441 | −748 | −248 | −991 | −5,034 | .8108864028 |
| B0 CDF / B1 XDF | +451 | +80 | +212 | +112 | +250 | +1,167 | 1.038152441 |
| B2 CDF / B2 XDF | −245 | +13 | 0 | +60 | +5 | −695 | .9783320067 |

B1はXDF B0のRZを15.2252%減らし、保存CDF B0より全6指標で小さかった。B2はCDFのRZを9.85984%、XDFを18.9114%減らした。両arm共通融合後のCDF/XDF RZ比は.978332（CDFが2.1668%少ない）で、旧nativeの約12%差から縮小した。一方でCDFのRZ depth／CX depth／total depthはそれぞれ13／60／5大きく、全指標一律の優位ではない。百分率は保存比較ratioの表示用算術であり、新条件での再集計ではない。

これで**今回の単一kernel差にXDF基底構築と既知境界合成が寄与することを観測**した。B1はCDF B0との比較、B2は元basisを保った両arm比較であり、B1とB2を併用したB3の効果は測っていない。旧13roundの約12%条件付き総RZを、この表だけで取消し・再資格付け・新順位へ置き換えない。新しいshots・全round合成・同CA総資源・全分子／全rank／全compilerへの優位・独立新規性は未判定である。

## 科学call・古典負担・資源・保全

| 分類 | 実績／上限 |
|---|---:|
| loader/identity/baseline/models/spin証拠/reference/shell | 49/49 |
| raw CAR context＋元XDF generators | 10/10 |
| absolute Gaussian要求＋相対Gaussian cache miss | 27＋32＝59/59 |
| B1 local reference/build/Operator/gate | 36/36 |
| kernel build＋fusion gate | 6/6 |
| P0 compile＋kernel Operator＋segment | 12/12 |
| actual adjoint build/Operator/gate | 18/18 |
| signal reference/prepare/axis/complex gate | 29/29 |
| metrics＋登録比較 | 8/8 |
| **合計** | **227/227 COMPLETE** |

journal attempt=COMPLETE=227、全36種類の上限と実績が一致、FAIL/inflight0。新fit／optimizer／rank探索／DF-PF誤差再計算／B0 kernel再build-compile／B3／s4／13round合成／CA／shots／総費用／retry-resumeは0。新しいHamiltonian分解や旧one-shot runnerの再起動はない。same SOURCE/environmentの人工101 PASSを再利用し、不要な再試験はしていない。

開始前affinity32 CPU、空きRAM約52.8 GiB、CPU/memory/IO pressure avg10=0を確認。閲覧範囲にcpu.max・memory.maxの数値が現れず、無制限とは断定していない。既存resource_planが独立3候補のworker3、job内BLAS1を選択。P0とactual inverse生成は既存mutex、raw basis inverseはP0前に完了。GPU・全512 SVDは使わず、保存referenceを再利用した。

runner wall81.45065613秒／process CPU260.298666604秒／peak RSS730604 KiB。分類別COMPLETE wallの合計はabsolute Gaussian要求.04811秒、相対Gaussian.03763秒、P0 compile6.43458秒、kernel Operator100.44580秒、adjoint Operator100.55743秒。これは**同時実行を含むcall時間の総和**で、runner wallやCPUと同じ量ではない。実装間の速度向上倍率は主張しない。

Python3.11.1と固定9 packageは既存venvをread-onlyで使用した。cache/tmpはtask内、shared venv/package/cache/settingsや他ユーザーjobの変更なし。CPU/RAM/worker/wallの人工capを追加していない。

raw60 files／16,737,766 bytes、QPY21、kernel Operator6を保存。SOURCE32＋saved input45の77件、原packet、契約／manifestの実行前後hashが一致した。旧H2/H4/H6、約27%／約12%比較、XDF9、C4／singleton終了状態、grant／ledger／marker／STOPは保全した。新CSV・summaryはこのrunのJSONとjournalの転記・整形だけで、回路／norm／solver／総費用を追加呼出ししていない。公開ローカル実行証拠であり、外部再現・CI実行とは区別する。

## 次の担当・未資格事項・mandatory STOP

今回の単一kernel結果は、CDFの資源差を元XDF basisや既知fusionと切り分けずに一般化することを支持しない。残るB2のRZ差245だけで独立新規性を認定せず、複数指標とscopeを研究Cへ返す。今回の科学的事実は登録5比較と数値correctnessに限定し、重要な科学レビュー・正式な継続／終了判断は実施していない。

**ユーザー／研究C**がB3（spin基底＋相対融合）の追加情報価値、またはC1を区切る選択肢を判断する。B3を選ぶ場合の次の**Codex**作業案は、XDFの単一s1新routeだけを対象とする最小の技術接続・固定契約準備である。新しい科学実行や重要レビューにはそれぞれ別の明示許可が必要。B3、s4、全round合成、CA／shots／総費用、追加fitや別研究を自動開始しない。

今回scopeの未完了科学call・技術blockerはない。**commit・push・remote SHA／SOURCE／入力／成果物照合後mandatory STOP。**
