# H4強baseline：spin対応XDF基底と相対Gaussian融合の技術準備

2026-10-26 JST。**技術準備を完了。新25＋既存76＝101 unique人工・回帰tests PASS、実H4の新規数値decode／build／compile／Operator／Statevector／再解析／資源計算は0。** 基底選択と融合の改善量、約12%の条件付き総RZ差が残るか、独立新規性は未判定。科学実行と重要科学レビューは今回実施していない。公開照合後mandatory STOP。

- 実装SOURCE：`ea5fff03be761056fdf826b4b6eebd442bfc2d89`。起点は前段公開 `686497c2970023deaa2201bc11174a6207cee3d3`。
- Branch：`codex/research-c-c1-h4-strong-baseline-preparation-20261026`、独立worktree `/tmp/prt-c1-h4-strong-baseline-preparation-20261026`。
- [opt-in adapter／限定runner](../../src/trottertracks/c1_h4_strong_baseline.py)／[CLI](../../scripts/run_research_c_c1_h4_strong_baseline.py)／[人工tests](../../tests/management/test_research_c_c1_h4_strong_baseline.py)。新規3ファイルだけで旧NativeBuilder・PrecisionBuilder・既存defaultは変更しない。
- [未承認227-call契約案](research_c_c1_h4_strong_baseline_science_contract_draft_20261026.json)／[SOURCE・検証・テスト原本](../../artifacts/research_c_c1_h4_strong_baseline/2026-10-26/technical/)。
- SOURCE manifest SHA256：`728ff6eae2684d1934c2af2aa279f3062db1573c5a0d1bd8f8f3b5ab7cee2510`。
- 契約SHA256：`e5ca93221b78a846a3c8c7fb56f2caebf6e30a748fd5f7c366b7efc04eb21f7c`。

## 固定資料と今回の問い

GitHubで固定4資料を取得し、同commitのローカル版とSOURCEを確認した。

| 資料 | 科学SOURCE／結果公開 | 今回の扱い |
|---|---|---|
| [共通precision H4 native/full512](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f4de9f62c2ff1cbaaa75b6cdaa07217d00dfd957/docs/research/research_c_c1_h4_native_fair_execution_20261026.md) | `862a97d4…`／`f4de9f62…` | B0、元CAR/PF reference、shell、元local PASS、metricsの再利用元 |
| [H4共通CA・条件付き総資源](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3a8356a7843d3cb46bb8069955640381b490ac4c/docs/research/research_c_c1_h4_ca_resources_execution_20261026.md) | `18045333…`／`3a8356a7…` | 約12%は保存結論の引用。今回shots・総費用を再集計しない |
| [H2/H4/H6統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a4cda44a45675c2518b2252a98e78eeeb320942/docs/research/research_c_c1_h2_h4_h6_integration_update_20261026.md) | `9a4cda44…` | rank・codec・baseline・精度資格の条件差を維持 |
| [C4終了・引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ec27625839564dc22db2728923120f497b7cadee/docs/research/research_c_c4_closeout_handoff_20261024.md) | `ec276258…` | 既知の強いbaselineを両armへ公平に与える。再開はしない |

対象は保存直鎖H4、隣接1.00 Å、STO-3G、Nα=Nβ=2／sector36、system8 mode／full256、ancilla-last controlled full512。通常number CDF8、保存OpenFermion1.6.1 eigh元順prefix8 XDF、元g補正κ・核scalar一回・signed λ/G・fragment順を保持する。全frame ordered symmetric S2、δ=.1、m0..12、s1/4、PR非適用という既存背景のうち、**初回の新実行対象はs1のh=.1 kernelだけ**とする。

CA=.00159 Ha、DF reserveCA/8=.00019875 Ha、own-DF ground oracle／ideal IID Born／状態準備除外、元RPE条件・serial modelは継承する。ただし今回の契約はCA・RPE shots・13round総費用を評価しない。CDF/XDFの保存DF近似誤差が異なる点も維持する。2026-10-26 GPTレビューの指摘は調査動機であり、性能差の原因が確定したとは扱わない。

## 現行XDFと合法な基底自由度

`c1_stage2_native.prepare_xdf` はκと各spin-orbital Gを `diag_hermitian(..., assume_hermitian=True)` で再対角化し、|η|降順のη/Uから既存square primitiveを作る。`diag_hermitian` は `np.linalg.eigh` とcolumnの並べ替えであり、spin構造を指定しない。CDFのframeは `c1_general_k_adapter` が保存spatial Uをinterleaved spinへ展開する。CDFのone-body κは現行でもspin-orbital eighである。

G=A⊗I₂なら、V diag(η)V†=AからU=V⊗I₂、eigenvalues=repeat(η,2)が合法である。`spin_adapted_eigh` はこの構造を**np.array_equalで完全一致**確認し、spatial部分へ同じ既存eighを適用する。非zeroのoff-spin要素やα/β block差を、極小でも捨てない。blocked-spin順をinterleaved順と推定して変換せず、不適格なら明示STOPする。H4への適用可否は将来の実入力gateで再確認する。

これは保存されていない歴史的XDF固有ベクトルを復元する主張ではなく、同じGを再構成する一つのspatial基底を構成する処理である。各Uのunitarity・G再構成・Hermitian性を最大成分1e-10／rtol0で検査する。判定はbinary64の数値pilotで、厳密丸めenclosureではない。

固有ベクトルのcolumn位相・符号、固有値と一緒のcolumn順序変更、厳密に縮退した固有空間内のunitaryには自由度がある。非縮退columnの混合は一般にGを変える。近い固有値をcluster化する最適化、費用を見た符号・順序探索は実装しない。人工テストでは位相／順序／厳密縮退の合法性と、不適格な非縮退回転の拒否を確認した。一般CDFのJへ、XDFの縮退自由度をそのまま流用することもない。

`prepare_xdf_basis(..., basis_policy=LEGACY)` は旧関数へそのまま委譲する。opt-in `exact_spin_lift_spatial_eigh_v1` はbase＋8 frameへ適用し、columnに対応するηから同じ既存 `one_body_diagonal_primitives`／`df_squared_diagonal_primitives` を使う。λ・G・κ・核scalar・rank・fragment順は変更しない。λの符号や1/2係数を新定義しない。基底を変える際の対角重みの順序も対応させる。

spin-orbital eighとspatial spin-liftは同じGを表せても、縮退空間内の選択・列順・位相が異なり得る。既存Givens合成は渡された正方Uを分解するため、gate列も異なり得る。ただし、構造化した基底がcompiled RZ/CX/depthを削減するかは未測定である。

## 相対Gaussian融合と位相

旧builderのchronological列は Bᵢ⁻¹→controlled Dᵢ→Bᵢ。その直後のBⱼ⁻¹との境界の行列作用は **Bⱼ†Bᵢ** であり、相対一粒子行列は **R=Uⱼ†Uᵢ**。単純なUᵢUⱼ†や、転置だけの積には置換しない。

固定precision emitterはRZ／XXPlusYYを用い、単一粒子transferを真空位相に対して検査する。したがってB(U)=exp(iφ(U))Γ(U)と書ける。φは `circuit.global_phase − Σ RZ_angle/2`、XXPlusYYの真空作用は1である。境界をB(R)へ置き換える際は、**φ(Uᵢ)−φ(Uⱼ)−φ(R)** を宣言global phaseへ加える。位相branchをまたぐ場合の符号をこれで保持する。Operatorを見たphase fittingやglobal-phase alignmentではない。未知gateを真空位相0とみなさず明示FAILにする。

`StrongBuilder(relative_fusion=True)` は既存 `runs(..., optimized=True)` と `combined_diagonal` をそのまま使い、先頭B⁻¹、各controlled diagonal、合法な隣接境界の相対B、末尾Bをemitする。同一basis共有、fsum角度結合、中央I/Z/ZZ相対位相、核scalarのancilla p一回を維持する。非可換なfragmentを並べ替えず、control0枝も含む全空間で検査する。相対Bは既存 `nature_jw_exact_zero_lapack_atan2_v1` へ渡し、新しいGaussian compiler／Givens分解器は作らない。

数学的rotation cacheは版・precision・basis policy・元SOURCE/input・方向付きUᵢ/Uⱼをkeyにする。各routeのbuilder/cacheは独立。cache hitでも、fusion記録のfragment IDは現在要求されたものを残す。compile cacheから旧baseline費用を新routeへ転用せず、P0は新3 kernelを個別にcompileする。`relative_fusion=False` は旧PrecisionBuilderのemitへ委譲する。

別branchの `df_partial_s2_repeated.py`、D2 `rte_development/boundary.py`、Gaussian singleton設計・C4 adapterも確認した。前二者の該当処理は同一basis境界やliteral inverse cancellationであり、このfull Gaussian相対積をprecision/真空位相付きで直接emitするadapterではない。singletonのrectangular部分空間法も今回のfull density-squareでは適合条件が異なる。旧研究を再開せず、適合する既存Gaussian、中央emitter、actual adjoint、reference、P0、metrics、journalだけを再利用した。

## 最小比較集合

| 比較route | 基底 | 相対融合 | 比較先・識別する効果 |
|---|---|---|---|
| B0 CDF8／XDF8 | 保存方式 | なし | 保存s1のmetrics・元reference・PASSを再利用。再buildなし |
| B1 XDF8 | exact spin対応 | なし | B0 XDFとの差で基底構築効果。保存CDFとの差も記録 |
| B2 CDF8 | B0 CDFのまま | あり | B0 CDFとの差 |
| B2 XDF8 | B0 XDFのまま | あり | B0 XDFとの差、B2 CDFとの公平比較 |
| B3 XDF8 | spin対応 | あり | **初回には含めない**。相互作用は未識別のまま |

新3route・保存2baselineで基底単独と融合単独の効果を分離できる。B3を省いたので両改善の相互作用や「強化手段をすべて併用した後の最終優位」は初回では結論できない。結果に応じてB3の情報価値を研究C／ユーザーが判断する。s1単一kernelの順位をs4・13round・条件付き総資源へ外挿しない。

両armに同じP0 `qiskit_P0_mpfr160_declared_phase_v1`、Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false、既存layout等の未指定条件を適用する。RZ/CX count、RZ/CX/total depth、sizeを保存し、独自のweighted scoreを作らない。Gaussian生成・compileのwallは既存call journal、全体CPU/RSSはresourcesへ保存する。全回路一体compileやlong-round compiler最適化の費用とは区別する。

## 独立referenceと限定科学実行案

元P-H4 SHA256は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、CDF exportは`c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`。旧契約、旧SOURCE、input、κ/scalar/rank/frame、DF/PF/ground/aliasの来歴を旧byte／identity処理で確認する。保存科学配列は今回decodeしていない。

将来は元Gから独立raw CAR9因子を構成し、B1の9 complete controlled local因子をτ=.05でfull512検査する。B2のlocal fragmentはB0と同一で、保存local資格を再利用する。kernel targetは**元の独立CARから作成済みの保存CDF／XDF s1 full256 reference**を読み、変換後基底からreferenceを自己生成しない。

続いてnew kernel3のraw/P0をfull512全入力column・全出力成分で `diag(I,U)` と比較する。最大成分allowance1e-10／rtol0、unitarityも既存segmentで確認。6 actual adjointを回路から作り、dagger／reference／復帰を別々に比較する。保存raw/compiled head・X/Y shellとown-DF groundを使い、m0のcomplex X/Yは2e-10で確認する。shellに変更がないため、新しいbuild/compileはしない。位相fit、sector-only、有限probe代用、事後緩和は禁止する。

新しいfull512 Frobenius残差は数値pilotとして保存するが、13roundのcoherent反復伝播・厳密合成上界・最終CA・shots・総費用は初回契約外。旧回路の合成PASSを新回路へ転用せず、未知値はnullとする。

| 科学call分類 | 最大logical units |
|---|---:|
| 旧loader27＋identity/baseline2＋models3＋spin証拠9＋元reference2＋shell行列6 | 49 |
| raw CAR context1＋元XDF generator9 | 10 |
| absolute Gaussian要求27＋相対Gaussian cache miss最大32 | 59 |
| B1 local reference/build/Operator/gate各9 | 36 |
| kernel build3＋fusion順序gate3 | 6 |
| compile3＋kernel Operator6＋segment3 | 12 |
| actual adjoint build/Operator/gate各6 | 18 |
| own-ground signal reference2＋prepare6＋X/Y axis12＋complex gate9 | 29 |
| metrics3＋事前登録比較5 | 8 |
| **上限** | **227** |

9blockのS2は最大17run／16境界なので、B2両armで相対合成は最大32。完全一致sharing／cache hitにより実績が上限より少なくなる場合はそのまま記録する。27は3route×9blockのabsolute basis要求で、cache hitも要求として数える。内部primitive操作を追加の独立science unitとはしない。runは新規QPY21、kernel Operator6、spin/fusion記録、local gate、candidate、metrics、比較、SOURCE／journal／STOPを保存する。

SOURCE32件・saved input45件と原packet・依存9版のpassive byte preflightがPASS。新科学実行のfresh outputは`artifacts/research_c_c1_h4_strong_baseline/<execution-date>/run01`、retry/resume0。最初のcorrectness FAILで依存callを停止し、開始済み独立jobを安全回収、partial evidenceとSTOPを保存する。不適格spin構造の黙ったlegacy fallbackや、結果に応じたB3追加は行わない。

## 人工・回帰検証、資源、保全

新25 testsは複素／実spin構造、位相／符号／順序／縮退、非適格構造、相対行列の向き・共役、正負時間、同一basis sharing、scalar、control0、full32のforward/adjoint/X/Y/P0、route/cache、未知誤差、SOURCE/input mismatch、fresh outputとpartial STOPを確認した。人工end-to-endは3route・5比較・QPY21・Operator6と全call種類を通過した。実H4の数値arrays／QuantumCircuitは使用していない。

real-Qiskit群74 PASS（新25＋旧fair20＋Gaussian precision24＋compiler phase5、37.86秒）、旧Stage2 mockは別process27 PASS（.66秒）、**101 unique PASS**。初回混在batchは100 PASS／1 FAILで、旧mockがfake RZZGate下での初回ops importを前提とする既知の失敗だった。これは[旧準備報告](research_c_c1_h4_native_fair_preparation_20261026.md)に記録済みと同じ条件であり、旧SOURCEを変更せずprocess分離した。混在ログも保全した。

開発中に、同じ相対rotationのcache hitが初回fragment IDを記録する不具合を修正した。負対照は位相branchを明示的にまたぐfixtureへ直し、独立B1がB2 FAIL前に完了し得るSTOP assertionも修正した。最終の人工／回帰PASSと実分子の科学的PASSは区別する。

passive snapshotはaffinity32 CPU、MemAvailable約52.6 GiB、CPU/memory pressure avg10=0。独立3routeのfull512行列workを既存resource_planで並列化し、BLASの重複並列をjob内で協調する。P0／actual inverse生成は既存mutexを維持し、raw basis inverseはP0開始前に完了する。実行時に空き資源と実quotaを再確認する。CPU/RAM/worker/wallの人工capなし、速度向上倍率は未計測。今回shared venvはread-only、cache/tempはtask内で、共有設定・package・cache・他ユーザーjobは変更していない。

旧H2/H4/H6のSOURCE、raw、grant、ledger、marker、STOP、C4・singletonの終了状態を変更しない。既存科学ファイルの変更は0、新規scientific callは0。元workspaceも変更していない。通常の新規承認証明書・承認原文保存・grantは作成しない。

## 研究C／統括へ返す判断と次の担当

次のs1限定実行で、基底構築単独と両armの相対融合単独が、保存native kernel費用差をどこまで説明するかが分かる。基底改善で差が消えれば旧basis選択依存、融合で消えれば既知Gaussian合成依存という原因候補を記録する。差が残っても独立アルゴリズム新規性や同CA総資源優位は自動認定しない。情報不足ならB3の最小追加検証またはC1を区切る選択肢を返し、rank探索・compiler総当たりへ進まない。

**ユーザー／研究C**が3route・s1限定227-call案の採否と科学実行を別途明示許可する。**Codex**は許可後だけ公開SOURCE／契約／入力／環境を照合してfresh run01を一度実行する。現時点の技術blockerはなく、実H4の構造適合性・full512 correctness・実際の資源改善は未検証である。

**公開照合後mandatory STOP。** s4、B3、13round合成、CA/shots/総費用、H6 CDF13追加fit、XDF9 full512、RC-DF/SCDF、C4／singleton再開、重要科学レビューへ自動進行しない。
