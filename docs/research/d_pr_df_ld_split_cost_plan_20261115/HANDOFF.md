# 統合担当・A/B/Cへ：PR-DFのL_D別費用比較の初回計画

**推奨はLD1保存単価＋LD2三点full-wrapper価格pilotを用いる二分割の有限比較。** 現在はLD1/LD2各V0のQ3適合と信号を再利用できるが、同条件のLD2単価がなく費用順位は未判定。LD1を代表又は最良と固定しない。厳密E[C]/CI/全support下界を探索用モデル参照規則の採用条件にしない。

[REPORT](REPORT.md)／[L_D=0..10・保存候補・52軸正本](CANDIDATES.json)／[CSV](CANDIDATES.csv)／[一つの有限pilot](NEXT_FINITE_PLAN.json)／[静的照合](CHECKS.json)。作業は既存LD1 D2 branch、起点fb50f41e。原成果・LD2専用branchは変更しない。

## 候補の意味と現在地

L_Dは元保存順DF fragmentの決定論prefix本数。one-body correctionは別のhead blockで本数に含まず、tailはindex L_D..9。元順・scalar c+s_R総T一回を保持する。default重みsortを使わずgeneration_partition→explicit_ordered_partitionの実経路を参照する。

- LD1：a4dda8f7…、LD2：1fa33556…、いずれもS2/.1/K2/元κ。A a9abf8deの13round schedule、A71424761の自身の信号、B a82b0f0eの自身のV0 Q3適合を保存再利用。
- LD1の26 C_refは85dab58dの分子/分母を**そのままcopy**。新予測0。m0/m3/m6較正とm4独立holdout、P1/replay参照を保持。モデル参照であり正式E_I[C]ではない。
- LD2は現候補のC/S/E[S]/保存モデルが全26軸で未同定。m12旧T2はsampling前STOP、旧134088ad…原receiptはq4/R4で現m3と不一致、Pauli LD2も別方式。
- LD3/5/7/9：保存λ/scalar descriptorと旧signal候補があり、現在のschemaで構成可能だが現configuration/全13round/Q3/価格は未評価。旧signalはT=.8/1.2/1.6/2のgrid。LD3/5の旧compiled価格も異なる条件/旧loweringで参考のみ。
- LD4/6/8：prefix APIはあるが保存descriptorがなく、候補入力のλ/scalar/provenance整備から必要。
- LD0：one-body headが残りFull-H pure RTEとは異なる。現signal contextが拒否する境界。LD10：空tail/正λguardの境界。Full PF endpointを新PRのλ=0候補に読み替えない。

## 共通価格規則

`C_ref(L,m,a)=B_L(m,a)+θ0[L,a]+θq[L,a] q+θE[L,a] qR`。共通quantityはP0・測定付きfull controlled Hadamard wrapperのcompiled RZ/shot、新ZZ・Qiskit1.3/rz,sx,x,cx/opt1/seed17・symmetric-directional、scalar総T一回。

LD1はB_1=保存E_I[S]と凍結残差を含んだ既存26値を再利用。LD2はB_2=0という**モデルoffset**を選び、LD2自身のfull Cへfitする。headが0円/E[S]=0という主張ではない。prefixごとのhead・basis・tail分布・角度/phase・境界相殺を係数と全wrapper観測で反映し、同一係数を他Lへ移さない。qRは厳密なevent加法則ではない。

元Q sampler有限cohortを、Iを目標とする探索モデルの根拠として明示して使う。I/Q一致、CIや全path誤差を認定しない一方、未認定を初回参照比較の停止理由にしない。モデル族/offsetの根拠の非対称性、長round外挿を結果に残す。m4を再fitせず、最終C_refへの残差・Sの二重加算、normalization/標本数の再乗算を避ける。

## 最初の一作業：後続の段階的有限pilot

| 段階 | 対象 | 上限 |
|---|---|---:|
| LD2較正 | m0 ce7ef02e… q1/R1、m3 334b1fd3… q8/R5、m6 fe234d98… q64/R40、各fresh4 path×X/Y | 新path12、compile24（各cell8） |
| LD2参照凍結 | full Cの6 cell-axis平均、(1,q,qR)を軸別fit、全13roundへ適用。LD1単価は保存再利用 | fit2、新単価26、LD1 fit/単価再計算0 |
| 二分割の独自配分Q3・比較 | 各LDの自身の基準参照総額の3/4予算でA_SHAPEとB_PRICEを1本ずつ。自身のsignal/Q3、保存V0適合は2件再利用 | 新vector/Q3各4、最大6行menu、報告用新総額最大5 |

追加信号0、全L/allround compileなし。n4は既存LD1有限pilotの規模を再利用するもので、精度・信頼水準から導いたnではない。LD2独立価格holdoutは今回は提案枠0。三点fit後のm7..m12はq64/R40を超える外挿であり、同prefix/同scheduleでCを(1,q,qR)で近似するという明示仮定を置く。モデルPASS/FAILや仮の許容差を付けない。

取得入口はe8273676 `ld12_price_requests.request_for → NativeRuntime.context/sample → Adapter.acquire`を優先し、旧C1のprefix1固定contextをLD2として呼ばない。corrected c+s_R・head/tail/λ/SOURCEを検査。元P2 fresh/m3のidentity/lowering/fingerprint検査を保ち、元経路で未実施の数値作用gate/full-wrapper statevectorのPASSを追加しない。科学前に実際のSOURCE・入力・cohort/seed・出力を固定し、実入口を人工/fake compilerで確認する。 adapterがaxes/observerを渡すfast backendは26d4f23e/e8273676の対応版を固定する。現在LD1 checkoutの旧fastはこのkwargsを持たず、そのままの組合せでは動かない。

配分はB06f4210eのbreakpoint式、N_i=max(1,min(V0_i,floor(λw_i)))、w=V0（A）又はV0/C_ref（B）。候補ごとの予算内の最大点を選ぶ。LD間/X/Yで同じNを強制しない。元generatorは価格を整数限定するため、正のFractionの価格処理へ小変更・人工確認が必要。整数RZへ丸めない。fitではexact _solve/predictを再利用し、cellごとのpoint lawを保存したprefix2 schedule-family wrapperを定義する。汎用fitのlaw完全一致guardへ異なるpoint lawを同一と偽って通さない。4新Q3では元Z1/T1・proxy/全事象loss/exact閾値・自身のlaw/signalを使い、新Nの閉包は個別確認する。旧Nの閉包PASSやMSE単調性は流用しない。

## 後続の選択とロードマップ

各LDの有限menu内最小適合T_ref、選択N、Q3状態、round別N×C_refと較正/外挿roleを比較する。差が未検証長roundに依存するなら、そのLD2価格の有限確認を先に別計画とする。tail負担削減とhead増加の釣合いを次の分割で見る価値があれば、保存descriptorを持つ隣接LD3を次候補とする。二点の順位を単調性や全分割の枝刈りへ一般化しない。

PF参考値はb4296b11のFull PF615,304,502 RZ、B0I p7 435,481,090 RZ（固定有限集合内・条件付き閉包価格）。PR内の必須予算にはしない。今回は新差額/比/ランキング0。旧LD1参照総額・旧PF境界・m9下界UNKNOWN・旧STOPは保全し、下界再証明は待ち依存にしない。ロードマップ本体は統合担当が更新する。

## 重複・実行前差分・STOP

旧Aの二prefix menu/Cの四cell案/D2の26軸調査を再利用。今回新たに必要な差分はLD2の価格参照と候補固有N/Q3であり、新signalやLD1再計算ではない。最新のB PF予算/幅Q3、B2 Pauli配分、A Pauli S0+、C Full-H価格案は別scope。今回と同等のPR-DF横断比較/同名jobは確認範囲で未同定。

次回は計画後の同等receipt・fit/26軸表・同一NのQ3、進行job、出力/cache競合、実際のSOURCE/入力/compiler差分だけを照合。同等結果は出所付きで再利用、不明は保留。旧T2/m12や他担当jobには介入しない。

後続でidentity/SOURCE/scalar/axis不一致、取得・保存失敗、cell欠測・fit rank失敗、非正/非有限価格なら部分保存して新規投入STOP。retry/seed差替え/追加path、観測後のfeature変更・clip・再fit、Q3結果による別vector追加はしない。finite menuの不適合/不確定も保存する。

**今回の新sampling・信号・build/compile・fit・価格予測・shot生成・Q3・総額再計算/再ランキングは全て0。** 計画公開・remote照合でmandatory STOP。後続pilot全体は未採択・未実行で、実行は別指示。
