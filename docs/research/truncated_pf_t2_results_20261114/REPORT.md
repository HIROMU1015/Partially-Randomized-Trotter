# Truncated PF T2：p=7,8,9のH_eff有限探索（42 cell）

2026-11-14 JST。Claude Code（主agentのみ。独立レビューではない）。

承認されたT2の主探索（p=7,8,9×S2/Y4/M5の幅梯子、42 cell）と、保存B0I signalとの照合を、それぞれ一回だけ実行した。

**結果の概要**
- 42 cellすべてで、数値gate・分枝・状態対応の検査を通過した。UNKNOWNは0件。
- 保存B0I signalとの13round照合も通過した（最大差2.7e−12、基準1e−9）。

**実施しなかったこと**：T2b、価格、compile、全roundの新signal、risk、N、MSE、G。

**主な成果物**
- 正本：[run01](run01/)
- 一覧：[CELLS.csv](CELLS.csv)、[CONFIGURATIONS.csv](CONFIGURATIONS.csv)、[T3_CANDIDATES.json](T3_CANDIDATES.json)
- 集計script：[SUMMARIZE.py](SUMMARIZE.py)。保存結果だけを読み、H4の新しい数値計算はしない。

## 実行条件・SOURCE

| 項目 | 内容 |
|---|---|
| 固定条件 | 保存H4 DF10（tol-only 1e−8）、weighted-Hermitian、sector(2,2)/36、元H/state/snapshot/fragment順、h=.1、M12、P0。targetは元のfull DF-H。B0I scalar（元定数＋c_tail）は1 stepに一回含め、direct経路では総Tで一回 |
| 実行caller | `cab3a9acb55b54c3d3c94bedb25bcbd53d50af72:scripts/run_h4_truncated_t2.py`（実行前にcommitして固定）。人工テスト4件とmetadataテスト1件がPASS（[CALLER_TESTS.txt](CALLER_TESTS.txt)）。T1の6件も引き続きPASS |
| 再利用した実装（無変更） | H_eff評価器：A `878000d8d`。PF kernel：`dd24dd329`（C numerics経由、C numerics `5cc7c07e`＝`86f37d25`）。元Hの固有系cache：A `69ee0ff8`（新しいeighは0件）。保持Hの固有系とδE_trunc：T1 `5e6f9fd2`（再計算せず、残差・Gramを照合）。数値policy：A `9af1ac8e` |
| 分枝の規則 | Aのfollow-up PLAN（`d48d8732e`）のcellごとの規則をそのまま使用。M5 16hだけENERGY_WINDOW `[−2.5,1.0]`、それ以外はPRINCIPAL。結果を見てwindowを変えていない |
| schedule | 既存のB/D/C generatorで再生成し、PLAN_METADATA（`c3ae31f3`）の27 configuration IDと全physical IDに一致した。early roundの実幅は別cellとして扱った |
| 処理件数 | context load 1。PF step・評価器はそれぞれ42件。direct meanとspectral meanはそれぞれ84件（初使用roundとT=409.6）。保存signalとの比較13件 |
| 実行 | 所要8.0秒。retryなし。実行後も入力3ファイルのSHA256がINDEXと一致。Python 3.12.3、NumPy 1.26.4、SciPy 1.14.1 |

## 1. 保存B0I signalとの照合（必須gate）

p7/S2/hのcellのH_effから、13roundのsignalをspectral再構成し、`367dc98d`の保存raw（direct経路）と比べた。
- 全roundで差は2.7e−12以下（基準1e−9）。gate PASS。
- 13roundのphysical IDは保存recordと一致した。
- 詳細は[SAVED_B0I_CROSS_CHECK.json](run01/SAVED_B0I_CROSS_CHECK.json)。

## 2. 切断・PF・合計の固有値シフト

`δE_total=δE_trunc+δE_PF|ret`。δE_truncはT1の値（p7 +8.336e−6、p8 −2.753e−6、p9 −2.946e−8 Ha）。名目幅のcellの値を示す（Ha、丸め表示）。

| 公式・幅 | Full PF（p10、A） | p7 PF\|ret | p7 合計 | p8 合計 | p9 合計 | 判定（全p） |
|---|---:|---:|---:|---:|---:|---|
| S2 h | +1.337e−4 | +1.337e−4 | +1.420e−4 | +1.310e−4 | +1.337e−4 | WITHIN_L2_ROBUST |
| S2 2h | +5.354e−4 | +5.354e−4 | +5.437e−4 | +5.326e−4 | +5.354e−4 | WITHIN_L2_ROBUST |
| S2 16h/7 | +6.996e−4 | +6.996e−4 | +7.079e−4 | +6.968e−4 | +6.996e−4 | WITHIN_L2_ROBUST |
| S2 8h/3 | +9.529e−4 | +9.529e−4 | +9.612e−4 | +9.501e−4 | +9.529e−4 | EXCEEDS（ε_trot超） |
| Y4 4h | −1.906e−4 | −1.906e−4 | −1.823e−4 | −1.933e−4 | −1.906e−4 | WITHIN_L2_ROBUST |
| Y4 16h/3 | −5.907e−4 | −5.907e−4 | −5.823e−4 | −5.934e−4 | −5.908e−4 | WITHIN_L2_ROBUST |
| Y4 32h/5 | −1.201e−3 | −1.201e−3 | −1.192e−3 | −1.203e−3 | −1.201e−3 | EXCEEDS |
| M5 8h | −7.29e−7 | −7.29e−7 | +7.61e−6 | −3.48e−6 | −7.58e−7 | WITHIN_L2_ROBUST |
| M5 16h | −1.049e−5 | −1.049e−5 | −2.16e−6 | −1.32e−5 | −1.052e−5 | WITHIN_L2_ROBUST |

early cell（S2 h/2h、Y4 h/2h/4h、M5 h/2h/4h/8h）もすべてWITHIN_L2_ROBUST。全42行は[CELLS.csv](CELLS.csv)にある。

**観察**
1. **切断してもPF誤差はほとんど変わらない。** `δE_PF|ret`とFull PFのδEとの差は、p7で最大1.05e−7、p8で7.4e−8、p9で5e−11 Ha（相対で最大1e−4程度）。合計シフトは「Full PFのシフト＋T1の切断シフト」でほぼ決まる。
2. **ε_trotを満たす幅の上限はFull PFと同じ。** S2は16h/7まで（8h/3は超過）、Y4は16h/3まで（32h/5は超過）、M5は梯子の上端16hまで。M5でさらに粗い幅（T2b）は未承認のため調べていない。
3. **相殺の影響はない。** 符号相殺に依存する候補（`CANCELLATION_DEPENDENT`）は0件。L2とL3の差は高々2|δE_trunc|≤1.7e−5。M5 16hのp7（+8.3e−6と−1.05e−5で部分的に相殺）もL2で予算内。oracleを使わないL1（λ_R＋|PF|）では、p7（λ_R=3.13e−4）で予算内に残るのはS2 h、Y4 4h、M5全幅だけで、S2 2h〜16h/7とY4 16h/3は超過する（例：S2 16h/7でL1＝1.01e−3）。p8（λ_R=8.3e−5）とp9ではL2と同じ判定になる。L1はWeyl上界による保守的な参考値で、主比較はL3（L2と一致）で行う。

## 3. 分枝・重なり・数値gate

| 項目 | 結果 |
|---|---|
| 評価器の状態 | 42/42 `EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE`。unitarity・Schur・再構成のgate PASS |
| 分枝 | 最小cut clearance .740 rad（M5 16h、window）。M5 16hのwindow端clearanceは.3984〜.3985 rad。それ以外のcellは≥1.408 rad |
| 状態対応 | 全cellで、元Hの基底群に対する最大重なりの群と、保持Hの基底群に対する最大重なりの群が一致し、それがH_effの最小群で、多重度は1。重なりは最小で.99999609 |
| 混合による位相誤差 | `arcsin((1−w)/w)`は最大3.9e−6 rad（S2 8h/3） |
| 二時点の照合 | 初使用roundとT=409.6でdirectとspectralを照合し、最大差9.7e−12（基準1e−9） |
| 分解の恒等式 | 誤差0（同じ浮動小数点の和）。T1のδE_truncの再利用差も0 |

## 4. Full PFとの同条件比較と構造proxy

ここでいう構造proxyは「m12のsigned S2 piece数×保持block数（one-bodyを含め、p＋1。Full PFは11）」。価格ではなく、ブロック数に比例するという仮説に基づく順序づけの補助である。Q2の補助係数は`1/(1−(合計シフト/ε_E)²)`で、Nの代用ではなく診断用（Q1/Q2は未採択）。

| 公式・幅（予算内で最も粗い） | Full PF proxy | p7 | p8 | p9 | Q2補助係数（p7／Full） |
|---|---:|---:|---:|---:|---:|
| S2 16h/7（m12 1792 piece） | 19712 | 14336（.727） | 16128（.818） | 17920（.909） | 1.246／1.239 |
| Y4 16h/3（2304） | 25344 | 18432（.727） | 20736（.818） | 23040（.909） | 1.154／1.159 |
| M5 16h（2816） | 30976 | 22528（.727） | 25344（.818） | 28160（.909） | 1.000／1.000 |

- 同じ公式・同じ幅では、Truncated p7の合計シフトはFull PFと8e−6 Haしか違わない。このため、両者の比較は主に1shotあたりの価格の比較になる（risk・Nは未評価で、同じになると認定したわけではない）。
- 構造proxyの上では、p7は同じtemplateのFull PFの8/11（約1.375倍安い）。p8は9/11、p9は10/11。
- 公式間の順序はproxy上S2 16h/7 < Y4 16h/3 < M5 16hで、Q2補助係数を掛けても変わらない。ただし、wrapperの固定費、ブロックごとの価格差、隣接pieceの半stepの結合、長いqでの挙動を含まないので、Gの優位は認定しない。

## 5. T3（compiled価格）に向けた候補順位と理由（提案。T3は未承認）

| 順位 | template（p・公式・幅） | 役割 |
|---|---|---|
| 1 | p7・S2・16h/7 | proxy最小。Full PF S2 16h/7（Cの価格検証候補）と同じ幅で、block数の効果だけを分離できる |
| 2 | p7・Y4・16h/3 | 高次公式の対照。Full PF Y4 16h/3と対になる |
| 3 | p7・M5・16h | 合計シフトが最小（Q2で有利になる可能性）。Full PF M5 16hと対になる |
| 4〜6（任意） | p8・同じ3 template | 価格のblock数依存（p7・p8・p10の3点）を検証し、proxyの線形性を確かめる。p8は、oracleを使わないL1（Weyl）でも3 templateすべてが予算内に入る最小のprefix（p7はS2 16h/7とY4 16h/3がL1で超過） |

**価格検証が必要な理由**
- 構造proxyが正しければ、価格比は8/11になる。しかし、wrapperの固定費（a）、one-bodyとfragmentの単価差、identity scalarの位相処理（旧Track Aでは価格差0）、compilerによるstep間の結合で、比は変わり得る。
- 合計シフトがFull PFとほぼ同じなので、Truncatedに利点があるかどうかは実際のcompiled価格で決まる。

**検証の設計案**：CのFull PF案（NEXT_PRICE_VALIDATION）に合わせる。
- 各templateで、m4・m5の2点を校正点、m6を1点のholdoutとし、X/Y両軸で6 wrapper。p7だけなら18 wrapper、p8を含めると36。
- B0とB0Iの価格同値を1組で検査する。
- 事前に、Dのtruncated builder／receiptの拡張が必要になる。

## 6. 先行Track C（`da8b2c3f`）とのSOURCE照合（読取のみ。新しい数値計算は0）

| 確認した項目 | 結果 |
|---|---|
| 表現ラベル | Track Cの記録は「saved H4 DF12 original generation order」。one-bodyと12 DF block、各blockに37係数 |
| DF12とDF10の関係 | Track CのCANDIDATES.csvでは、prefix10とprefix11（I0/I1）のdrop normが5.6668e−14で、全体のmodel残差と同じ値。**block 10・11が数値的に空**であることを示す。DF10と2つの空blockからなる表現とみられる |
| 数値記録 | prefix k_I1（k=0〜9）のground差と重なりが、本研究T1の値と1e−15程度まで一致（例：prefix7_I1 `8.335886140020676e−06`、prefix1_I1 `−0.024623572977672037`）。prefix k_I0はB0の値と一致 |
| sector | Track Cは「N=4、70次元」（全スピン）、本研究は(2,2)の36次元。prefix7_I1のdrop norm `2.180415996479139e−4`は、本研究の36次元sector norm `2.1804159965112266e−4`と1e−14程度で一致。基底状態とdropの最大固有値は(2,2)部分空間にあるとみられる |
| 定数・energy | 定数2.29310124732はAのscalarと同じ。Track Cの元H energy `−2.166387448634761`と本研究の`−2.1663874486347643`の差は3e−15。gap `.23263021512018`も一致 |
| SOURCE・hash | Track Cの入力はc5a3の「public MODEL replay」（identity `acc2262d…`）。本研究のsnapshot hash `12219b7d…`とH hash `c1a6c428…`との対応は、公開記録に書かれていない |

**結論**：数値記録の上では、同じHamiltonian（DF10＋2つの空block）の同じ量とみてよい。ただし、**SOURCE・hashのレベルで同一だという記録はなく、同一性はUNKNOWN**とする。また、Track Cの採択候補coeff_L1_2（204係数をdrop）は、prefix7_I1（180係数）に24個の微小係数を加えたものとみられ、ground差とnormはprefix7_I1と1e−15程度で一致している。この照合はT2の科学条件や結果に影響しない。

## 7. 残る科学的UNKNOWNと未承認の作業

**UNKNOWN（今回の範囲外）**
- 全RPEでのbias、MSE、十分N、G、winner（すべてnull）。
- physical aliasの資格、厳密なground同定、数値誤差の厳密な認証（false）。
- 保存solver proxyと元Hの行列最小値の差は9e−16 Haで、targetの包含区間は未取得。
- Track CとのSOURCE同一性。

**未承認の作業**
- T2b（M5 32h、S2 16h/5、Y4 8hの境界challenge）。
- p0〜p6の追加探索、p5・p6の診断cell。
- T3以降の価格・compile、全roundの新signal、risk、N、MSE、G。
- W0のQ1/Q2の判断。
- 係数単位のdrop、round別のpolicy。

固定SOURCE・input・raw・結果・grant・ledger・marker・KEEP、他担当のworktree・process・branch、共有環境は変更していない。

**mandatory STOP**：次はユーザーが、T3（推奨はp7×3 template、任意でp8×3を追加）とT2bの承認可否、およびW0の契約を判断する。
