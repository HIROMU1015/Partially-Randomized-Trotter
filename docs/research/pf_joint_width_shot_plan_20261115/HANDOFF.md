# A/D・統合担当へ：決定論PFの幅・shot配分共同比較

2026-11-15 JST、B。[REPORT](REPORT.md)が比較規約・初回有限案、[WIDTH_SIGNAL_COVERAGE.json](WIDTH_SIGNAL_COVERAGE.json)が幅ごとの元ID・schedule・保存mean・不足roundの索引。**今回は読取・metadata整理・計画のみ。科学call、価格・費用算術は0件。**

## Bの最初の実行案（未採択・未実行）

- 対象は現6幅：Full PF／B0I p7 × S2 16h/7、標準Y4 16h/3、M5 16h。A1adb7e65の6 packet（統合科学SOURCE fb9a1878／carrier06a15ee4）を再利用する。
- 旧N0=[59,55,51,47,43,39,35,31,27,23,19,15,11]のQ3は全6候補でcomponent適合済み。原API return・CPU/peak RAM欠測の部分保存を変更せず、6件を再実行しない。
- 各方式のS2達成費用（Full 838,693,580 RZ／p7 593,555,352 RZ、B2引用）を基準に、**β=3/4という共通費用challengeを提案**する。βは科学的な安全率やMSE閾値ではない。新配分・費用はまだ算出していない。
- REPORT §3の生成規則A（元shot形状）とB（N0/Cによる単価を考慮した振替）を、各系列で同じ費用上限・1≤N≤N0の下に適用する。各X/Y列は価格に依存して別々に確定する。
- **各候補最大2配分、計最大12 Q3**。N0・保存結果・進行中job・A/B同一列との重複は§10.5で再利用／統合し、件数を減らす。無理に12件へ増やさない。
- Z1＋T1/Q3、保存DF10 solver proxy、全事象平方loss、exact τ、元IID二項lawを維持。PFはdeterministic raw/B=1で、PR finite-Kを追加しない。
- scope案はρ1、outcomes4225、verification55043173、非終端frontier8192、transitions141559036／call。具体列のcount-only確認と新閉包gateが必要。新配分へ旧閉包PASSを流用しない。
- 上界≤τで適合、下界>τで不適合、跨ぐとINCONCLUSIVE、必要gate不成立はUNKNOWN。保存状態は別記。重大異常時は未開始callを止め、開始済み証拠を保全し、終了後mandatory STOP。

次の実行許可には、提案β・2生成規則、保存価格を使った具体列の確定算術、新固定配分の条件付き費用会計、最大12 Q3の有限範囲を含める。実行前に新INPUTS・callerをcommitする。これは今回の実行許可を示すものではない。

## Dへの依頼境界

**現6候補は価格完成待ちではない。** 1002f1dfのAXIS_TERMS.jsonの26行／候補を、元configuration・round・axisで選択する。元receipt／C source、96直接＋60条件付き閉包、固定compiler/profile・full measured wrapper/P0の条件をそのまま使う。新Nに合わせて1shot価格を変えたり、normalizationを掛けたりしない。

別幅は次の元configurationに対してcoverageを確認する。全桁・全13physical ID・q/T/実幅のlocatorは索引にある。

| 調査する幅 | Full PF元ID先頭 | B0I p7元ID先頭 | 目的 |
|---|---|---|---|
| S2 8h/3 | 7754671b | 5576418d | 近い粗幅の費用節約と必要shot増の交換を最終Q3で比較 |
| Y4 32h/5 | 9e930a47 | dd8bfc24 | 同上、高次公式の粗幅challenge |
| S2 2h | 9e96ed75 | 155a3a2e | 細幅の精度改善が1shot価格増を補えるかの対照 |
| Y4 4h | 99e53903 | 05d246cb | 同上 |
| M5 8h | 24286ac5 | a62d4ab0 | 同上 |

この表は新compile指示ではない。既存の同物理round・receiptがあれば先に再利用可否を照合し、6幅の閉包モデルを別幅へ無条件に移さない。M5 32hは旧提案だけで新固定IDがないため、A/Bのmetadata固定前に取得を始めない。

## Aへの依頼境界

- 近い粗幅S2 8h/3のFull/p7は現行mean候補m0,1,2,3,12、不足は**m4〜m11**。
- 近い粗幅Y4 32h/5のFull/p7は現行mean候補m0〜m5,12、不足は**m6〜m11**。
- 細幅S2 2h・Y4 4h・M5 8hは両方式で現行mean候補m0,1,2,3,12、不足は**m4〜m11**。m4のlegacy meanは再資格化前として分ける。
- p7 S2 h（e6754564）は保存13round packetとtotal metadataがあり、signal再取得不要。新Q3・同候補価格は今回の12件の範囲外。
- 索引は同physical metadataの保存meanを挙げたもので、別canonicalへ再ラベルしてintake完了とするものではない。保存H/state/snapshot・law・normalization・target・scalar・元数値SOURCEを照合し、必要なsource-bound packetだけを作る。
- M5 32hや、early実幅cellしかないp7 Y4 h・2h・M5 h・2h・4hは、独立configuration・必要科学scopeの固定が先。元16hのbranch窓や局所診断は移送しない。

旧局所ε_trot超過は新Q3不適合の代用にしない。全幅の信号を一括取得せず、Bの初回結果とDの価格coverageから、次の研究判断を変える幅を選ぶ。各不足の新科学計算には別の明示許可が必要。

## 既存実装の再利用とロードマップへの影響

既存total binding/kernel/高速backend・whole-shot仕様を使う。prepare/旧閉包basisには元configuration・元signalを使い、T.evaluateの既存bridgeで返却をtotal canonicalへ束縛する。各jobはallocation1本。worker自身のAPI/partials/resource保存はrun_pr_df_fullh_total_q3_fixed.pyの修正済み経路を再利用し、PF metadataを渡す実入口・監視例外を人工fixtureで確認してからscienceへ進む。MCは起動しない。

正式rpe_shot_allocation.compare／cost/selectは資格Nを要求するため、Q3条件付き適合を十分Nへ昇格して通さない。B2の既存price join・exact integer算術を使い、**条件付き固定配分Tの有限menu**として比較する。新controller、N資格producer、価格モデルは不要である。

統合担当が追補する差分：現6幅/N0は達成済み基準点として保持し、「幅は公式ごと固定・N共通」から「公式ごとにΔ_PF・whole-shot配分を選べる有限探索計画」へ次課題を具体化した。初回は保存6 packetと固定価格で進められ、別幅価格のD待ちはない。29保存configurationのcoverageと、別幅に必要なA/D証拠を分離した。**まだ共同探索を実行した結果ではなく、12件も実行待ちである。**

ロードマップ本体は統合担当が更新する。旧T2・Q3・B2と全旧STOP、SOURCE/raw/契約/grant/ledger/marker/KEEPを保全。sufficient_N・global RMSE・G_qualified・大域winnerは新認定しない。計画commit/push・remote照合後 **mandatory STOP**。
