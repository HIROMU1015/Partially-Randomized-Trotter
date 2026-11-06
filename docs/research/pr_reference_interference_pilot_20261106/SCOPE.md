# 新A reference干渉：限定pilotの実行範囲

ユーザーがA-1〜A-4の条件付き実装・correctness・simulation・compile・限定価格・必要なRPE再会計を明示承認した範囲で実施する。承認証明や新grantは作らない。旧Track C/C2、新B等は再開しない。

## 再利用する固定条件とSOURCE

- 調査基準2397c5ac511148423ae89f25f74f1de967d1d39a、科学SOURCE602ca4b67b1037e221e910a2733444630c5add62。
- 保存済みLinear H4、1.00 Å、STO-3G、DF12。Partial S2、LD3、h=.02、finite RTE K=2。新しいH・DF fit・ground solveは行わない。
- 旧Track Cのordinary diag(I,V)・fresh iid Hadamard shotをprimaryとする。独立2分枝の別meanへ無断変更しない。独立joint realizationのV0†V1は人工correctnessでも確認する。
- RPEへ進む場合は既存18round、energy精度1.5936e-4 Ha、全失敗率.05、位相予算.4 radを参照し、新しい成功保証を主張しない。
- 元H groundの供給を仮定し準備費を除外する。追加8 reference qubit、2×8 CSWAP、位相補正、8 reference resetは計上する。compiler Aの全結合モデルを維持し、実接続のrouting費やT価格を推定しない。

## 有限な実行範囲

A-1は登録physical basis familyと正負D時間、canonical finite RTEの両stratumでvacuum phaseを確認する。phaseはgate/係数記録だけから求め、未知E0/gap/ground vectorを使わない。理想RZ・XXPlusYY、共役Z/ZZという条件をgeneric Pauliへ広げない。

A-2は2qubit人工例の全operator・X/Y、非可換F/G、非固有target、scalar/sign/負時間、actual adjoint、独立2分枝、reference不成立反例。実入力はq1/r1の通常・rare realization各1件で、9qubit直接制御と17qubitreferenceのsource/native信号を比較する。新17qubit全operatorは生成しない。人工許容1e-11、H4診断1e-9はimplementation診断であり、全RPE誤差上界ではない。

A-3の価格はq=1,2,8、r=1、通常/少なくとも1個のorder2があるrare stratum、各8標本（計48paired realization）、両X/Yに限定する。seed namespaceはSOURCE内に固定する。既存canonical conditional samplerを再利用し、適切なstratum確率で加重する。候補は全trajectoryを時系列の固定境界で2 laneへ分割する1方式だけ。block配置optimizerや別PFを追加しない。

対照は既存basis非制御・diagonal中央制御と、適用可能なSimon–Love型のpartial-S2制御である。後者は左Dを非制御、右Dを方向制御し、control0でactual inverseが相殺し、control1で同じrandom Rを挟む。R部分は既存の制御を保持する。新gadgetと同様にcorrectnessを確認してから価格を採用する。

価格取得前に、既存 `df_partial_s2_repeated.py` のboundary_optimized構成を直接再利用し、隣接する同じD blockの融合とscalar集約を両方式へ認める。Simon–Love側も同じ既存境界融合を制御規約に合わせる。q2/q8の全9qubit operatorで元の同じtrajectoryとの一致を確認する。A-3の移行判定は、両方の実装済みbaselineに対して利益の余地が残る資源があるかで行う。

A-3でq8のいずれの登録資源にもpaired差の診断上の利益がなければSTOP。少なくとも一資源に利益の余地があればA-4の既存shot再利用条件と取得済み短round価格を接続する。未取得長roundやr>1価格、T、全support native誤差を0で補完しない。診断MC区間を保証付き優越としない。

## 実装差分と保全

既存DF builder、rte._make_event、conditional_flags、compiler A、phase/X/Y/statevector規約を再利用する。新しい差分は2-register reference gadget、phase追跡、強制御lift、限定比較runnerだけ。過去のrunner、prepare_common、合成、price関数は起動しない。

SOURCEと出力はこのbranchと独立artifact領域に限定する。既存cacheを読取専用でsnapshotし、公表済みcommon arrays・DF/ground入力との一致を確認する。CPU/RAMに独自上限を設けず、独立task数とOS affinityに応じたworkerを使う。BLASの1thread/workerはprocess並列時の重複並列を防ぐものであり、計算全体のCPU制限ではない。

A-1/A-2不成立、A-3の利益消失、必要保証の不足、または有限scope完了でmandatory STOP。既存STOPは変更せず、別PF・law・配置・分子へ自動拡張しない。
