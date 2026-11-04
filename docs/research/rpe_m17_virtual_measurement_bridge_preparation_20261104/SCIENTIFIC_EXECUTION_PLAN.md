# m17仮想測定bridge：実行条件案と未確定事項

2026-11-04 JST。**実行不可：現在のSOURCEには実環境kernelの導出済み丸め係数がなく、科学runnerも未完成。**
本資料は[前段計画](../rpe_m17_virtual_measurement_bridge_preflight_20261104/MINIMAL_VALIDATION_PLAN.md)の実装到達点を具体化する。
科学実行を承認済みとして記録したり、前段のmandatory STOPを解除したりするものではない。

## 維持するscope

linear H4、隣接1.00 Å、STO-3G、保存legacy DF12、Bのraw physical ground-state vector、
4電子sector70、L_D3、単一δ=.02、m17/q131072/r32/K2、extract_identity_phase、cutoff1e-12。
各軸N411/α=.016957089147912167/ε=.15244325851620388、保存μ0/PF/RTE/stat/βを変更しない。
数学的IID uniform53 categoricalと独立uniform53 measurement wordのModel IIだけを対象にする。
固定PCG64・他round・全18round・compiled/noise・最終energy/費用は対象外。

固定入力はrun_virtual_bridge.INPUTSの6原本：latest RESULT/export/laws/包含operators、B state、元m17 F/R64。
SOURCE hashとbyte照合で数値再現性を保持する。別のSOURCE replay・SCF/DF/H生成・state solve・CDF再構成は行わない。
latest exportのlocal binary64 matrixから新numeric P_i64を構成する点を明示し、run02の未保存payloadとは区別する。

## 実行前に不足するもの

1. 実際のmatvec、complex scale/subtract、vdotの丸め・underflowを含むKernelBoundsの導出。
   既存BLASとの対応SOURCE、実行経路、必要な数値前提を説明できること。toy toleranceやversionから値を作らない。
2. 全supportでintermediateが有限であることの上界と、`abs(signal)`評価の誤差上界。
   coordinate/magnitudeの二つのguardを区別する。
3. 実行時の入力/θ/成分順とcontextの照合、数値構成1回、最大包含pass、fresh出力と中断保存をつなぐ薄いrunner。
   現在の--runは不足を表示してexit2するだけで、これらを科学実行しない。
4. 完成SOURCEを公開commitへ固定し、下記scopeを対象としたユーザーの新科学実行許可を受ける。
   承認証明書・ID・grant等は作らない。重要科学レビューを行う場合は別の許可。

## 完成後に提案する最大科学call

- 固定数値contextの構成1回：numeric basis11/P_i64 218、保存F/R64・ψの照合、保存θ2個に対するcos/sin出力の取得。
- residual/majorant passは96 bitから1回。純粋な包含丸め幅だけで判定できない場合のみ160、224 bitへ、最大3回。
  218 residualや必要normは各pass内に含め、numeric contextは再生成しない。
- 保存logical intervalと保存sを読み、Model IIの上界・strict予算比較を最大各pass1回。
  新たな理想平均q乗・保存sの再計算は不要。phase transferの二重計上を避ける。
- 全support完了やkernel前提が未確立ならU_virtual/U_execはnull。平均majorantが得られてもguard保証へ転用しない。

異常時のretry/resumeやalgorithm切替は自動で行わず、partial結果を残してSTOP。
数値条件を変更してPASSを狙わない。新namespaceによる再試行も自動開始しない。
sampling・RNG replay・m17直接trajectory・仮想shot・回路build/compile・費用/成功率再解析は0。

## 判定と出力

算術条件、law/context identity、全supportのguard完了が揃った場合だけ

    U_virtual,IID,a ≤ U_logical,a + η_action/raw,a + 3·2^-53,
    upper(U_virtual,IID,a+s_a) < lower(ε_a)

を両軸で判定する。strict境界一致は認定しない。上界の保守性で不成立でもRPE失敗の証明ではない。
guard未解決なら`ARITHMETIC_BOUND_OBTAINED_BUT_COMPLETION_UNRESOLVED`、上界null。
その他は未判定/入力不整合/中断を区別して記録する。
固定PCG64に対する`U_exec_upper`とPRNG bridgeは、Model II判定にかかわらずnullのまま。
旧結果を更新せず、新結果には前提、context、各上端、平均/全supportの別、部分結果を残す。

## 性能・STOP

支配的処理は218 componentの70次元residual評価。basis cache、独立componentのbatch/並列、FLINT内部threadを使う。
q/rはaffine mapのbinary powerで、trajectory列は構成しない。
実行前にCPU/RAM/GPU/共有負荷を確認し、quotaと非干渉を守って自jobのprocess/BLAS/FLINTを調整する。
CPU/RAM/worker/wall-timeの人工capは設けない。今回の小さな人工検査のthread1は科学契約の制限ではない。
高速化率や実H4の上界幅・所要時間は未測定であり主張しない。

今回の技術準備は公開照合後にmandatory STOP。
上記の未確定事項を解決する前に、--runを実H4上界評価として起動しない。
