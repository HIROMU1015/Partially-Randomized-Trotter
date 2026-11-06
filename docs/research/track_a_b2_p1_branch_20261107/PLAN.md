# B2 P1保存raw平均：全18round branch/energy復元

2026-11-07 JST。ユーザーが固定保存信号の再解析・branch/energy推定を明示承認済み。
保存全18roundはd7a55ac181de71cd2b2c49644810373cafd0a450、endpointはea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5。
P1と予算・allowance・入力を変更しない。実行前に薄いadapterと人工testをcommit固定する。

保存metadataと各元結果を照合し、raw平均のArgだけを既存reconstruct_rpe_phaseへ渡す。
初期aliasは[-128,128]、delta=.02、shift0、beta=.4から検証し、既知Eを使わない。
scalar/vectorizedの既存選択と曖昧性規則、best/runner-up gapを診断する。
全復元完了後のみ固定元H energyを読み、correctnessとP1精度を評価する。
oracle branchは事後照合だけで、実decoderの選択を修正しない。

既存strict beta<pi/3と共通target phase条件の成立範囲だけを報告し、
raw平均を中心とする任意の測定摂動や95%測定成功保証はUNASSESSED/NOT_CERTIFIEDに保持する。
不一致・alias不適格・branch曖昧性等で停止し、既知Eやパラメータ変更で救済しない。
状態/operator/信号生成、private入力読込、sampling、回路、shot、価格、baseline、レビューは行わない。
結果・REPORT公開とremote SHA/tree/全文照合後mandatory STOP。
