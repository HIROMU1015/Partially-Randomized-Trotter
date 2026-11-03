# 2026-10-09 Track A：H4限定scientific runnerの準備

利用者の「続きを行って」を受け、直前のAX-2A native準備v2に残したrunner/cap/input bindingを実装した。
今回の範囲は準備とsynthetic検証であり、科学実行認可を発行しない。

判断：H4第一phaseの8 correctness cellと28 wrapperへ入口を限定する。
理由：H4一致と負荷を確認する前にH6生成・評価を開始する必要はなく、未確定のH6 DF/state政策を
既存H4 rank12の入力へ混在させないため。H6/H8/main campaignは別判断のまま保持する。
H4の保存NPZはM1 byte hashと一致し、stdlibで保存metadataだけを照合した。数値配列loadは0。

独立parent watchdog、child CPU/RLIMIT、before-call counters、exclusive outputと単回claimを接続した。
H4 referenceはexpm/eigh比較とRayleigh residualを分離し、roundoffを含む総数値allowanceは未認定とする。
pilotが一致してもaccuracy eligibility/shot/winnerを主張せず、後のheadroom検証を必要条件として残す。

新規57＋旧117の174 local tests passed、fail/skip0。
固定toyとダミーprocess/compiler/samplerだけで、分子科学計算・sampling・compile・GPUは0。
既存科学source、旧result/manifest、原稿、Track B、v1/v2準備証拠はbyte不変で保持した。
隔離branchのlocal未commit準備で、immutable CI・外部再現ではない。

[仕様と次段条件](../track_a_ax2b_h4_runner_preparation_v3.md)と
[別inventory](../../../artifacts/resource_applicability/track_a_ax2b_h4_runner_preparation/2026-10-09/h4_runner_preparation_inventory_v3.json)を入口とする。
終端：`AX2B_H4_RUNNER_PREPARED_SCIENCE_NOT_AUTHORIZED`。科学実行と次段は未認可、mandatory STOP。
