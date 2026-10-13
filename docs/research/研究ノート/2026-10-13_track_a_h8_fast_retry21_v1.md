# H8 posttranspile guard承認と高速化版一回再実行

2026-10-13 JST。利用者が「続けて大丈夫です」と回答し、提示済み1000万instructions・累積compile21・control44、fast S4/q8一度再実行→成功後残7への変更を承認した。old input/science/compiler/seed・CPU/RSS/ASを維持し、failed compile1/control4とH6/H4/stateのcarry/未知予約を保持する。source/postguard/call上限を勝手に変更しない。

new23 metadata/mock tests PASS、old44/40は再実行せずnamespace cloneのworkerへ新bindingだけを適用した。SOURCE1d064749でnewgrant/outputを準備したが、待機controller CPU31を子coordinatorが継承してfresh CPU gateがSTOP。science worker・lock・compile/trajectoryは0であり、計算失敗分を返却したものではない。子exec前にapproved assigned CPUsを復元し、SOURCE9bfea8bde44cbb2ea9187476ddd496066a8853b3へ固定、新auth-heavy02/control02/heavy02へ起動した。

coordinator410647、worker410910、pipeline410386、source/manifest/grant/input/profile/carryとfresh resourceを確認して起動。最初の独立監視はSTOP理由null・pressure0/newOOM0。重いcase1worker、successとnative wait/reap後だけ残7最大3workerへ接続し、最終20metric/累積21call/44controlを照合する。full H8や次stageは許可しない。

API既存GitHub接続でsourceをnon-force公開し、別REVIEWで軽量auth/plan/launch/profile/test/closure資料を固定する。secret/NPZ/runtime/cacheをcommitせず、共有環境/既存venv/H4/他job/GPUを変更しない。[資料入口](../../../artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/README.md)。計算開始後はユーザー指示に従いチャットを終了できる。完了/STOP時の最終集計はnative記録を見て別に行う。

## 保存結果の完了集約

[完了結果](docs/research/track_a_h8_fast_feasibility_result_v1.md)：linear H8/1Å/STO-3G/tol-only DF28/16modes/sector4900/T=.8、SOURCE9bfea8bd。8/8ケース・20/20 full-wrapper/累積21actual、control40+old4=44、trajectory4/occurrence15。heavy約1時間59分・RSS6.815GiB、残7約50分・最大3worker。全exit0/reap/cleanup0・STOPなし。新解析はmetadata/streaming SHAだけでscience0、母平均/正式精度一致/本格mapは未認定。本格実行grantなし。

posttranspile sizeは5,115,015で新1000万guard内、旧failed runの欠測を補完しない。heavyのhash約60%・control約30%・transpile約6.5%、build内のhashを二重加算しない。全4PR trajectoryの44event次数0なので希少費用は未認定。normal8/heavy3のbudgetは提案、今回実測最大3のみ。新科学計算/再tests/新grant0、軽量CSV/JSON/監査/研究資料だけを公開する。
