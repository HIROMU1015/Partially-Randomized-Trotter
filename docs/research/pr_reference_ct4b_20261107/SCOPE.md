# CT-4B：実行scope

2026-11-07 JST、SERVER 2、独立branch `codex/pr-reference-ct4b-20261107`。基準commit `69e2c0ae4e79febe54df59b6ea7704238467a968`。[承認対象のscope](../pr_reference_ct4a_20261107/SCOPE_PROPOSAL.md)をユーザーが明示承認した範囲に限定する。承認証明書やgrantを新設しない。

固定H4/1.00 Å/STO-3G/DF12、元Hの厳密ground供給・準備費除外、Partial S2/LD3/h=.02/finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit lawを維持。エネルギー精度1.5936e-4 Ha、全失敗率.05、位相cap .4、Qiskit1.3.0/compiler A/rz,sx,x,cx/全結合。B′4-CX controlled-RZZと既存Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar/event符号・X/Y wrapperを固定する。

保存q128/r1/m7・rare/nonrare各j0の2 trajectory、B′/Reference×X/Yの8 native構成だけ。event配列を保存archiveから読み、seedから再抽選しない。seed/hash/vacuum angleは[preflight](../pr_reference_ct4a_20261107/PREFLIGHT.md)とFIXED_INPUTSに固定する。common.pickleを再利用し、SCF/DF/ground solve、新samplingは行わない。

一様precisionはe=-11、epsilon_G=10^-11。保存C2-Bの同一binary angle・同一e・backend2.0.0/strict/seed17のqualified recordだけ再利用する。CT-1 e=-9 wordの再ラベルやprecision/価格最適化は行わない。strict RZ binary target、W、written product順序、負角のactual adjointを保持し、phase fit/理想pi snap/2pi縮約はしない。

承認された順序は、固定入力→8 native準備compile→保存metric/位相/reset/測定の照合→実key固定→missing合成→correctness/有限T/誤差・保存shot条件の確認。旧runner/STOPは再開しない。新SOURCEは[pr_reference_ct4.py](../../../src/trottertracks/pr_reference_ct4.py)、入口は[run_pr_reference_ct4.py](../../../scripts/run_pr_reference_ct4.py)。旧builder/compiler/strict engine/emitter/検証関数を再利用するadapterであり、新合成法・回路構成・fallbackは実装しない。

準備完了時に299 positive keyを固定し、同一eの再利用111、新規取得188と分離した。local検証はwhole依存閉包内の25対象。CT-1のGaussian単体代表1件はwholeにないkeyを要求するため除外し、新規keyを追加しない。D0 frame/inverseの全作用空間で対応Gaussianを含む局所列を検証する。D端/融合block0 half/full・実SOURCE inverse、RTE Z/ZZ中央order0/2、CZ/Z、CSWAP、H/Sdg、保存2 vacuum Pを検証対象とする。localのglobal共通phaseはnative/CT双方に同じ値を用い、strict W/相対phaseを合わせる。

全299 wordはexact W付きstrict guard、±実角のemitter、actual adjointを検証する。各physical RZ位置を実wordへ対応させてT+T†/T†を課金し、共有keyを実行1回にまとめない。native全体直接計数を正本に、role別signed histogramが一致した場合だけV3 role会計を採用する。D融合のsigned補正を負の物理gate数と扱わない。

誤差は同じjoint/reset/observableでtelescopingし、raw synthesis diskを旧diskへ一度だけ加える。CT-0 m7 endpointとworst-case RZを再利用し、alpha1/720、保存199 shot/axisの十分条件を確認する。有限pilotのword成功をcanonical law全体のCT RPE成功保証へ拡張しない。

q128 whole state simulation、全expanded CT QPY、17-qubit dense全operatorは実施しない。保証はstrict局所guardとtelescoping、実装確認は必要な局所full-space/人工検査に限定する。全support期待T、全18round総T、最小T、T-depth/FT depth/space-time、非一様precision/alpha/shot最適化、別q/r/trajectory、SK fallback/CT-3Bへ進まない。

8 native、固定missing、必要局所検査を実利用可能資源に合わせ並列化する。SOURCE/入力/演算順を保ち、CPU/RAM/workerへ恣意的な上限を設けない。共有設定・他ユーザー・他Trackは変更しない。旧SOURCE/raw/result/grant/ledger/marker/STOPを保全する。

固定scopeの完了、strict guard/phase/actual adjointの本質的不一致、公平性の不成立、誤差/保存shot条件の不成立、合成後の対象外keyや別precisionが必要な場合はmandatory STOP。失敗後に条件や候補を自動変更しない。著者の技術検証であり、独立科学レビュー・第三者再実行・新規性判定ではない。
