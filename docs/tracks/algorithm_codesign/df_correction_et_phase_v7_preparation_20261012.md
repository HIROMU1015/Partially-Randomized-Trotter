# E/T v7：raw scalar phaseとW loweringの会計整合修正

v6の実測ledgerで、nearest W distance約0.098だけがguard停止を支配した。
compiled raw circuitをphase-strictに評価した四項の合計は約4.41e-8。
source v6/result/markerを保持し、利用者の継続認可に基づき技術修正を行う。

## 正当化と変更の範囲

実行したnative circuitはraw scalar phaseを保持する。Operator comparisonはその位相も含む。
W substitutionがunsupportedで選択されない場合、raw phaseとnearest Wの距離は
実行回路を変える摂動ではない。これをerror upperへ加算していた不整合を修正する。
nearest W距離は新diagnostic fieldに保存し、supported Wの場合は従来と同じ距離を計上する。
unsupportedの場合はraw phaseとIRを変更せず、存在しないlowering errorだけ0とする。
誤差許容2e-7、phase-strict Operator、reference radius、forward guard、angle loweringは維持する。
誤った実行位相はfull Operator residualで検出する。phase minimizationは行わない。

非対応phaseの実装費用はunknownを明示し、T completionの既存policyを維持する。
phase_supported=falseならactual total T/lowerはNone、full certification=false。
費用0として扱わず、phase key追加・source条件・scientific comparison・capsを変更しない。
共有APIは変更せず、新namespace `df_correction_et_phase_v7`内のinventory会計だけを修正。
v6 native evaluator/Operator guard/storage/guard/accountingはbyte同一。

## 検証と実行

91 focused PASS（phase9、observer6、integration53、storage23）。
raw phase保持、unsupported T不確定、supported W距離計上、誤位相/符号/配線拒否を確認。
初回90 PASS/1 FAILは負位相のQiskit modulo2pi正規化をtestが反映していなかったためで、
保存raw値を比較するtestへ修正し、全attemptを保持した。登録fixture/synthesisは0。

新contract v7は科学sectionとcapsを継承し、このimplementation fixをmetadataに明記。
同じE-r3/r6、seed/rank/time/precision/baseline、native22/T64、2workers/threads1、64MiBを維持。
CDは旧6行参照だけ。G10/RA-D0/分子/GPU/LP/samplingは行わない。

source固定/push後、authorization-only直接子を作り、fresh result v4で一回実行する。
各runのretry0/marker/terminalを保持する。利用者は技術的停止の修正継続を認可済み。
科学的条件/保証/capsの変更が必要なら判断を戻し、比較完了時はmandatory STOP。
科学的勝敗/新規性/研究方針の判断はGPTへ戻す。
