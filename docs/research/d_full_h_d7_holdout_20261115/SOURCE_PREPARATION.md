# D7独立holdout：実行前固定

利用者がC2の4件の記述的比較を採択し、§10.5照合、指定3 fresh pathのsampling、X/Y個別最大6 compile、
固定予測との4比較、公開までを明示許可した。本書は科学条件・実装来歴を記すもので、承認証明機構ではない。

## 参照と観測前照合

- 規則：f0765067a5a8b39888e36f3a8b80f1bdbda69223、full_h_d7_holdout_rule_proposal_20261115/PROPOSAL.mdとHANDOFF_D1.md。
- モデル/予測：b5165f11bb63f9eb2a6bfe385ba75d8003b7e724、full_h_d7_model_freeze_20261115/run01/FROZEN_MODELS.json。
- D7 plan：7d0c39066d4ed900cfe7f5de6f4c7c5a3b2ec41f、d_full_h_p2_calibration_resume_20261115/FROZEN_JOBS.jsonのplan。
- 較正完了/実装来歴：e628c0d69c7da559369d99f323e3656a96c0728a。

既存frozen_model_gateを観測前に実施しPASS。4保存予測を(cell_id,axis)で一意に選び、physical/cohort/index/step_seeds、
feature/order、path数へ結合した。predicted_RZの分子・分母を直接保持し、係数から再予測・再fitしない。
exact特徴量の正規化不足は実行前metadata検査で検出して修正。科学sampling/build/compileはその時点で0。

[PREOPEN_CHECK.json](PREOPEN_CHECK.json)はCの公開前照合を引継ぎ、その後の関連ref、既知private rootの名前、関連jobを確認。
Cの照合後に動いたm5-m4 refの差分はCモデル凍結/公開metadataで、固定モデルblobはb516と同一、holdout取得出力なし。
既知保存先にM5/M6なし、関連科学jobなし。同等既取得compileの再利用0。観測価格を開かずに照合した。
未知host/archiveを網羅したとは主張しない。固定holdout cohort/seedは較正と非重複。

## 固定実行範囲

| cell | q/R/T | physical ID | path | 最大compile |
|---|---|---|---|---|
| M5_ROUND | 32/283/3.2 | FULL_H_bc8aab34256b73b68025 | 0,1 | 4 |
| M6_ROUND | 64/566/6.4 | FULL_H_4ec8d73571760a6830b8 | 0 | 2 |

保存H4直線1Å/STO-3G/tol-only DF10、one-body込みFull-H pure RTE/L_D0/K2/κ2、元scalar/event phase、
symmetric-directional、新ZZ lowering、測定付きfull wrapper/P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を維持。
元native作用gate（index0の7検査）と全pathのparity/scalar/event/XY gateを再利用し、各pathのsamplingは一度。
X/Yは同じpathの別測定軸であり、標本数を合算しない。

## 実装と性能

run_full_h_d7_holdout.pyは既存P.execute/accelerated childへ接続する最小入口。
既存normal sampler/native trial/compiler/profileを使用。独立2 cell・3 pathをprocess並列、各X/Yも個別processでcompile。
既存exact controlled gate cache・canonical byte同値のfast fingerprintを再利用。
許可外のH4試運転は行わず、実行前にSOURCEと[FROZEN_JOBS.json](FROZEN_JOBS.json)をcommitで固定する。

既存caller/acceleratorへ任意の協調的progress callbackとcompiler/path終了codeを追加した。計算内容は変更しない。
親process observerを再利用しPATH別stageを読む。worker内の非同期stack dump/threadは追加しない。
共有環境・package/OS設定は変更しない。64CPU/約489GiB空きmemoryを観測し、過去runのworker数を恒常上限にしない。

人工検証は既存caller11、既存acceleration13、新比較/固定入口8。toy/fake compilerを再利用し、normal sampling1回、
別PIDのX/Y、phase/fingerprint/record対応、0観測・欠測・不一致・exact有理数・モデル判定なしを検査する。
testはD1本人による技術検査であり、独立reviewや実H4の性能保証ではない。

結果：caller11/acceleration13/holdout8の計32 test PASS。追加したCLI人工testの初回2回は
mockがGit metadataまで置き換えたためERRORとなり、mockを対象commandへ限定して8件を再検査した。
import-only preflightは23 module PASS。SOURCE固定前に関連refが観測前照合時点から変わらず、両科学出力先がfreshであることを再確認した。

## 比較・公開・STOP

Pは保存予測、Oはm5同軸2 path平均、m6同軸1 path値。P/O/P−O/(P−O)/Oをexact有理数で保存し、%だけ6桁で表示。
O=0は相対誤差nullとZERO_OBSERVED_PRICE。欠測・gate不成立は比較不能として保全し、無断retry/seed差替えなし。
4件有効ならDESCRIPTIVE_COMPLETE（O=0を含む場合はRELATIVE_UNDEFINED）、不足はINCOMPLETE_OR_INVALID。
tolerance=null、model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE。モデルPASS/FAILは付けない。

publish_full_h_d7_holdout.pyは既存lossless receipt公開経路を再利用し、各6軸の元compile bytes、価格・実装証拠・raw locatorを保存する。
公開SHAからreceiptと各軸証拠を照合してから、4比較行をCの予測locatorとD1の観測locatorへ結合する。
較正raw、C/C2原本、旧STOPは保全。m6 n2・追加round・再fit・観測後閾値・全law E[C]/CI/N/G・winnerへ進まない。
commit/push・remote/公開成果物照合後mandatory STOP。
