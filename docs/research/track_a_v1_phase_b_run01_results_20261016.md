# V1 Phase B run01：登録S4全4軸PASS・正常終了STOP

Phase Bの登録4軸検証は全てPASSし、wall3037.384秒（50分37秒）、native exit0、全process回収で正常終了した。driverは`COMPLETE_MANDATORY_STOP`を記録し、observerも同じ終了状態を確認した。再fit・追加compile・自動retry・5精度費用再評価・V2実行は0。次Phaseへは進めていない。

[4軸全表・実行終了・保全監査](../../artifacts/resource_applicability/track_a_v1_phase_b_run01/2026-10-16/README.md)、[固定モデルと検証契約](track_a_v1_phase_b_readiness_20261016.md)、[起動時点の履歴](track_a_v1_phase_b_run01_started_20261016.md)。
科学実行SOURCEは `fcf37e6794e2e51fc95e958a75fc8f75fb049423`、Phase A入力のresult commitは `bf865de1efae51ce1663c878ad1e9daa5bb915c2`。起動後の文書commitは科学SOURCEを変更していない。

## 条件と4軸結果

linear H6、間隔1Å、STO-3G、tol-only DF19、12 modes、sector400、T=.8。登録対象はB0 prefix11/q1（δ=.8）とB0+I prefix15/q8（δ=.1）のS4、各cosine/sine。
P=0・RZ、symmetric_directional、既存native Gaussian lowering、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・topologyなし。compiler hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

|構成|軸|固定予測RZ|実測RZ|相対誤差|compile秒|
|---|---|---:|---:|---:|---:|
|B0 p11 q1 S4|cosine|65196.8|65261|0.0983742%|46.602|
|B0 p11 q1 S4|sine|65196.8|65261|0.0983742%|47.394|
|B0+I p15 q8 S4|cosine|699389|699389|0%|530.674|
|B0+I p15 q8 S4|sine|699389|699389|0%|534.899|

誤差は各軸で`abs(frozen_prediction - measured_RZ)/measured_RZ`を有理数計算した。B0の厳密値は`321/326305`、B0+Iは0。全4軸で2%未満。両軸平均で判定していない。各軸PASSを確認してから次のcompileを開始し、総compileは登録どおり4。固定model SHA `406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211` は終了後も一致し、実測による係数修正は行っていない。

各軸のCX実測はB0が29948、B0+Iが323156。今回のmodel合否基準はRZのみで、CX費用modelを検証したとはしない。
native branch/wrapperの数値照合で最大discrepancyはB0が1.027e-14、B0+Iが6.001e-14。入力fingerprintはcompile前後で一致、compiler hashも固定値に一致した。数値的一致は厳密な近似誤差certificateではない。

## 終了・監視・保全

予約と完了はsnapshot load1、archive decode1、native evolution2、control probe8、measured wrapper4、compile4で上限と一致。新しいsignal、scalar、SCF、DF分解、state solver、trajectoryは実行していない。

owned launcherのnative wait0、driver退出、all_reaped=true、first_stop=null、cleanup_errors=[]。外部監視の最大間隔1.006952秒は上限5秒内。peak総RSSは8.8903GiB、最大観測総CPUは3034.87秒で契約内、stderr0 bytes。総wallは回路構築、数値control、fingerprint、compile、回収を含む。4回のcompile秒だけを全wallと混同しない。

終了後にSOURCE17ファイル、provider230ファイル、runtime34,273ファイル、入力9点、既存grant/ledger/carry等650ファイルをhash照合し、全て不変だった。H5・V2・旧runのgrant・ledger・carryは変更していない。予約をrefund/resetせず、既存rawと今回のone-shot証拠を保全した。primary agentによる保存記録の照合であり、別担当・外部再現・CIの検証ではない。

## PASSの範囲と残る判断

得られた結論は**登録S4の4軸で固定モデルの相対誤差が2%未満**という局所的検証である。追加104候補全域で同じ誤差上限が成り立つ証明、旧B0 binary profileの同値証明、matched B0/B0Iのidentity効果の因果比較は得ていない。
旧B0のbinary profile未在庫、経験的数値allowance、保守的bias保証未解決を維持する。model推定を実測として扱わず、最良候補の点選択・PR標本費用・母集団保証を区別する。

5精度（.05/.01/.005/.001/.0005）の最良決定論費用とPR削減率の再評価に必要な検証入力は揃った。[将来再会計計画](../../artifacts/resource_applicability/track_a_v1_phase_b_readiness/2026-10-16/future_reaccounting_plan.json)と`validated_additional_cost_rows`は準備済みだが、今回それらを呼ぶ再会計は実行していない。結果を報告してmandatory STOPし、次Phaseは別の明示指示後に判断する。
