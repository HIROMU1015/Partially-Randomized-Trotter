# B0I p7／標準Yoshida4・32h/5の有限価格較正

科学実行前固定。保存linear H4/1Å/STO-3G/tol-only DF10・元H/state/snapshot・prefix7/fragment0..6保存順・P0除外・標準Y4 signed weights・scalar c_H+tail identityをtotal Tへ一回・symmetric_directional制御・測定付きfull Hadamard X/Yを維持する。configuration dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0。実行runtimeは既存 /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python、Python3.11.0rc1/NumPy1.26.4/Qiskit1.3.0。compiler rz/sx/x/cx、opt1、seed17、元native/lowering/wrapper SOURCEを変更しない。

| role | m | T | q | actual幅 | physical ID | 上限 |
|---|---:|---:|---:|---:|---|---:|
| CALIBRATION | 5 | 3.2 | 5 | .64 | 4eccd52bab8d4d790ab9 | X/Y 2 compile |
| CALIBRATION | 6 | 6.4 | 10 | .64 | 77d24715893869133820 | X/Y 2 compile |
| HOLDOUT | 9 | 51.2 | 80 | .64 | f0a99e91997c44c59ad4 | X/Y 2 compile |

m0..m4保存10軸はplan d01c780d0のreceipt/元trial/C受入へ個別に結合し、旧IDと元SOURCE/cohortを保持してearly単価に使う。要求幅でなくactual幅・q/T・signed structure・scalar・wrapper/compiler/control/measurementを照合した。m3の2軸だけはCが既に受入済みのe2179f68 legacy nativeと固定同等性証拠を保持し、af284 SOURCEへ付け替えない。10軸はactual幅.1/.2/.4/.4/16h/3なので粗幅fitから除外する。

比較用モデルは軸別 C_a(q)=alpha_a+beta_a*q。Cの既存price_model_validation.record/fit/predictを使い、CALIBRATIONのm5/q5とm6/q10だけでexact二係数を解く。公開・C実装guard通過済み4軸からFROZEN_MODEL.jsonに両軸係数・元receipt SOURCE・m9/q80のexact予測を保存してcommitする。それまではm9の観測RZを取得・開封しない。旧のm4/m5/m6モデル・係数・2% policyは変更/流用しない。

その後HOLDOUT m9を別namespace・run labelでX/Y個別compileし、実装gateとC guardを適用する。記録差は observed_RZ−frozen_prediction、相対差は同差/observed_RZ（abs相対差も保存）。新しい採否閾値は採択せず tolerance=null。差0はEXACT_MATCH_AT_Q80、差ありはDEVIATION_AT_Q80として有効直接価格とモデル支持範囲を分ける。m9をfitへ戻さず、特徴量変更・再fit・追加compileはしない。q640の誤差保証はUNKNOWN/nullで、q80の確認を長qの証明へ昇格しない。

元m0 callerのchild/worker/fast context・compose再利用・fingerprint・compiler/cache・7作用gateとreceipt/C guardを再利用する。scope callbackと公開済みprice-binding helperだけを共有化し、元m0/m1/m2の既定契約は維持する。metadata親とinstall先行native子を別processにする。独立cal2 cellは利用可能資源で並列、m9は係数凍結後だけ投入する。恒常的CPU/RAM制限・共有設定変更は0、元資源観測helperを使用。協調logとchild/worker exit code/signalを保存し、periodic faulthandlerは使用しない。

§10.5の差分照合はPREFLIGHT.json。保存10軸に加え全late roundの同物理点を既存carrier/catalog/private exact path・関連jobでmetadata照合し、確認範囲で新価格は0。関連branchの後続結果を保存し、同候補・同軸の重複が疑われる場合は投入せず解決する。未参照archive/hostの不存在は主張しない。旧raw/STOPや他担当worktreeを変更しない。

最終26軸は、受入済み直接値のある軸を優先し、残る同actual幅の軸だけ凍結モデルを適用する。想定は保存early10＋新cal4＋新holdout2＝直接16軸、残るm7/m8/m10/m11/m12の10軸が比較用モデル。m7/m8は観測q範囲内での補間的利用、m10..m12/q160..640は未検証外挿として区別する。型を直接compiled証拠へ昇格しない。

Q3は8855f420aのC_RESULT_JOINを引用し、新Q3は0。新allocation ID B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01、X/Y各軸 [37,34,32,29,27,24,22,19,17,14,11,9,6] を固定。各軸N/価格/N×価格/sourceをexactで保存し、26軸総額・直接部分・モデル部分と割合を計算する。比較差は粗幅A_SHAPE総額−固定比較値で統一する。固定値は同じA_SHAPE現幅Y4=444807010、現幅p7/Y4最安=435641568、今回指定のp7全体最安=395962678 RZ。これらの配分・Q3・費用は再評価しない。

成果は条件付き価格モデルによる費用比較であり、厳密上界/CI・物理的global RMSE・最小N・G_qualified・winnerの認定ではない。新規上限6 compile、信号/Q3/他配分0。重大実装gate失敗・source/scalar不整合・OOM/保存不能では部分結果を保存して新規投入STOP、科学retry/軸補完/閾値緩和なし。最後はcommit/push・remote SHA/tree/成果物照合後mandatory STOP。ロードマップ本体は変更せずHANDOFFで差分を渡す。
